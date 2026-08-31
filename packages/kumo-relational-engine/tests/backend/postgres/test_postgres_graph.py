# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import uuid
from collections.abc import Generator
from typing import Any

import pytest
from kumo_relational_engine.rfm import Graph

try:
    from kumo_connectors import connect, read
    from kumo_relational_engine.rfm.backend.postgres import PostgresSampler
except ImportError:
    pytest.skip("'postgres' extension not installed", allow_module_level=True)


class _RecordingCursor:
    def __init__(self, cursor: Any, statements: list[str]) -> None:
        self._cursor = cursor
        self._statements = statements

    def __enter__(self) -> _RecordingCursor:
        self._cursor.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self._cursor.__exit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)

    def execute(self, statement: str, *args: Any, **kwargs: Any) -> Any:
        self._statements.append(statement)
        self._cursor.execute(statement, *args, **kwargs)
        return self


class _RecordingConnection:
    def __init__(self, connection: Any) -> None:
        self._connection = connection
        self.statements: list[str] = []

    def cursor(self, *args: Any, **kwargs: Any) -> _RecordingCursor:
        return _RecordingCursor(
            self._connection.cursor(*args, **kwargs), self.statements
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


@pytest.fixture()
def postgres_schema() -> Generator[tuple[str, str], None, None]:
    r"""A disposable integration schema when a PostgreSQL DSN is supplied."""
    dsn = os.getenv('KUMO_POSTGRES_TEST_DSN')
    if not dsn:
        pytest.skip('KUMO_POSTGRES_TEST_DSN is not set')

    schema = f'kumo_test_{uuid.uuid4().hex}'
    connection = connect('postgres', dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f'CREATE SCHEMA "{schema}"')
            cursor.execute(
                f'CREATE TABLE "{schema}".customers ('
                'customer_id BIGINT PRIMARY KEY, name TEXT NOT NULL)'
            )
            cursor.execute(
                f'CREATE TABLE "{schema}".orders ('
                'order_id BIGINT PRIMARY KEY, '
                'customer_id BIGINT NOT NULL REFERENCES '
                f'"{schema}".customers(customer_id), '
                'ordered_at TIMESTAMP NOT NULL)'
            )
            cursor.execute(
                f'INSERT INTO "{schema}".customers '
                "SELECT value, 'customer-' || value "
                'FROM generate_series(1, 20) AS value'
            )
            cursor.execute(
                f'INSERT INTO "{schema}".orders '
                'SELECT value, 1 + (value % 20), '
                "TIMESTAMP '2025-01-01' + value * INTERVAL '1 day' "
                'FROM generate_series(1, 100) AS value'
            )
        connection.commit()
    finally:
        connection.close()

    try:
        yield dsn, schema
    finally:
        connection = connect('postgres', dsn)
        try:
            with connection.cursor() as cursor:
                cursor.execute(f'DROP SCHEMA "{schema}" CASCADE')
            connection.commit()
        finally:
            connection.close()


def test_from_postgres_discovers_constraints_and_pushes_down_sampling(
    postgres_schema: tuple[str, str],
) -> None:
    dsn, schema = postgres_schema
    flat_table = read(
        'postgres',
        conninfo=dsn,
        table=f'{schema}.customers',
    )
    assert len(flat_table) == 20

    graph = Graph.from_postgres(
        connection=dsn,
        schema=schema,
        verbose=False,
    )
    try:
        assert graph.backend == 'postgres'
        assert set(graph.tables) == {'customers', 'orders'}
        assert len(graph.edges) == 1

        recording = _RecordingConnection(graph._connection)
        for table in graph.tables.values():
            table._connection = recording
        sampler = PostgresSampler(graph, verbose=False)
        columns = {column.name for column in graph['customers'].columns}
        sample = sampler._sample_entity_table(
            'customers', columns, num_rows=7, random_seed=42
        )

        assert len(sample) == 7
        sampling_sql = [
            sql for sql in recording.statements if 'ORDER BY random()' in sql
        ]
        assert len(sampling_sql) == 1
        assert 'LIMIT 7' in sampling_sql[0]
    finally:
        assert graph._connection is not None
        graph._connection.close()
        graph._connection = None

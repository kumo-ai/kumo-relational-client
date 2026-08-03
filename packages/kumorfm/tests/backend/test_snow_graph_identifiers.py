# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Warehouse identifiers passed to the Snowflake graph constructors must be
quoted, not interpolated. See
``bugs/security-sql-injection-interpolated-identifiers.md``.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip('snowflake.connector')
pytest.importorskip('yaml')

from snowflake.connector import SnowflakeConnection  # noqa: E402

import kumorfm.rfm as rfm  # noqa: E402

_INJECTION = "X' OR 1=1 --"
_SEMANTIC_VIEW_YAML = 'tables: []\n'


class _Cursor:
    def __init__(self, connection: '_Connection') -> None:
        self._connection = connection

    def __enter__(self) -> '_Cursor':
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, sql: str, parameters: Any = None) -> None:
        self._connection.calls.append((sql, parameters))

    def fetchone(self) -> tuple:
        return (self._connection.scalar, )

    def fetchall(self) -> list:
        return []


class _Connection(SnowflakeConnection):
    def __init__(self, scalar: Any = None) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.scalar = scalar

    def cursor(self) -> _Cursor:  # type: ignore[override]
        return _Cursor(self)


def test_from_snowflake_quotes_database_and_schema() -> None:
    connection = _Connection()

    # The fake connection discovers no tables, which the constructor now
    # rejects; the statement under test has already been issued at that point:
    with pytest.raises(ValueError, match='No tables found'):
        rfm.Graph.from_snowflake(
            connection,
            database='MY_DB',
            schema='MY_SCHEMA',
            verbose=False,
        )

    sql, _ = connection.calls[0]
    assert 'FROM "MY_DB".INFORMATION_SCHEMA.TABLES' in sql
    assert "WHERE TABLE_SCHEMA = 'MY_SCHEMA'" in sql


def test_from_snowflake_neutralises_an_injected_schema() -> None:
    connection = _Connection()

    with pytest.raises(ValueError, match='No tables found'):
        rfm.Graph.from_snowflake(
            connection,
            database='MY_DB',
            schema=_INJECTION,
            verbose=False,
        )

    sql, _ = connection.calls[0]
    assert "WHERE TABLE_SCHEMA = 'X'' OR 1=1 --'" in sql
    assert "'X' OR 1=1 --'" not in sql


def test_from_snowflake_neutralises_an_injected_database() -> None:
    connection = _Connection()

    with pytest.raises(ValueError, match='No tables found'):
        rfm.Graph.from_snowflake(
            connection,
            database='MY_DB". INFORMATION_SCHEMA.TABLES; DROP TABLE T; --',
            schema='MY_SCHEMA',
            verbose=False,
        )

    sql, _ = connection.calls[0]
    assert ('FROM "MY_DB"". INFORMATION_SCHEMA.TABLES; DROP TABLE T; --"'
            '.INFORMATION_SCHEMA.TABLES') in sql


def test_from_snowflake_semantic_view_quotes_the_view_name() -> None:
    connection = _Connection(scalar=_SEMANTIC_VIEW_YAML)

    # The view definition is empty, so the graph fails validation afterwards;
    # the statement under test has already been issued at that point:
    with pytest.raises(ValueError, match='At least one table'):
        rfm.Graph.from_snowflake_semantic_view(
            "MY_VIEW' OR 1=1 --",
            connection=connection,
            verbose=False,
        )

    sql, _ = connection.calls[0]
    assert ("SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW('MY_VIEW'' OR 1=1 --')"
            in sql)
    assert "'MY_VIEW' OR 1=1 --'" not in sql

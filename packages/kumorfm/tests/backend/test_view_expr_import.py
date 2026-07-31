# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Column expressions lifted out of a semantic/metric view definition run as
SQL under the caller's warehouse role, so a hostile view definition must not
reach the sampler. See
``bugs/security-column-expr-executes-verbatim-warehouse-sql.md``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from kumorfm.api.typing import Dtype

from kumorfm.rfm.base import SourceColumn, SourceForeignKey
from kumorfm.rfm.graph import _unsafe_expr_reason

_EXFILTRATION = '(SELECT MAX(SALARY) FROM PAYROLL)'


@pytest.mark.parametrize('expr', [
    _EXFILTRATION,
    'NAME; DROP TABLE USERS',
    'NAME -- trailing comment',
    'NAME /* block */',
    'CASE WHEN 1=1 THEN (SELECT 1) ELSE NAME END',
    'name) AS x, (SELECT 1',
])
def test_unsafe_expressions_are_reported(expr):
    assert _unsafe_expr_reason(expr) is not None


@pytest.mark.parametrize('expr', [
    "'x' UNION SELECT SECRET FROM PAYROLL",
    "COALESCE(NAME, 'none') || (SELECT MAX(SALARY) FROM PAYROLL)",
    '"drop" || (SELECT 1)',
    "CASE WHEN NOTE = 'select' THEN (SELECT 1) ELSE 0 END",
])
def test_keywords_outside_quotes_are_still_reported(expr):
    r"""Ignoring quoted text must not become a way to smuggle a sub-query in:
    an expression that also carries a quoted literal is still refused when a
    keyword sits outside the quotes.
    """
    assert _unsafe_expr_reason(expr) is not None


@pytest.mark.parametrize('expr', [
    'NAME',
    '"my name"',
    'DB.PUBLIC.USERS.NAME',
    'UPPER(NAME)',
    'CAST(CREATED_AT AS DATE)',
    'QUANTITY * UNIT_PRICE_USD',
    "CASE WHEN SEGMENT = 'SMB' THEN 1 ELSE 0 END",
    "CONCAT(FIRST_NAME, ' ', LAST_NAME)",
    'DATE_TRUNC(month, CREATED_AT)',
    'COALESCE(OFFSET_DAYS, 0)',
])
def test_ordinary_scalar_expressions_are_accepted(expr):
    assert _unsafe_expr_reason(expr) is None


@pytest.mark.parametrize('expr', [
    "CASE WHEN STATUS = 'select' THEN 1 ELSE 0 END",
    "CONCAT('drop', NAME)",
    "COALESCE(REASON, 'no merge')",
    "CASE WHEN NOTE = 'update pending' THEN 1 ELSE 0 END",
    '"drop"',
    '`delete`',
    "IFF(ACTION = 'delete', 1, 0)",
])
def test_keywords_inside_quoted_text_are_accepted(expr):
    r"""A keyword inside a string literal or a quoted identifier is data, not
    SQL, so refusing it would drop a legitimate column.
    """
    assert _unsafe_expr_reason(expr) is None


# Snowflake semantic views ####################################################

snowflake_connector = pytest.importorskip('snowflake.connector')
pytest.importorskip('yaml')

SnowflakeConnection = snowflake_connector.SnowflakeConnection

_SEMANTIC_VIEW = """
tables:
  - name: USERS
    base_table:
      database: DB
      schema: PUBLIC
      table: USERS
    primary_key:
      columns:
        - USER_ID
    dimensions:
      - name: USER_ID
        expr: USER_ID
        data_type: NUMBER
      - name: NAME_UPPER
        expr: UPPER(NAME)
        data_type: VARCHAR
{extra}"""

_LEAKED_DIMENSION = f"""      - name: LEAKED
        expr: {_EXFILTRATION}
        data_type: NUMBER
"""

_SOURCE_COLUMNS = [
    SourceColumn(name='USER_ID', dtype=Dtype.int64, is_primary_key=True,
                 is_unique_key=True, is_nullable=False),
    SourceColumn(name='NAME', dtype=Dtype.string, is_primary_key=False,
                 is_unique_key=False, is_nullable=True),
]


class _Cursor:
    def __init__(self, connection: '_Connection') -> None:
        self._connection = connection

    def __enter__(self) -> '_Cursor':
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, sql: str, parameters: Any = None) -> None:
        self._connection.calls.append(sql)

    def fetchone(self) -> tuple:
        return (self._connection.scalar, )

    def fetchall(self) -> list:
        return []


class _Connection(SnowflakeConnection):
    def __init__(self, scalar: Any = None) -> None:
        self.calls: list[str] = []
        self.scalar = scalar

    def cursor(self) -> _Cursor:  # type: ignore[override]
        return _Cursor(self)


def _stub_table_class(base):
    r"""A table class that answers the source-metadata probes offline, so the
    view-to-graph conversion runs without a warehouse.
    """

    class _StubTable(base):  # type: ignore[valid-type,misc]
        def _get_source_columns(self) -> list[SourceColumn]:
            return list(_SOURCE_COLUMNS)

        def _get_source_foreign_keys(self) -> list[SourceForeignKey]:
            return []

        def _get_source_sample_df(self) -> pd.DataFrame:
            return pd.DataFrame({
                'USER_ID': [1, 2, 3],
                'NAME': ['a', 'b', 'c'],
            })

        def _get_expr_sample_df(self, columns) -> pd.DataFrame:
            return pd.DataFrame({
                column.name: ['x', 'y', 'z']
                for column in columns
            })

        def _get_num_rows(self) -> int | None:
            return 3

    return _StubTable


@pytest.fixture()
def snow_graph(monkeypatch):
    import kumorfm.rfm.backend.snow as snow

    monkeypatch.setattr(snow, 'SnowTable', _stub_table_class(snow.SnowTable))

    from kumorfm.rfm import Graph

    def build(extra: str = ''):
        return Graph.from_snowflake_semantic_view(
            'MY_VIEW',
            connection=_Connection(
                scalar=_SEMANTIC_VIEW.format(extra=extra)),
            verbose=False,
        )

    return build


def test_semantic_view_expression_is_not_executed(snow_graph):
    with pytest.warns(UserWarning, match='LEAKED'):
        graph = snow_graph(_LEAKED_DIMENSION)

    table = graph['USERS']
    assert not table.has_column('LEAKED')
    assert [
        str(column.expr) for column in table.columns
        if column.expr is not None
    ] == ['UPPER(NAME)']


def test_semantic_view_keeps_ordinary_expression_columns(snow_graph):
    graph = snow_graph()

    assert str(graph['USERS']['NAME_UPPER'].expr) == 'UPPER(NAME)'

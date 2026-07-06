from __future__ import annotations

import re
from typing import Any

import pandas as pd

from nvidia_sdfm.errors import MissingExtraError, SdfmError


def read(source: str, **kwargs: Any) -> pd.DataFrame:
    reader = _READERS.get(source)
    if reader is None:
        raise SdfmError(
            f'Unknown connector {source!r}; supported: {sorted(_READERS)}',
            code='UNKNOWN_CONNECTOR',
        )
    return reader(**kwargs)


_TABLE_IDENTIFIER_RE = re.compile(
    r'^[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*)*$',
)


def _resolve_sql(*, table: str | None, query: str | None) -> str:
    if (table is None) == (query is None):
        raise SdfmError(
            "exactly one of 'table' or 'query' must be provided",
            code='INVALID_CONNECTOR_ARGS',
        )
    if query is not None:
        return query
    if not _TABLE_IDENTIFIER_RE.fullmatch(table):
        raise SdfmError(
            f'invalid table identifier {table!r}',
            code='INVALID_CONNECTOR_ARGS',
            details={'table': table},
        )
    return f'SELECT * FROM {table}'


def _read_local(
    *,
    data: pd.DataFrame | dict[str, Any] | None = None,
    path: str | None = None,
) -> pd.DataFrame:
    if data is not None:
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if path is None:
        raise SdfmError(
            "local connector requires either 'data' or 'path'",
            code='INVALID_CONNECTOR_ARGS',
        )
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _read_sqlite(
    *,
    database: str,
    table: str | None = None,
    query: str | None = None,
) -> pd.DataFrame:
    import sqlite3

    sql = _resolve_sql(table=table, query=query)
    with sqlite3.connect(database) as connection:
        return pd.read_sql_query(sql, connection)


def _read_duckdb(
    *,
    database: str = ':memory:',
    table: str | None = None,
    query: str | None = None,
) -> pd.DataFrame:
    try:
        import duckdb
    except ImportError as error:
        raise MissingExtraError('duckdb', 'duckdb') from error
    sql = _resolve_sql(table=table, query=query)
    with duckdb.connect(database) as connection:
        return connection.execute(sql).fetchdf()


def _read_snowflake(
    *,
    table: str | None = None,
    query: str | None = None,
    **connection_kwargs: Any,
) -> pd.DataFrame:
    try:
        import snowflake.connector
    except ImportError as error:
        raise MissingExtraError('snowflake', 'snowflake-connector-python') from error
    sql = _resolve_sql(table=table, query=query)
    with snowflake.connector.connect(**connection_kwargs) as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(sql)
            return cursor.fetch_pandas_all()
        finally:
            cursor.close()


_READERS = {
    'local': _read_local,
    'sqlite': _read_sqlite,
    'duckdb': _read_duckdb,
    'snowflake': _read_snowflake,
}

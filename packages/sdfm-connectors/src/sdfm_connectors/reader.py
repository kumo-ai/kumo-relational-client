from __future__ import annotations

from typing import Any

import pandas as pd

from sdfm_connectors.backends import connect, owns_connection
from sdfm_connectors.sql import ConnectorError, resolve_sql

_SQL_SOURCES = ('sqlite', 'duckdb', 'snowflake', 'databricks')


def read(source: str, **kwargs: Any) -> pd.DataFrame:
    if source == 'local':
        return _read_local(**kwargs)
    if source not in _SQL_SOURCES:
        raise ConnectorError(
            f'Unknown connector {source!r}; supported: '
            f'{sorted(("local", *_SQL_SOURCES))}',
            code='UNKNOWN_CONNECTOR',
        )

    table = kwargs.pop('table', None)
    query = kwargs.pop('query', None)
    if source in ('sqlite', 'duckdb'):
        uri = kwargs.pop('database', None)
        uri = kwargs.pop('uri', uri)
        connection = connect(source, uri, **kwargs)
    else:
        connection = connect(source, **kwargs)

    try:
        return read_table(connection, table=table, query=query)
    finally:
        if owns_connection(connection):
            connection.close()


def read_table(
    connection: Any,
    *,
    table: str | None = None,
    query: str | None = None,
) -> pd.DataFrame:
    sql = resolve_sql(table=table, query=query)
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        if hasattr(cursor, 'fetch_pandas_all'):
            return cursor.fetch_pandas_all()
        if hasattr(cursor, 'fetchdf'):
            return cursor.fetchdf()
        if hasattr(cursor, 'fetch_arrow_table'):
            return cursor.fetch_arrow_table().to_pandas()
        if hasattr(cursor, 'fetchall_arrow'):
            return cursor.fetchall_arrow().to_pandas()
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        return pd.DataFrame(rows, columns=columns)
    finally:
        cursor.close()


def _read_local(
    *,
    data: pd.DataFrame | dict[str, Any] | None = None,
    path: str | None = None,
) -> pd.DataFrame:
    if data is not None and path is not None:
        raise ConnectorError(
            "local connector accepts exactly one of 'data' or 'path', "
            "not both",
            code='INVALID_CONNECTOR_ARGS',
        )
    if data is not None:
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if path is None:
        raise ConnectorError(
            "local connector requires either 'data' or 'path'",
            code='INVALID_CONNECTOR_ARGS',
        )
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)

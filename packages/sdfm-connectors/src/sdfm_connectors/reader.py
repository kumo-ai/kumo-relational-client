from __future__ import annotations

from contextlib import suppress
from typing import Any

import pandas as pd

from sdfm_connectors.backends import connect, owns_connection
from sdfm_connectors.sql import (
    ConnectorError,
    driver_guard,
    require_driver,
    resolve_sql,
)

_SQL_SOURCES = ('sqlite', 'duckdb', 'snowflake', 'databricks')
_FILE_SOURCES = ('local', 's3')


def read(source: str, **kwargs: Any) -> pd.DataFrame:
    if source == 'local':
        return _read_local(**kwargs)
    if source == 's3':
        return _read_s3(**kwargs)
    if source not in _SQL_SOURCES:
        raise ConnectorError(
            f'Unknown connector {source!r}; supported: '
            f'{sorted((*_FILE_SOURCES, *_SQL_SOURCES))}',
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
            with suppress(Exception):
                connection.close()


def read_table(
    connection: Any,
    *,
    table: str | None = None,
    query: str | None = None,
) -> pd.DataFrame:
    sql = resolve_sql(table=table, query=query)
    with driver_guard('QUERY_FAILED', 'query execution failed', sql=sql):
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
            with suppress(Exception):
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
        if isinstance(data, pd.DataFrame):
            return data
        try:
            return pd.DataFrame(data)
        except (ValueError, TypeError) as error:
            raise ConnectorError(
                f'local connector could not build a DataFrame from '
                f"'data': {error}",
                code='INVALID_CONNECTOR_ARGS',
            ) from error
    if path is None:
        raise ConnectorError(
            "local connector requires either 'data' or 'path'",
            code='INVALID_CONNECTOR_ARGS',
        )
    return _read_file(path)


def _read_s3(
    *,
    path: str | None = None,
    storage_options: dict[str, Any] | None = None,
    **extra: Any,
) -> pd.DataFrame:
    if extra:
        raise ConnectorError(
            f's3 connector got unexpected arguments {sorted(extra)}; '
            "supported: 'path' and 'storage_options'",
            code='INVALID_CONNECTOR_ARGS',
            details={'arguments': sorted(extra)},
        )
    if path is None:
        raise ConnectorError(
            "s3 connector requires 'path' (an s3:// object URI)",
            code='INVALID_CONNECTOR_ARGS',
        )
    if not path.startswith('s3://'):
        raise ConnectorError(
            f's3 connector requires an s3:// URI, got {path!r}',
            code='INVALID_CONNECTOR_ARGS',
            details={'path': path},
        )
    require_driver('s3', 's3fs', 's3fs')
    return _read_file(path, storage_options=storage_options)


def _read_file(
    path: str,
    storage_options: dict[str, Any] | None = None,
) -> pd.DataFrame:
    with driver_guard('READ_FAILED', f'failed to read {path!r}',
                      missing_as_not_found=True, path=path):
        if path.lower().endswith('.parquet') or path.endswith('/'):
            return pd.read_parquet(path, storage_options=storage_options)
        return pd.read_csv(path, storage_options=storage_options)

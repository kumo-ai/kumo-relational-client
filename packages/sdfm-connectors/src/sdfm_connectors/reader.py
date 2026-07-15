from __future__ import annotations

import importlib
from typing import Any

import pandas as pd

from sdfm_connectors.backends import connect, owns_connection
from sdfm_connectors.sql import (
    ConnectorError,
    MissingBackendError,
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
    try:
        importlib.import_module('s3fs')
    except ModuleNotFoundError as error:
        if error.name != 's3fs':
            raise
        raise MissingBackendError('s3', 's3fs') from error
    return _read_file(path, storage_options=storage_options)


def _read_file(
    path: str,
    storage_options: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if path.lower().endswith('.parquet') or path.endswith('/'):
        return pd.read_parquet(path, storage_options=storage_options)
    return pd.read_csv(path, storage_options=storage_options)

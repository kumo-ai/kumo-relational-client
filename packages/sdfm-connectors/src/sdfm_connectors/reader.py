# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from contextlib import suppress
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import pyarrow as pa

from sdfm_connectors.backends import connect, owns_connection
from sdfm_connectors.sql import (
    ConnectorError,
    driver_guard,
    require_driver,
    resolve_sql,
)

_SQL_SOURCES = ('sqlite', 'duckdb', 'snowflake', 'databricks')
_FILE_SOURCES = ('local', 's3')

_PARQUET_SUFFIXES = ('.parquet', '.pq', '.parq')
_CSV_SUFFIXES = ('.csv', '.txt')
_COMPRESSION_SUFFIXES = ('.gz', '.bz2', '.zip', '.xz', '.zst', '.tar')
_FILE_FORMATS = ('csv', 'parquet')

_NULLABLE_INTEGER_DTYPES = {
    pa.int8(): pd.Int8Dtype(),
    pa.int16(): pd.Int16Dtype(),
    pa.int32(): pd.Int32Dtype(),
    pa.int64(): pd.Int64Dtype(),
    pa.uint8(): pd.UInt8Dtype(),
    pa.uint16(): pd.UInt16Dtype(),
    pa.uint32(): pd.UInt32Dtype(),
    pa.uint64(): pd.UInt64Dtype(),
}


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
        if 'database' in kwargs and 'uri' in kwargs:
            raise ConnectorError(
                f"{source} connector accepts exactly one of 'database' or "
                f"'uri', not both",
                code='INVALID_CONNECTOR_ARGS',
            )
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


def _query_handle(connection: Any) -> tuple[Any, bool]:
    r"""Return the ``(handle, owned)`` to run a query on.

    ``DuckDBPyConnection.cursor()`` hands back a *new independent connection*
    rather than a cursor on the same session, so the caller's temp tables and
    registered DataFrames are invisible to it. A driver whose ``cursor()``
    yields an object of the connection's own type is duplicating the
    connection, so the query runs on the handle the caller passed in — and only
    a genuine cursor we created is ours to close.
    """
    cursor = connection.cursor()
    if isinstance(cursor, type(connection)):
        with suppress(Exception):
            cursor.close()
        return connection, False
    return cursor, True


def read_table(
    connection: Any,
    *,
    table: str | None = None,
    query: str | None = None,
) -> pd.DataFrame:
    sql = resolve_sql(table=table, query=query)
    with driver_guard('QUERY_FAILED', 'query execution failed', sql=sql):
        cursor, owned = _query_handle(connection)
        try:
            cursor.execute(sql)
            if hasattr(cursor, 'fetch_arrow_all'):
                arrow_table = cursor.fetch_arrow_all()
                if arrow_table is not None:
                    return _arrow_to_pandas(arrow_table)
            if hasattr(cursor, 'fetch_pandas_all'):
                return cursor.fetch_pandas_all()
            if hasattr(cursor, 'fetchdf'):
                return cursor.fetchdf()
            if hasattr(cursor, 'fetch_arrow_table'):
                return _arrow_to_pandas(cursor.fetch_arrow_table())
            if hasattr(cursor, 'fetchall_arrow'):
                return _arrow_to_pandas(cursor.fetchall_arrow())
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return pd.DataFrame(rows, columns=columns)
        finally:
            if owned:
                with suppress(Exception):
                    cursor.close()


def _arrow_to_pandas(arrow_table: Any) -> pd.DataFrame:
    r"""Convert an Arrow table without widening null-bearing integers.

    The numpy default represents nulls in an integer column by promoting it to
    ``float64``, which silently rounds values beyond 2**53. Such columns use a
    null-aware integer dtype instead, matching what DuckDB already returns.
    Integer columns without nulls keep their numpy dtype.
    """
    df = arrow_table.to_pandas(types_mapper=_NULLABLE_INTEGER_DTYPES.get)
    for position, dtype in enumerate(df.dtypes):
        if (isinstance(dtype, pd.api.extensions.ExtensionDtype)
                and pd.api.types.is_integer_dtype(dtype)):
            column = df.iloc[:, position]
            if not column.isna().any():
                df.isetitem(position, column.astype(dtype.numpy_dtype))
    return df


def _read_local(
    *,
    data: pd.DataFrame | dict[str, Any] | None = None,
    path: str | None = None,
    format: str | None = None,
    **extra: Any,
) -> pd.DataFrame:
    if extra:
        raise ConnectorError(
            f'local connector got unexpected arguments {sorted(extra)}; '
            "supported: 'data', 'path' and 'format'",
            code='INVALID_CONNECTOR_ARGS',
            details={'arguments': sorted(extra)},
        )
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
    _require_local_path(path)
    return _read_file(path, format=format)


def _require_local_path(path: str) -> None:
    r"""Reject a URI handed to the connector named ``local``.

    ``pandas`` resolves any fsspec/urllib-supported URL, so without this the
    'safe, no-network' source reaches the network and returns the response as
    a DataFrame. A single-character scheme is a Windows drive letter, not a
    URI. A ``file://`` URI naming a host is refused too: ``urllib`` drops the
    authority and reads the path locally, so the URI does not mean what it
    says. See ``bugs/security-local-connector-fetches-arbitrary-urls.md``.
    """
    parsed = urlparse(path)
    if len(parsed.scheme) > 1 and parsed.scheme != 'file':
        raise ConnectorError(
            f'the local connector reads local filesystem paths, got a '
            f'{parsed.scheme!r} URI: {path!r}. Use the matching connector, '
            f"e.g. source='s3'.",
            code='INVALID_CONNECTOR_ARGS',
            details={'path': path},
        )
    if parsed.scheme == 'file' and parsed.netloc.lower() not in ('',
                                                                'localhost'):
        raise ConnectorError(
            f'the local connector reads local filesystem paths, got a '
            f'file:// URI naming the host {parsed.netloc!r}: {path!r}',
            code='INVALID_CONNECTOR_ARGS',
            details={'path': path},
        )


def _read_s3(
    *,
    path: str | None = None,
    storage_options: dict[str, Any] | None = None,
    format: str | None = None,
    **extra: Any,
) -> pd.DataFrame:
    if extra:
        raise ConnectorError(
            f's3 connector got unexpected arguments {sorted(extra)}; '
            "supported: 'path', 'storage_options' and 'format'",
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
    return _read_file(path, storage_options=storage_options, format=format)


def _resolve_format(path: str) -> str:
    r"""Decide the file format from ``path``, or reject it.

    Falling back to CSV for any unrecognised suffix does not fail loudly: a
    ``.json`` file parses into junk column names and zero rows, and a ``.tsv``
    into a single column. Only the documented formats are read, and a directory
    (with or without a trailing separator) is a Parquet dataset.
    """
    if path.endswith(('/', os.sep)) or os.path.isdir(path):
        return 'parquet'
    suffix = PurePosixPath(path.lower()).suffix
    if suffix in _COMPRESSION_SUFFIXES:
        suffix = PurePosixPath(path.lower()).suffixes[-2:-1]
        suffix = suffix[0] if suffix else ''
    if suffix in _PARQUET_SUFFIXES:
        return 'parquet'
    if suffix in _CSV_SUFFIXES:
        return 'csv'
    raise ConnectorError(
        f'unsupported file format {suffix or path!r} for {path!r}; supported: '
        f'{sorted(_PARQUET_SUFFIXES + _CSV_SUFFIXES)} (optionally compressed) '
        f'and Parquet directories. Pass format="csv" or format="parquet" to '
        f'read a file whose name does not carry one of these suffixes.',
        code='INVALID_CONNECTOR_ARGS',
        details={'path': path, 'suffix': suffix},
    )


def _read_file(
    path: str,
    storage_options: dict[str, Any] | None = None,
    format: str | None = None,
) -> pd.DataFrame:
    if format is None:
        resolved = _resolve_format(path)
    elif format in _FILE_FORMATS:
        resolved = format
    else:
        raise ConnectorError(
            f'unknown format {format!r}; supported: {sorted(_FILE_FORMATS)}',
            code='INVALID_CONNECTOR_ARGS',
            details={'format': format},
        )
    with driver_guard('READ_FAILED', f'failed to read {path!r}',
                      missing_as_not_found=True, path=path):
        if resolved == 'parquet':
            return pd.read_parquet(path, storage_options=storage_options)
        return pd.read_csv(path, storage_options=storage_options)

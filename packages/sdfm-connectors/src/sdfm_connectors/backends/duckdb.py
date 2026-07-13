from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

from sdfm_connectors.sql import MissingBackendError

try:
    import duckdb
except ModuleNotFoundError as error:
    if error.name != 'duckdb':
        raise
    raise MissingBackendError('duckdb', 'duckdb') from error

Connection: TypeAlias = duckdb.DuckDBPyConnection


def connect(uri: str | Path | None = None, **kwargs: Any) -> Connection:
    return duckdb.connect(str(uri) if uri is not None else ':memory:', **kwargs)

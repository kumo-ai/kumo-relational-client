from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

from sdfm_connectors.sql import MissingBackendError

try:
    import adbc_driver_sqlite.dbapi as adbc
except ModuleNotFoundError as error:
    if error.name != 'adbc_driver_sqlite':
        raise
    raise MissingBackendError('sqlite', 'adbc-driver-sqlite') from error

Connection: TypeAlias = adbc.AdbcSqliteConnection


def connect(uri: str | Path | None = None, **kwargs: Any) -> Connection:
    return adbc.connect(str(uri) if uri is not None else None, **kwargs)

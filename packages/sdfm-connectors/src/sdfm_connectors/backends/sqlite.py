from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

from sdfm_connectors.sql import require_driver

adbc = require_driver(
    'sqlite', 'adbc-driver-sqlite', 'adbc_driver_sqlite.dbapi',
)

Connection: TypeAlias = adbc.AdbcSqliteConnection


def connect(uri: str | Path | None = None, **kwargs: Any) -> Connection:
    return adbc.connect(str(uri) if uri is not None else None, **kwargs)

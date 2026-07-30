# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

from sdfm_connectors.sql import ConnectorError, require_driver

duckdb = require_driver('duckdb', 'duckdb', 'duckdb')

Connection: TypeAlias = duckdb.DuckDBPyConnection


def connect(
    uri: str | Path | None = None,
    *,
    database: str | Path | None = None,
    **kwargs: Any,
) -> Connection:
    r"""Open a DuckDB database. ``database`` is an alias for ``uri``."""
    if uri is not None and database is not None:
        raise ConnectorError(
            "duckdb connector accepts exactly one of 'database' or 'uri', "
            'not both',
            code='INVALID_CONNECTOR_ARGS',
        )
    uri = uri if uri is not None else database
    return duckdb.connect(str(uri) if uri is not None else ':memory:', **kwargs)

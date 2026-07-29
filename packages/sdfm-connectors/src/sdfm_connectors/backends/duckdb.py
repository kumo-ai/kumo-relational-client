# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

from sdfm_connectors.sql import require_driver

duckdb = require_driver('duckdb', 'duckdb', 'duckdb')

Connection: TypeAlias = duckdb.DuckDBPyConnection


def connect(uri: str | Path | None = None, **kwargs: Any) -> Connection:
    return duckdb.connect(str(uri) if uri is not None else ':memory:', **kwargs)

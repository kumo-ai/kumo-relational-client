# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, TypeAlias

from sdfm_connectors.sql import require_existing_database

try:
    import adbc_driver_duckdb.dbapi as adbc
except ImportError:
    raise ImportError("No module named 'adbc_driver_duckdb'. Please install "
                      "KumoRFM with the 'duckdb' extension via "
                      "`pip install kumorfm[duckdb]`.")

Connection: TypeAlias = adbc.Connection


def connect(uri: str | Path | None = None, **kwargs: Any) -> Connection:
    r"""Opens a connection to a :class:`duckdb` database.

    DuckDB opens a missing path in create-if-missing mode, so a mistyped path
    would silently leave a new empty database in the caller's working tree and
    then fail as an empty graph. The shared existence guard is applied here as
    well as in the ``sdfm_connectors`` DuckDB backend, because this module is
    a separate entry point onto a different driver (``adbc_driver_duckdb``,
    which the sampler needs) and so does not inherit that one.

    uri: The path to the database file to be opened, or ``None`` for an
        in-memory database.
    kwargs: Additional connection arguments, following the
        :class:`adbc_driver_duckdb` protocol.
    """
    if uri is None:
        return adbc.connect(None, **kwargs)
    uri = str(uri)
    require_existing_database('duckdb', uri)
    return adbc.connect(uri, **kwargs)


from .table import DuckDBTable  # noqa: E402
from .sampler import DuckDBSampler  # noqa: E402

__all__ = [
    'connect',
    'Connection',
    'DuckDBTable',
    'DuckDBSampler',
]

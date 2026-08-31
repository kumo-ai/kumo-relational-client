# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pandas as pd

from kumo_relational_client import relational
from kumo_relational_client._version import __version__
from kumo_relational_client.base import ModelCapabilities
from kumo_relational_client.client import RelationalClient
from kumo_relational_client.core.connectors import read as _read
from kumo_relational_client.errors import (
    MissingExtraError,
    NimRequestError,
    RelationalError,
    UnknownModelError,
)
from kumo_relational_client.models import (
    RelationalModel,
)

__all__ = [
    'MissingExtraError',
    'ModelCapabilities',
    'NimRequestError',
    'RelationalClient',
    'RelationalError',
    'RelationalModel',
    'UnknownModelError',
    '__version__',
    'read',
    'relational',
]


def read(source: str, **kwargs: Any) -> pd.DataFrame:
    r"""Read a table into a DataFrame. Endpoint-independent; needs no RelationalClient.

    Args:
        source: ``'local'`` (``data=`` a DataFrame/dict, or ``path=`` a CSV or
            Parquet file), ``'s3'`` (``path='s3://...'``, optional
            ``storage_options=``), or a SQL backend -- ``'sqlite'``,
            ``'duckdb'``, ``'snowflake'``, ``'databricks'``, ``'postgres'``.
            Each SQL backend needs its extra, e.g.
            ``pip install 'kumo-relational-client[duckdb]'``.
        **kwargs: For a SQL source, exactly one of ``table=`` / ``query=`` plus
            that backend's connection arguments.

    Returns:
        The rows as a ``pd.DataFrame``.

    Raises:
        MissingExtraError: If the source's optional driver is not installed.
        RelationalError: With a ``code`` of ``UNKNOWN_CONNECTOR``,
            ``INVALID_CONNECTOR_ARGS``, ``CONNECT_FAILED``, ``QUERY_FAILED``,
            ``READ_FAILED``, ``NOT_FOUND`` or ``DRIVER_LOAD_FAILED``.
    """
    return _read(source, **kwargs)

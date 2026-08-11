# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pandas as pd

from nemotron_predict import relational
from nemotron_predict._version import __version__
from nemotron_predict.base import ModelCapabilities
from nemotron_predict.client import PredictClient
from nemotron_predict.core.connectors import read as _read
from nemotron_predict.errors import (
    MissingExtraError,
    NimRequestError,
    PredictError,
    UnknownModelError,
)
from nemotron_predict.models import (
    RelationalModel,
    TabularModel,
)

__all__ = [
    'MissingExtraError',
    'ModelCapabilities',
    'NimRequestError',
    'PredictClient',
    'PredictError',
    'RelationalModel',
    'TabularModel',
    'UnknownModelError',
    '__version__',
    'read',
    'relational',
]


def read(source: str, **kwargs: Any) -> pd.DataFrame:
    r"""Read a table into a DataFrame. Endpoint-independent; needs no PredictClient.

    Args:
        source: ``'local'`` (``data=`` a DataFrame/dict, or ``path=`` a CSV or
            Parquet file), ``'s3'`` (``path='s3://...'``, optional
            ``storage_options=``), or a SQL backend -- ``'sqlite'``,
            ``'duckdb'``, ``'snowflake'``, ``'databricks'``. Each SQL backend
            needs its extra, e.g. ``pip install 'nemotron-predict-client[duckdb]'``.
        **kwargs: For a SQL source, exactly one of ``table=`` / ``query=`` plus
            that backend's connection arguments.

    Returns:
        The rows as a ``pd.DataFrame``.

    Raises:
        MissingExtraError: If the source's optional driver is not installed.
        PredictError: With a ``code`` of ``UNKNOWN_CONNECTOR``,
            ``INVALID_CONNECTOR_ARGS``, ``CONNECT_FAILED``, ``QUERY_FAILED``,
            ``READ_FAILED``, ``NOT_FOUND`` or ``DRIVER_LOAD_FAILED``.
    """
    return _read(source, **kwargs)

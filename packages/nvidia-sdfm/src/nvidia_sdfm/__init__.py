# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pandas as pd

from nvidia_sdfm import kumorfm
from nvidia_sdfm._version import __version__
from nvidia_sdfm.base import ModelCapabilities
from nvidia_sdfm.client import SDFMClient
from nvidia_sdfm.core.connectors import read as _read
from nvidia_sdfm.errors import SdfmError
from nvidia_sdfm.models import RFMModel, TabICLModel

__all__ = [
    'ModelCapabilities',
    'RFMModel',
    'SDFMClient',
    'SdfmError',
    'TabICLModel',
    '__version__',
    'kumorfm',
    'read',
]


def read(source: str, **kwargs: Any) -> pd.DataFrame:
    r"""Read a table into a DataFrame. Endpoint-independent; needs no SDFMClient.

    Args:
        source: ``'local'`` (``data=`` a DataFrame/dict, or ``path=`` a CSV or
            Parquet file), ``'s3'`` (``path='s3://...'``, optional
            ``storage_options=``), or a SQL backend -- ``'sqlite'``,
            ``'duckdb'``, ``'snowflake'``, ``'databricks'``. Each SQL backend
            needs its extra, e.g. ``pip install 'nvidia-sdfm[duckdb]'``.
        **kwargs: For a SQL source, exactly one of ``table=`` / ``query=`` plus
            that backend's connection arguments.

    Returns:
        The rows as a ``pd.DataFrame``.

    Raises:
        MissingExtraError: If the source's optional driver is not installed.
        SdfmError: With a ``code`` of ``UNKNOWN_CONNECTOR``,
            ``INVALID_CONNECTOR_ARGS``, ``CONNECT_FAILED``, ``QUERY_FAILED``,
            ``READ_FAILED``, ``NOT_FOUND`` or ``DRIVER_LOAD_FAILED``.
    """
    return _read(source, **kwargs)

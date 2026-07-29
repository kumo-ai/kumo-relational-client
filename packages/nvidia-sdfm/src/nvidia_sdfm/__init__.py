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
    'SDFMClient',
    'RFMModel',
    'TabICLModel',
    'ModelCapabilities',
    'read',
    'kumorfm',
    'SdfmError',
    '__version__',
]


def read(source: str, **kwargs: Any) -> pd.DataFrame:
    r"""Read a table into a DataFrame. Endpoint-independent; needs no SDFMClient."""
    return _read(source, **kwargs)

from __future__ import annotations

from typing import Any

import pandas as pd

from nvidia_sdfm import kumorfm
from nvidia_sdfm._version import __version__
from nvidia_sdfm.base import AdapterRegistry, ModelAdapter, ModelCapabilities
from nvidia_sdfm.client import SDFMClient
from nvidia_sdfm.core.connectors import read as _read
from nvidia_sdfm.errors import SdfmError
from nvidia_sdfm.requests import KumoRFMRequest, ModelRequest, TabICLRequest

__all__ = [
    'SDFMClient',
    'ModelRequest',
    'TabICLRequest',
    'KumoRFMRequest',
    'ModelCapabilities',
    'ModelAdapter',
    'AdapterRegistry',
    'read',
    'kumorfm',
    'SdfmError',
    '__version__',
]


def read(source: str, **kwargs: Any) -> pd.DataFrame:
    r"""Read a table into a DataFrame. Endpoint-independent; needs no SDFMClient."""
    return _read(source, **kwargs)

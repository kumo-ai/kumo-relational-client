from __future__ import annotations

from typing import Any

import pandas as pd

from nvidia_sdfm._version import __version__
from nvidia_sdfm.adapters import RFMAdapter, TabICLAdapter
from nvidia_sdfm.base import AdapterRegistry
from nvidia_sdfm.core.connectors import read as _read
from nvidia_sdfm.core.transport import TFMClient
from nvidia_sdfm.errors import NotInitializedError, SdfmError

__all__ = ['init', 'predict', 'read', 'registry', '__version__', 'SdfmError']

registry = AdapterRegistry()
registry.register(TabICLAdapter())
registry.register(RFMAdapter())

_client: TFMClient | None = None


def init(
    url: str,
    api_key: str | None = None,
    *,
    verify_ssl: bool = True,
) -> None:
    global _client
    _client = TFMClient(url, api_key, verify_ssl=verify_ssl)


def predict(model: str, **kwargs: Any) -> pd.DataFrame:
    if _client is None:
        raise NotInitializedError()
    adapter = registry.get(model)
    return adapter.predict(_client, **kwargs)


def read(source: str, **kwargs: Any) -> pd.DataFrame:
    return _read(source, **kwargs)

from __future__ import annotations

from types import TracebackType
from typing import Any

import pandas as pd

from nvidia_sdfm.base import AdapterRegistry, ModelCapabilities
from nvidia_sdfm.core.transport import Transport
from nvidia_sdfm.errors import SdfmError
from nvidia_sdfm.requests import ModelRequest


def _default_registry() -> AdapterRegistry:
    from nvidia_sdfm.adapters import KumoRFMAdapter, TabICLAdapter

    registry = AdapterRegistry()
    registry.register(TabICLAdapter())
    registry.register(KumoRFMAdapter())
    return registry


class SDFMClient:
    r"""A connection to one Universal TFM NIM.

    Each ``SDFMClient`` owns its own transport and adapter registry, so several
    clients can target different endpoints (or tenants) at once without sharing
    process-global state. Use it as a context manager, or call ``close()``.

    >>> from nvidia_sdfm import SDFMClient, TabICLRequest
    >>> with SDFMClient(url="http://localhost:8000") as client:
    ...     df = client.predict(TabICLRequest(context=ctx, predict=q,
    ...                                       task="classification", target="y"))
    """

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        *,
        verify_ssl: bool = True,
        timeout: float = 60.0,
        max_retries: int = 3,
        registry: AdapterRegistry | None = None,
    ) -> None:
        self._transport = Transport(
            url,
            api_key,
            verify_ssl=verify_ssl,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._registry = registry if registry is not None else _default_registry()

    @property
    def url(self) -> str:
        return self._transport.url

    def models(self) -> list[str]:
        r"""The model ids this client can serve."""
        return self._registry.names()

    def capabilities(self, model: str) -> ModelCapabilities:
        r"""What ``model`` supports (tasks, outputs, request type)."""
        return self._registry.get(model).capabilities()

    def health_ready(self) -> bool:
        return self._transport.health_ready()

    def predict(self, request: ModelRequest) -> pd.DataFrame:
        r"""Run a typed prediction request against the NIM."""
        adapter = self._registry.get(request.model)
        if not isinstance(request, adapter.request_type):
            raise SdfmError(
                f"model {request.model!r} expects a "
                f"{adapter.request_type.__name__}, got "
                f"{type(request).__name__}",
                code='INVALID_REQUEST',
            )
        return adapter.predict(self._transport, request)

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> SDFMClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f'SDFMClient(url={self.url!r}, models={self.models()})'

    def register(self, adapter: Any) -> None:
        r"""Add a model adapter to this client's registry."""
        self._registry.register(adapter)

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import TracebackType
from typing import Any

import pandas as pd

from nvidia_sdfm.base import (
    AdapterRegistry,
    ModelAdapter,
    ModelCapabilities,
    PredictResult,
    request_type_names,
)
from nvidia_sdfm.core.transport import Transport
from nvidia_sdfm.errors import SdfmError
from nvidia_sdfm.models import RFMModel, TabICLModel
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

    Run inference through a model handle:

    >>> from nvidia_sdfm import SDFMClient, kumorfm
    >>> with SDFMClient(url="http://localhost:8000") as client:
    ...     df = client.kumorfm(graph).predict("PREDICT ... FOR ...", [1, 2])
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
        r"""Opens a client against one NIM.

        No request is made here: the endpoint is first contacted by
        :meth:`health_ready` or by a prediction.

        Args:
            url: Base URL of the NIM, e.g. ``'http://localhost:8000'``. Must be
                ``http://`` or ``https://``.
            api_key: Sent as ``X-API-Key`` when an authenticating gateway
                fronts the NIM. Refused over plaintext ``http://`` to a
                non-local host.
            verify_ssl: Whether to verify the server's TLS certificate.
            timeout: Seconds to wait for each individual attempt -- not for the
                call as a whole. A call that exhausts ``max_retries`` can take
                up to ``(max_retries + 1) * timeout`` plus backoff.
            max_retries: Retries on transient failures (429, 500, 502, 503,
                504) with exponential backoff. ``0`` disables retrying.
            registry: The adapter registry to dispatch with. Defaults to the
                built-in TabICL and KumoRFM adapters.
        """
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
        r"""The NIM base URL this client was opened against."""
        return self._transport.url

    def models(self) -> list[str]:
        r"""The model ids this client can serve.

        Describes the client's own adapter registry, not the connected
        endpoint: a NIM serving only one of these models still reports both.
        """
        return self._registry.names()

    def capabilities(self, model: str) -> ModelCapabilities:
        r"""What ``model`` supports (tasks, outputs, request type).

        Like :meth:`models`, this describes the client-side adapter and is not
        read from the connected endpoint.
        """
        return self._registry.get(model).capabilities()

    def health_ready(self) -> bool:
        r"""Whether the NIM answers ``GET /v1/health/ready`` with 200.

        Raises:
            SdfmError: With code ``TRANSPORT_ERROR`` if the endpoint cannot be
                reached at all.
        """
        return self._transport.health_ready()

    def kumorfm(self, graph: Any) -> RFMModel:
        r"""A KumoRFM handle: ``client.kumorfm(graph).predict(query, ...)``.

        This is the supported way to run KumoRFM inference.
        """
        return RFMModel(self, graph)

    def tabicl(
        self,
        context: pd.DataFrame,
        *,
        target: str,
        task: str,
    ) -> TabICLModel:
        r"""A TabICL handle:
        ``client.tabicl(context, target=..., task=...).predict(rows)``.

        This is the supported way to run TabICL inference.
        """
        return TabICLModel(self, context, task=task, target=target)

    def _predict(self, request: ModelRequest) -> PredictResult:
        r"""Internal dispatch used by the model handles.

        Not a public API: run inference through ``client.kumorfm(...)`` or
        ``client.tabicl(...)``. Returns the adapter's typed result: a
        prediction ``pd.DataFrame``, or a ``kumorfm.rfm.rfm.Explanation`` when a
        KumoRFM request asks to explain.
        """
        self._transport._require_open()
        adapter = self._registry.get(request.model)
        if not isinstance(request, adapter.request_type):
            raise SdfmError(
                f"model {request.model!r} expects a "
                f"{request_type_names(adapter.request_type)}, got "
                f"{type(request).__name__}",
                code='INVALID_REQUEST',
            )
        return adapter.predict(self._transport, request)

    def close(self) -> None:
        r"""Releases the pooled connections and retires this client.

        Every later call raises; construct a new ``SDFMClient`` instead.
        Leaving a ``with`` block does this for you.
        """
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

    def _register(self, adapter: ModelAdapter) -> None:
        r"""Add a model adapter to this client's registry."""
        self._registry.register(adapter)

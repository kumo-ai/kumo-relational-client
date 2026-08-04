# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import TracebackType
from typing import Any

import pandas as pd
from nvidia_sdfm.base import (AdapterRegistry, ModelAdapter, ModelCapabilities,
                              PredictResult, request_type_names)
from nvidia_sdfm.core.serving import ServingTarget
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
    clients can target different endpoints (or tenants) at once: a prediction
    is always issued against the endpoint and credential of the client that
    started it, including when clients are used concurrently from several
    threads. Use it as a context manager, or call ``close()``.

    One caveat on the KumoRFM path: the driver underneath keeps a process-wide
    configuration, which each prediction reconfigures. Predictions are pinned
    to their own client and are unaffected, but the driver's own
    ``kumorfm.init()`` and anything else reading that global observe whichever
    client configured it last. Do not mix ``SDFMClient`` with direct driver
    initialization in one process.

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
        :meth:`health_ready` or by a prediction. To address a model served by
        name on a managed platform instead, use
        :meth:`for_databricks_serving`.

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
        self._configure(
            Transport(
                url,
                api_key,
                verify_ssl=verify_ssl,
                timeout=timeout,
                max_retries=max_retries,
            ),
            registry,
        )

    def _configure(
        self,
        transport: Transport | ServingTarget,
        registry: AdapterRegistry | None,
    ) -> None:
        r"""The one place an ``SDFMClient``'s fields are populated.

        Both construction paths route through here, so a field added to a
        client cannot be missing from clients built the other way.
        """
        self._transport = transport
        self._registry = registry if registry is not None else _default_registry()

    @classmethod
    def for_databricks_serving(
        cls,
        endpoint: str,
        *,
        workspace_client: Any | None = None,
        registry: AdapterRegistry | None = None,
    ) -> "SDFMClient":
        r"""A client for a model served by Databricks Model Serving.

        The counterpart to the constructor, which addresses a NIM by base URL.
        A serving endpoint is addressed by name through a workspace client, so
        there is no url, api_key, verify_ssl, timeout or retry policy to give --
        the platform owns those.

        >>> client = SDFMClient.for_databricks_serving("kumo-rfm")
        >>> df = client.kumorfm(graph).predict("PREDICT ... FOR ...", [1, 2])

        Args:
            endpoint: The serving endpoint name.
            workspace_client: An existing ``WorkspaceClient``. When omitted one
                is built from the ambient Databricks configuration, which is
                how a notebook authenticates without handling a token.
            registry: As for the constructor.

        Raises:
            SdfmError: with ``code='INVALID_CONFIGURATION'`` if ``endpoint`` is
                empty, is not a string, carries surrounding whitespace, or
                looks like a URL rather than a name.

        Nothing here contacts Databricks, so nothing here can fail on
        authentication. A missing ``databricks-sdk`` surfaces at the first
        ``predict`` as ``MissingExtraError``, and a workspace that refuses the
        ambient configuration as ``SdfmError``.
        """
        return cls._from_transport(
            ServingTarget(endpoint, workspace_client), registry,
        )

    @classmethod
    def _from_transport(
        cls,
        transport: Transport | ServingTarget,
        registry: AdapterRegistry | None = None,
    ) -> 'SDFMClient':
        r"""Build a client around an already-constructed target.

        ``__init__`` takes the arguments a NIM needs and builds a
        ``Transport``; a serving target is built from different arguments
        entirely. Both then land in ``_configure``.
        """
        client = cls.__new__(cls)
        client._configure(transport, registry)
        return client

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
        target = getattr(self._transport, 'endpoint', None)
        where = (f'endpoint={target!r}' if target is not None
                 else f'url={self._transport.url!r}')
        return f'SDFMClient({where}, models={self.models()})'

    def _register(self, adapter: ModelAdapter) -> None:
        r"""Add a model adapter to this client's registry."""
        self._registry.register(adapter)

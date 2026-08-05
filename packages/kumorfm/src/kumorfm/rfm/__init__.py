# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import threading
from dataclasses import dataclass

import kumorfm
from kumorfm.client.transport import RFMTransport

from .base import Table
from .backend.local import LocalTable
from .diagnostics import (
    GraphSanitizationReport,
    SanitizationStatus,
    TableSanitizationReport,
    TaskReferenceError,
)
from .graph import Graph, ViewConversionWarning
from .task_table import TaskTable
from .rfm import (
    ExplainConfig,
    Explanation,
    KumoRFM,
    MaterializedPredictionRequest,
)

logger = logging.getLogger('kumorfm_rfm')

_SDFM_CLIENT_TOKEN = object()

_DIRECT_USE_MESSAGE = (
    "Direct use of the KumoRFM engine is not supported. Run inference through "
    "the NVIDIA SDFM SDK:\n"
    "    from nvidia_sdfm import SDFMClient, kumorfm\n"
    "    graph = kumorfm.Graph.from_data(...)\n"
    "    with SDFMClient(url=...) as client:\n"
    "        client.kumorfm(graph).predict(query, indices=[...])"
)


@dataclass
class RfmGlobalState:
    _url: str = '__url_not_provided__'
    _thread_local = threading.local()

    # Thread-safe init-once.
    _initialized: bool = False
    _lock: threading.Lock = threading.Lock()

    @property
    def client(self) -> RFMTransport:
        if not self._initialized:
            raise RuntimeError(_DIRECT_USE_MESSAGE)
        return kumorfm.global_state.client

    def reset(self) -> None:  # For testing only.
        with self._lock:
            self._initialized = False
            self._url = '__url_not_provided__'
            self._thread_local = threading.local()


global_state = RfmGlobalState()


def _configure(
    url: str | None,
    api_key: str | None,
    verify_ssl: bool,
    log_level: str,
    timeout: float | None,
    max_retries: int = 3,
) -> None:
    resolved_url = (url or os.getenv("RFM_API_URL")
                    or os.getenv("KUMO_API_ENDPOINT"))

    kumorfm.init(url=resolved_url, api_key=api_key, verify_ssl=verify_ssl,
                 log_level=log_level, timeout=timeout,
                 max_retries=max_retries)

    global_state._url = kumorfm.global_state._url
    global_state._initialized = True


def init(
    url: str | None = None,
    api_key: str | None = None,
    verify_ssl: bool = True,
    log_level: str = "INFO",
    *,
    timeout: float | None = None,
    max_retries: int = 3,
    _token: object | None = None,
) -> None:
    if _token is not _SDFM_CLIENT_TOKEN:
        raise RuntimeError(_DIRECT_USE_MESSAGE)
    with global_state._lock:
        _configure(url, api_key, verify_ssl, log_level, timeout,
                   max_retries)


def close_client(_token: object | None = None) -> None:
    r"""Releases this thread's pooled connections to the engine's endpoint.

    The counterpart to :func:`init_client`, for a caller that is done with the
    endpoint it configured. The client is closed and evicted, but the
    configuration is left in place, so the next prediction on this thread
    rebuilds a client and carries on rather than failing. That is what makes
    this safe to call while another caller is still using the same endpoint:
    the cost of releasing early is a reconnect.

    Only this thread's client is released. Another thread's is its own, and
    closing it from here would break a prediction already in flight on it.

    Does nothing if this thread never built a client.
    """
    if _token is not _SDFM_CLIENT_TOKEN:
        raise RuntimeError(_DIRECT_USE_MESSAGE)
    with global_state._lock:
        thread_local = kumorfm.global_state.thread_local
        client = getattr(thread_local, '_client', None)
        if client is None:
            return
        del thread_local._client
        if hasattr(thread_local, '_client_config'):
            del thread_local._client_config
        client.close()


def init_client(
    url: str | None = None,
    api_key: str | None = None,
    verify_ssl: bool = True,
    log_level: str = "INFO",
    *,
    timeout: float | None = None,
    max_retries: int = 3,
    _token: object | None = None,
) -> RFMTransport:
    r"""Configures the engine and returns the client that configuration
    resolves to, as one atomic step.

    The engine's endpoint and credential live in a process-wide singleton, so
    configuring it and then reading back the resulting client as two steps is a
    race: a second caller configuring a different endpoint in between makes the
    first caller read the second's client. Because
    :attr:`~kumorfm.rfm.KumoRFM._api_client` resolves lazily -- at the first
    HTTP call, after graph materialization and subgraph sampling -- that window
    is wide enough to matter in practice.

    Resolving under the same lock that guards configuration closes it, provided
    the caller binds the returned client to its prediction rather than reading
    the global again later. Pass it to :class:`~kumorfm.rfm.KumoRFM` as
    ``_client``.
    """
    if _token is not _SDFM_CLIENT_TOKEN:
        raise RuntimeError(_DIRECT_USE_MESSAGE)
    with global_state._lock:
        _configure(url, api_key, verify_ssl, log_level, timeout,
                   max_retries)
        return kumorfm.global_state.client


def init_databricks_serving(
    endpoint: str,
    *,
    workspace_client: object | None = None,
    max_request_bytes: int | None = None,
    timeout: float | None = None,
    log_level: str = "INFO",
    _token: object | None = None,
) -> None:
    """Initialize against a Databricks Model Serving endpoint.

    The counterpart to :func:`init`, which resolves a NIM base URL from its
    argument or the environment; there is no URL to resolve here.

    Gated like :func:`init`: the engine is reachable only through
    ``SDFMClient``, so both entry points must refuse a direct call rather than
    leaving one of them as a way around the boundary.
    """
    if _token is not _SDFM_CLIENT_TOKEN:
        raise RuntimeError(_DIRECT_USE_MESSAGE)
    with global_state._lock:
        kumorfm.init_databricks_serving(
            endpoint,
            workspace_client=workspace_client,
            max_request_bytes=max_request_bytes,
            timeout=timeout,
            log_level=log_level,
        )
        global_state._url = f"databricks-serving:{endpoint}"
        global_state._initialized = True


LocalGraph = Graph  # NOTE Backward compatibility - do not use anymore.

__all__ = [
    'init',
    'init_client',
    'close_client',
    'init_databricks_serving',
    'Table',
    'LocalTable',
    'Graph',
    'GraphSanitizationReport',
    'ViewConversionWarning',
    'SanitizationStatus',
    'TableSanitizationReport',
    'TaskReferenceError',
    'TaskTable',
    'KumoRFM',
    'ExplainConfig',
    'Explanation',
    'MaterializedPredictionRequest',
]

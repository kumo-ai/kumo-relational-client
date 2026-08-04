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


def init(
    url: str | None = None,
    api_key: str | None = None,
    verify_ssl: bool = True,
    log_level: str = "INFO",
    *,
    timeout: float | None = None,
    _token: object | None = None,
) -> None:
    if _token is not _SDFM_CLIENT_TOKEN:
        raise RuntimeError(_DIRECT_USE_MESSAGE)
    with global_state._lock:
        resolved_url = (url or os.getenv("RFM_API_URL")
                        or os.getenv("KUMO_API_ENDPOINT"))

        kumorfm.init(url=resolved_url, api_key=api_key, verify_ssl=verify_ssl,
                     log_level=log_level, timeout=timeout)

        global_state._url = kumorfm.global_state._url
        global_state._initialized = True


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

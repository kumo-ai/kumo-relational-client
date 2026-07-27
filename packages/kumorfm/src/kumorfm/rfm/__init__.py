import logging
import os
import threading
from dataclasses import dataclass

import kumorfm
from kumorfm.client.client import KumoClient

from .base import Table
from .backend.local import LocalTable
from .diagnostics import (
    GraphSanitizationReport,
    SanitizationStatus,
    TableSanitizationReport,
    TaskReferenceError,
)
from .graph import Graph
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
    "    from nvidia_sdfm import SDFMClient, KumoRFMRequest, kumorfm\n"
    "    graph = kumorfm.Graph.from_data(...)\n"
    "    with SDFMClient(url=...) as client:\n"
    "        client.predict(KumoRFMRequest(graph=graph, query=...))"
)


@dataclass
class RfmGlobalState:
    _url: str = '__url_not_provided__'
    _thread_local = threading.local()

    # Thread-safe init-once.
    _initialized: bool = False
    _lock: threading.Lock = threading.Lock()

    @property
    def client(self) -> KumoClient:
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
    _token: object | None = None,
) -> None:
    if _token is not _SDFM_CLIENT_TOKEN:
        raise RuntimeError(_DIRECT_USE_MESSAGE)
    with global_state._lock:
        resolved_url = (url or os.getenv("RFM_API_URL")
                        or os.getenv("KUMO_API_ENDPOINT"))

        kumorfm.init(url=resolved_url, api_key=api_key, verify_ssl=verify_ssl,
                     log_level=log_level)

        global_state._url = kumorfm.global_state._url
        global_state._initialized = True


LocalGraph = Graph  # NOTE Backward compatibility - do not use anymore.

__all__ = [
    'init',
    'Table',
    'LocalTable',
    'Graph',
    'GraphSanitizationReport',
    'SanitizationStatus',
    'TableSanitizationReport',
    'TaskReferenceError',
    'TaskTable',
    'KumoRFM',
    'ExplainConfig',
    'Explanation',
    'MaterializedPredictionRequest',
]

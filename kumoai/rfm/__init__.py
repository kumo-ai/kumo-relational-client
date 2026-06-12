import ipaddress
import logging
import os
import re
import socket
import threading
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse

import kumoai
from kumoai.client.client import KumoClient
from kumoai.spcs import _get_active_session

from .authenticate import authenticate
from .sagemaker import (
    KumoClient_SageMakerAdapter,
    KumoClient_SageMakerProxy_Local,
)
from .base import Table
from .backend.local import LocalTable
from .graph import Graph
from .task_table import TaskTable
from .rfm import ExplainConfig, Explanation, KumoRFM

logger = logging.getLogger('kumoai_rfm')


def _is_local_address(host: str | None) -> bool:
    """Return True if the hostname/IP refers to the local machine."""
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
        for _, _, _, _, sockaddr in infos:
            ip = sockaddr[0]
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_loopback or ip_obj.is_unspecified:
                return True
        return False
    except Exception:
        return False


class InferenceBackend(str, Enum):
    REST = "REST"
    LOCAL_SAGEMAKER = "LOCAL_SAGEMAKER"
    AWS_SAGEMAKER = "AWS_SAGEMAKER"
    UNKNOWN = "UNKNOWN"


def _detect_backend(
        url: str,  #
) -> tuple[InferenceBackend, str | None, str | None]:
    parsed = urlparse(url)

    # Remote SageMaker
    if ("runtime.sagemaker" in parsed.netloc
            and parsed.path.endswith("/invocations")):
        # Example: https://runtime.sagemaker.us-west-2.amazonaws.com/
        # endpoints/Name/invocations
        match = re.search(r"runtime\.sagemaker\.([a-z0-9-]+)\.amazonaws\.com",
                          parsed.netloc)
        region = match.group(1) if match else None
        m = re.search(r"/endpoints/([^/]+)/invocations", parsed.path)
        endpoint_name = m.group(1) if m else None
        return InferenceBackend.AWS_SAGEMAKER, region, endpoint_name

    # Local SageMaker
    if parsed.port == 8080 and parsed.path.endswith(
            "/invocations") and _is_local_address(parsed.hostname):
        return InferenceBackend.LOCAL_SAGEMAKER, None, None

    # Default: regular REST
    return InferenceBackend.REST, None, None


def _get_snowflake_url(snowflake_application: str) -> str:
    snowpark_session = _get_active_session()
    if not snowpark_session:
        raise ValueError(
            "KumoRFM initialization failed. 'snowflake_application' is "
            "specified without an active Snowpark session. If running outside "
            "a Snowflake notebook, specify a URL and credentials.")
    with snowpark_session.connection.cursor() as cur:
        cur.execute(
            f"DESCRIBE SERVICE {snowflake_application}.user_schema.rfm_service"
            f" ->> SELECT \"dns_name\" from $1")
        result = cur.fetchone()
        assert result is not None
        dns_name: str = result[0]
    return f"http://{dns_name}:8000/api"


@dataclass
class RfmGlobalState:
    _url: str = '__url_not_provided__'
    _backend: InferenceBackend = InferenceBackend.UNKNOWN
    _region: str | None = None
    _endpoint_name: str | None = None
    _thread_local = threading.local()

    # Thread-safe init-once.
    _initialized: bool = False
    _lock: threading.Lock = threading.Lock()

    @property
    def client(self) -> KumoClient:
        if self._backend == InferenceBackend.UNKNOWN:
            raise RuntimeError("KumoRFM is not yet initialized")

        if self._backend == InferenceBackend.REST:
            return kumoai.global_state.client

        if hasattr(self._thread_local, '_sagemaker'):
            # Set the spcs token in the client to ensure it has the latest.
            return self._thread_local._sagemaker

        sagemaker_client: KumoClient
        if self._backend == InferenceBackend.LOCAL_SAGEMAKER:
            sagemaker_client = KumoClient_SageMakerProxy_Local(self._url)
        else:
            assert self._backend == InferenceBackend.AWS_SAGEMAKER
            assert self._region
            assert self._endpoint_name
            sagemaker_client = KumoClient_SageMakerAdapter(
                self._region, self._endpoint_name)

        self._thread_local._sagemaker = sagemaker_client
        return sagemaker_client

    def reset(self) -> None:  # For testing only.
        with self._lock:
            self._initialized = False
            self._url = '__url_not_provided__'
            self._backend = InferenceBackend.UNKNOWN
            self._region = None
            self._endpoint_name = None
            self._thread_local = threading.local()


global_state = RfmGlobalState()


def init(
    url: str | None = None,
    api_key: str | None = None,
    snowflake_credentials: dict[str, str] | None = None,
    snowflake_application: str | None = None,
    log_level: str = "INFO",
) -> None:
    with global_state._lock:
        if global_state._initialized:
            if url != global_state._url:
                raise RuntimeError(
                    "KumoRFM has already been initialized with a different "
                    "API URL. Re-initialization with a different URL is not "
                    "supported.")
            return

        if snowflake_application:
            if url is not None:
                raise ValueError(
                    "KumoRFM initialization failed. Both "
                    "'snowflake_application' and 'url' are specified. If "
                    "running from a Snowflake notebook, specify only "
                    "'snowflake_application'.")
            url = _get_snowflake_url(snowflake_application)
            api_key = "test:DISABLED"

        if url is None:
            url = os.getenv("RFM_API_URL", "https://kumorfm.ai/api")

        backend, region, endpoint_name = _detect_backend(url)
        if backend == InferenceBackend.REST:
            kumoai.init(
                url=url,
                api_key=api_key,
                snowflake_credentials=snowflake_credentials,
                log_level=log_level,
            )
        elif backend == InferenceBackend.AWS_SAGEMAKER:
            assert region
            assert endpoint_name
            KumoClient_SageMakerAdapter(region, endpoint_name).authenticate()
            logger.info("KumoRFM initialized in AWS SageMaker")
        else:
            assert backend == InferenceBackend.LOCAL_SAGEMAKER
            KumoClient_SageMakerProxy_Local(url).authenticate()
            logger.info(f"KumoRFM initialized in local SageMaker at '{url}'")

        global_state._url = url
        global_state._backend = backend
        global_state._region = region
        global_state._endpoint_name = endpoint_name
        global_state._initialized = True


LocalGraph = Graph  # NOTE Backward compatibility - do not use anymore.

__all__ = [
    'authenticate',
    'init',
    'Table',
    'LocalTable',
    'Graph',
    'TaskTable',
    'KumoRFM',
    'ExplainConfig',
    'Explanation',
]

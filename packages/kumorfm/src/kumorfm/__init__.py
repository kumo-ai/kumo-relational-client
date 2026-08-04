# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Optional

from kumorfm._logging import _ENV_KUMO_LOG, initialize_logging
from kumorfm._singleton import Singleton
from kumorfm._version import __version__
from kumorfm.api.typing import Dtype, Stype
from kumorfm.client.client import KumoClient
from kumorfm.client.transport import RFMTransport
from kumorfm.futures import initialize_event_loop

initialize_logging()
initialize_event_loop()


@dataclass
class GlobalState(metaclass=Singleton):
    r"""Global state needed by the RFM client.

    A deployment is addressed either by base URL (``_url``, the raw-NIM mode)
    or by ``_client_factory``, which builds a transport addressed some other
    way -- a named Databricks Model Serving endpoint, say. The two never
    coexist: every mode switch goes through :meth:`clear`, which resets both
    before the new mode installs its own. The ``_serving_*`` fields record what
    :func:`init_databricks_serving` was last called with, so repeating the call
    returns early rather than re-resolving credentials.
    """

    _url: Optional[str] = None
    _api_key: Optional[str] = None
    _verify_ssl: bool = True
    _timeout: Optional[float] = None
    _client_factory: Optional[Callable[[], Any]] = None
    _serving_endpoint: Optional[str] = None
    _serving_workspace: Optional[Any] = None
    _serving_overrides: Optional[dict] = None

    thread_local: threading.local = field(default_factory=threading.local)

    def clear(self) -> None:
        self._client_factory = None
        self._serving_endpoint = None
        self._serving_workspace = None
        self._serving_overrides = None
        if hasattr(self.thread_local, '_client'):
            try:
                self.thread_local._client.close()
            except Exception:
                pass
            del self.thread_local._client
        self._url = None
        self._api_key = None
        self._verify_ssl = True
        self._timeout = None

    @property
    def initialized(self) -> bool:
        return self._url is not None or self._client_factory is not None

    @property
    def client(self) -> RFMTransport:
        """The request client for this thread.

        A :class:`KumoClient` for the raw-NIM deployment, or whatever
        ``_client_factory`` builds otherwise. Cached per thread, so an
        existing :class:`KumoRFM` instance keeps the client it was built with.

        Auto-init from an ambient ``KUMO_API_ENDPOINT`` is gated on
        ``initialized`` rather than on ``_url``, because a serving deployment
        has no URL and would otherwise have the factory it just installed
        replaced. It is skipped under pytest so a test run cannot silently
        connect to a real endpoint.
        """
        if (not self.initialized and os.getenv("KUMO_API_ENDPOINT")
                and "pytest" not in sys.modules):
            init()

        if not self.initialized:
            raise ValueError("Client creation or authentication failed. "
                             "Please re-create your client before proceeding.")

        if hasattr(self.thread_local, '_client'):
            return self.thread_local._client

        if self._client_factory is not None:
            client = self._client_factory()
        else:
            client = KumoClient(self._url, self._api_key,
                                verify_ssl=self._verify_ssl,
                                timeout=self._timeout)
        self.thread_local._client = client
        return client


global_state: GlobalState = GlobalState()


def init(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    verify_ssl: bool = True,
    log_level: str = "INFO",
    timeout: Optional[float] = None,
) -> None:
    r"""Initializes the KumoRFM client against a Universal TFM NIM.

    NIMs are unauthenticated by contract, so ``api_key`` is optional and only
    needed when a deployment fronts the NIM with an authenticating gateway.
    Re-initializing with different settings reconfigures the client and closes
    the previous session; re-initializing with identical settings is a no-op.

    ``timeout`` bounds each individual request attempt, in seconds; ``None``
    leaves requests unbounded.
    """
    set_log_level(os.getenv(_ENV_KUMO_LOG, log_level))

    api_key = api_key or os.getenv("KUMO_API_KEY")
    url = url or os.getenv("KUMO_API_ENDPOINT")
    if not url:
        raise ValueError("KumoRFM initialization failed since no endpoint "
                         "URL was provided. Please either set the "
                         "'KUMO_API_ENDPOINT' environment variable or "
                         "explicitly call `kumorfm.init(url=...)`.")

    if global_state.initialized:
        unchanged = (url == global_state._url
                     and api_key == global_state._api_key
                     and verify_ssl == global_state._verify_ssl
                     and timeout == global_state._timeout)
        if unchanged:
            return
        global_state.clear()

    client = KumoClient(url=url, api_key=api_key, verify_ssl=verify_ssl,
                        timeout=timeout)
    client.authenticate()

    global_state._url = client._url
    global_state._api_key = client._api_key
    global_state._verify_ssl = verify_ssl
    global_state._timeout = timeout

    logging.getLogger('kumorfm').info(
        f"Initialized KumoRFM SDK v{__version__} against deployment '{url}'")


def init_databricks_serving(
    endpoint: str,
    *,
    workspace_client: Optional[Any] = None,
    max_request_bytes: Optional[int] = None,
    timeout: Optional[float] = None,
    log_level: str = "INFO",
) -> None:
    r"""Initialize against a Databricks Model Serving endpoint.

    The counterpart to :func:`init`, which addresses a Universal TFM NIM by
    base URL. A serving endpoint is addressed by **name** through a workspace
    client, so there is no URL, no ``api_key``, and no readiness probe -- the
    endpoint does not advertise one.

    Re-initializing with identical arguments is a no-op, as for :func:`init`.
    That matters more here: when ``workspace_client`` is omitted, building the
    client resolves ambient Databricks credentials, and callers score in
    partitions by calling this once per batch. The unchanged check therefore
    runs before anything is constructed, and the endpoint is validated before
    that, so a rejected name leaves the process exactly as it was, log level
    included. Overrides are forwarded only when given, leaving the transport
    owning its own defaults and their validation.

    Args:
        endpoint: The serving endpoint name.
        workspace_client: An existing ``WorkspaceClient``. When omitted one is
            built from the ambient Databricks configuration, which is how a
            notebook authenticates without handling a token.
        max_request_bytes: Overrides the client-side request size cap. Omit to
            use the transport's own default, which tracks the platform limit.
        timeout: Overrides the request timeout, in seconds. Applies only when
            no ``workspace_client`` is supplied; an injected one keeps its own
            configuration.
        log_level: As for :func:`init`.

    Raises:
        ValueError: if ``endpoint`` is empty, whitespace-padded, or URL-shaped
            (a serving endpoint is named, not addressed), or if an override is
            out of range.
        ImportError: if the ``databricks-serving`` extra is not installed and
            no ``workspace_client`` was supplied.
        HTTPException: if a workspace client has to be built from the ambient
            configuration and that authentication fails.
    """
    overrides: dict[str, Any] = {
        name: value
        for name, value in (('max_request_bytes', max_request_bytes),
                            ('timeout', timeout)) if value is not None
    }

    if global_state.initialized:
        unchanged = (global_state._client_factory is not None
                     and endpoint == global_state._serving_endpoint
                     and workspace_client is global_state._serving_workspace
                     and overrides == global_state._serving_overrides)
        if unchanged:
            set_log_level(os.getenv(_ENV_KUMO_LOG, log_level))
            return

    from kumorfm.client.databricks_serving import DatabricksServingClient

    probe = DatabricksServingClient(endpoint, workspace_client, **overrides)

    set_log_level(os.getenv(_ENV_KUMO_LOG, log_level))

    if global_state.initialized:
        global_state.clear()

    global_state._client_factory = lambda: DatabricksServingClient(
        endpoint, workspace_client, **overrides
    )
    global_state._serving_endpoint = endpoint
    global_state._serving_workspace = workspace_client
    global_state._serving_overrides = overrides
    global_state.thread_local._client = probe

    logging.getLogger('kumorfm').info(
        f"Initialized KumoRFM SDK v{__version__} against Databricks Model "
        f"Serving endpoint '{endpoint}'")


def set_log_level(level: str) -> None:
    r"""Sets the Kumo logging level."""
    logging.getLogger('kumorfm').setLevel(level)


@lru_cache
def in_streamlit_notebook() -> bool:
    try:
        import streamlit  # noqa: F401
        from snowflake.snowpark.context import get_active_session
        get_active_session()
        return True
    except Exception:
        return False


@lru_cache
def in_jupyter_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if 'google.colab' in str(shell.__class__):
            return True
        if 'DatabricksShell' in str(shell.__class__):
            return True
        return shell.__class__.__name__ == 'ZMQInteractiveShell'
    except Exception:
        return False


@lru_cache
def in_vnext_notebook() -> bool:
    try:
        from snowflake.snowpark.context import get_active_session
        get_active_session()
        return in_jupyter_notebook()
    except Exception:
        return False


@lru_cache
def in_notebook() -> bool:
    return in_streamlit_notebook() or in_jupyter_notebook()


@lru_cache
def in_tmux() -> bool:
    return 'TMUX' in os.environ or 'HERDR_ENV' in os.environ


__all__ = [
    'Dtype',
    'Stype',
    'KumoClient',
    'global_state',
    'init',
    'init_databricks_serving',
    'set_log_level',
    '__version__',
]

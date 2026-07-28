from functools import lru_cache
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Optional

from kumorfm.api.typing import Dtype, Stype

from kumorfm._logging import _ENV_KUMO_LOG, initialize_logging
from kumorfm._singleton import Singleton
from kumorfm._version import __version__
from kumorfm.client.client import KumoClient
from kumorfm.futures import initialize_event_loop

initialize_logging()
initialize_event_loop()


@dataclass
class GlobalState(metaclass=Singleton):
    r"""Global state needed by the RFM client."""

    _url: Optional[str] = None
    _api_key: Optional[str] = None
    _verify_ssl: bool = True

    thread_local: threading.local = field(default_factory=threading.local)

    def clear(self) -> None:
        if hasattr(self.thread_local, '_client'):
            try:
                self.thread_local._client.close()
            except Exception:
                pass
            del self.thread_local._client
        self._url = None
        self._api_key = None
        self._verify_ssl = True

    @property
    def initialized(self) -> bool:
        return self._url is not None

    @property
    def client(self) -> KumoClient:
        if self._url is None:
            raise ValueError("Client creation or authentication failed. "
                             "Please re-create your client before proceeding.")

        if hasattr(self.thread_local, '_client'):
            return self.thread_local._client

        client = KumoClient(self._url, self._api_key,
                            verify_ssl=self._verify_ssl)
        self.thread_local._client = client
        return client


global_state: GlobalState = GlobalState()


def init(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    verify_ssl: bool = True,
    log_level: str = "INFO",
) -> None:
    r"""Initializes the KumoRFM client against a Universal TFM NIM.

    NIMs are unauthenticated by contract, so ``api_key`` is optional and only
    needed when a deployment fronts the NIM with an authenticating gateway.
    Re-initializing with different settings reconfigures the client and closes
    the previous session; re-initializing with identical settings is a no-op.
    """
    set_log_level(os.getenv(_ENV_KUMO_LOG, log_level))

    api_key = api_key or os.getenv("KUMO_API_KEY")
    url = url or os.getenv("KUMO_API_ENDPOINT")
    if url is None:
        raise ValueError("KumoRFM initialization failed since no endpoint "
                         "URL was provided. Please either set the "
                         "'KUMO_API_ENDPOINT' environment variable or "
                         "explicitly call `kumorfm.init(url=...)`.")

    if global_state.initialized:
        unchanged = (url == global_state._url
                     and api_key == global_state._api_key
                     and verify_ssl == global_state._verify_ssl)
        if unchanged:
            return
        global_state.clear()

    client = KumoClient(url=url, api_key=api_key, verify_ssl=verify_ssl)
    client.authenticate()

    global_state._url = client._url
    global_state._api_key = client._api_key
    global_state._verify_ssl = verify_ssl

    logging.getLogger('kumorfm').info(
        f"Initialized KumoRFM SDK v{__version__} against deployment '{url}'")


def set_log_level(level: str) -> None:
    r"""Sets the Kumo logging level."""
    logging.getLogger('kumorfm').setLevel(level)


if "pytest" not in sys.modules and "KUMO_API_ENDPOINT" in os.environ:
    init()


@lru_cache
def in_streamlit_notebook() -> bool:
    try:
        from snowflake.snowpark.context import get_active_session
        import streamlit  # noqa: F401
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
    'set_log_level',
    '__version__',
]

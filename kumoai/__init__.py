from functools import lru_cache
import logging
import os
import sys
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from kumoapi.typing import Dtype, Stype

from kumoai._logging import _ENV_KUMO_LOG, initialize_logging
from kumoai._singleton import Singleton
from kumoai._version import __version__
from kumoai.client.client import KumoClient
from kumoai.futures import create_future, initialize_event_loop
from kumoai.spcs import (
    _get_active_session,
    _get_spcs_token,
    _run_refresh_spcs_token,
)

initialize_logging()
initialize_event_loop()


@dataclass
class GlobalState(metaclass=Singleton):
    r"""Global state needed by the RFM client."""

    _url: Optional[str] = None
    _api_key: Optional[str] = None
    _snowflake_credentials: Optional[Dict[str, Any]] = None
    _spcs_token: Optional[str] = None
    _snowpark_session: Optional[Any] = None

    # RFM does not expose RBAC selection helpers, but KumoClient injects these
    # fields when present to stay compatible with shared request code.
    _rbac_enabled: bool = False
    _group_id: Optional[str] = None
    _project_id: Optional[str] = None

    thread_local: threading.local = field(default_factory=threading.local)

    def clear(self) -> None:
        if hasattr(self.thread_local, '_client'):
            del self.thread_local._client
        self._url = None
        self._api_key = None
        self._snowflake_credentials = None
        self._spcs_token = None
        self._snowpark_session = None
        self._rbac_enabled = False
        self._group_id = None
        self._project_id = None

    def set_spcs_token(self, spcs_token: str) -> None:
        self._spcs_token = spcs_token

    @property
    def initialized(self) -> bool:
        return self._url is not None and (
            self._api_key is not None or self._snowflake_credentials
            is not None or self._snowpark_session is not None)

    @property
    def client(self) -> KumoClient:
        if self._url is None or (self._api_key is None
                                 and self._spcs_token is None
                                 and self._snowpark_session is None):
            raise ValueError("Client creation or authentication failed. "
                             "Please re-create your client before proceeding.")

        if hasattr(self.thread_local, '_client'):
            if self._spcs_token is not None:
                self.thread_local._client.set_spcs_token(self._spcs_token)
            return self.thread_local._client

        client = KumoClient(self._url, self._api_key, self._spcs_token)
        self.thread_local._client = client
        return client


global_state: GlobalState = GlobalState()


def init(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    snowflake_credentials: Optional[Dict[str, str]] = None,
    snowflake_application: Optional[str] = None,
    log_level: str = "INFO",
) -> None:
    r"""Initializes and authenticates the KumoRFM API client."""
    if global_state.initialized:
        warnings.warn("Kumo SDK already initialized. To re-initialize the "
                      "SDK, please start a new interpreter. No changes will "
                      "be made to the current session.")
        return

    set_log_level(os.getenv(_ENV_KUMO_LOG, log_level))

    api_key = api_key or os.getenv("KUMO_API_KEY")

    snowpark_session = None
    if snowflake_application:
        if url is not None:
            raise ValueError(
                "Kumo SDK initialization failed. Both 'snowflake_application' "
                "and 'url' are specified. If running from a Snowflake "
                "notebook, specify only 'snowflake_application'.")
        snowpark_session = _get_active_session()
        if not snowpark_session:
            raise ValueError(
                "Kumo SDK initialization failed. 'snowflake_application' is "
                "specified without an active Snowpark session. If running "
                "outside a Snowflake notebook, specify a URL and credentials.")
        description = snowpark_session.sql(
            f"DESCRIBE SERVICE {snowflake_application}."
            "USER_SCHEMA.KUMO_SERVICE").collect()[0]
        url = f"http://{description.dns_name}:8888/public_api"

    if api_key is None and not snowflake_application:
        if snowflake_credentials is None:
            raise ValueError(
                "Kumo SDK initialization failed. Neither an API key nor "
                "Snowflake credentials provided. Please either set the "
                "'KUMO_API_KEY' or explicitly call `kumoai.init(...)`.")
        required_password = {'user', 'password', 'account'}
        required_keypair = {'user', 'private_key', 'account'}
        keys = set(snowflake_credentials.keys())
        keypair_ok = required_keypair.issubset(keys) and 'password' not in keys
        password_ok = required_password.issubset(keys)
        if not (password_ok or keypair_ok):
            raise ValueError(
                f"Snowflake credentials must be either password-based "
                f"{{'user', 'password', 'account'}} or key-pair "
                f"{{'user', 'private_key', 'account'}} with optional "
                f"'private_key_passphrase'. Got keys: {keys}")

    url = url or os.getenv("KUMO_API_ENDPOINT")
    try:
        if api_key:
            url = url or f"http://{api_key.split(':')[0]}.kumoai.cloud/api"
    except KeyError:
        pass
    if url is None:
        raise ValueError("Kumo SDK initialization failed since no endpoint "
                         "URL was provided. Please either set the "
                         "'KUMO_API_ENDPOINT' environment variable or "
                         "explicitly call `kumoai.init(...)`.")

    spcs_token = _get_spcs_token(
        snowflake_credentials
    ) if not api_key and snowflake_credentials else None
    client = KumoClient(url=url, api_key=api_key, spcs_token=spcs_token)
    client.authenticate()

    global_state._url = client._url
    global_state._api_key = client._api_key
    global_state._snowflake_credentials = snowflake_credentials
    global_state._spcs_token = client._spcs_token
    global_state._snowpark_session = snowpark_session

    if not api_key and snowflake_credentials:
        create_future(_run_refresh_spcs_token(minutes=10))

    logging.getLogger('kumoai').info(
        f"Initialized Kumo RFM SDK v{__version__} against deployment '{url}'")


def set_log_level(level: str) -> None:
    r"""Sets the Kumo logging level."""
    logging.getLogger('kumoai').setLevel(level)


if ("pytest" not in sys.modules and "KUMO_API_KEY" in os.environ
        and "KUMO_API_ENDPOINT" in os.environ):
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

import logging
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from kumoai.client.endpoints import Endpoint, HTTPMethod

logger = logging.getLogger('kumoai')

_AUTH_STATUS_CODES = frozenset({401, 403})


def _raise_init_error(url: str, exc: BaseException) -> None:
    """Translate an init-time requests exception into a user-facing error."""
    if isinstance(exc, requests.exceptions.HTTPError):
        response = exc.response
        status_code = (response.status_code if response is not None else None)
        body = response.text if response is not None else ''
        if status_code in _AUTH_STATUS_CODES:
            raise ValueError(
                f"Client authentication failed for {url!r}. Please check "
                "that your API key / credentials are valid. "
                f"HTTP {status_code}: {body}") from exc
        raise ValueError(
            f"Kumo server at {url!r} returned HTTP {status_code} while "
            f"initializing the SDK. Response: {body}") from exc
    if isinstance(exc, requests.exceptions.Timeout):
        raise ValueError(f"Timed out connecting to Kumo server at {url!r}. "
                         f"Original error: {exc}") from exc
    if isinstance(exc, requests.exceptions.ConnectionError):
        raise ValueError(
            f"Could not connect to Kumo server at {url!r}. Verify the "
            f"server is running and the URL is correct. Original error: "
            f"{exc}") from exc
    raise ValueError(f"Failed to initialize Kumo SDK client against {url!r}. "
                     f"Exception: {exc}") from exc


class KumoClient:
    def __init__(
        self,
        url: str,
        api_key: Optional[str],
        spcs_token: Optional[str] = None,
        verify_ssl: bool = True,
    ) -> None:
        r"""Creates an authenticated client for KumoRFM API requests."""
        self._url = url
        self._api_key = api_key
        self._spcs_token = spcs_token
        self._verify_ssl = verify_ssl

        retry_strategy = Retry(
            total=10,
            connect=3,
            read=3,
            status=5,
            status_forcelist=[408, 429, 500, 502, 503, 504],
            backoff_factor=2.0,
        )
        http_adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount('http://', http_adapter)
        session.mount('https://', http_adapter)
        self._session = session
        if self._api_key:
            self._session.headers.update({"X-API-Key": self._api_key})
        elif self._spcs_token:
            self._session.headers.update(
                {'Authorization': f'Snowflake Token={self._spcs_token}'})

    def fetch_config(self) -> dict:
        r"""Fetches deployment feature flags when the endpoint exposes them."""
        try:
            response = self._session.get(self._url + '/config',
                                         verify=self._verify_ssl)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning("Failed to fetch feature flags: %s", e)
            return {}

    def authenticate(self) -> None:
        """Raises an exception if authentication fails."""
        try:
            response = self._session.get(self._url + '/v1/connectors',
                                         verify=self._verify_ssl)
            response.raise_for_status()
        except requests.RequestException as e:
            _raise_init_error(self._url, e)

    def set_spcs_token(self, spcs_token: str) -> None:
        r"""Sets the SPCS token for subsequent requests."""
        self._spcs_token = spcs_token
        if self._spcs_token:
            self._session.headers.update(
                {'Authorization': f'Snowflake Token={self._spcs_token}'})

    def _request_with_401_retry(
        self,
        method_func: Any,
        *args: Any,
        **kwargs: Any,
    ) -> requests.Response:
        response = method_func(*args, **kwargs)
        if response.status_code == 401 and self._spcs_token:
            from kumoai import global_state
            from kumoai.spcs import refresh_spcs_token
            logger.warning(f"Received HTTP 401 when calling {method_func}. "
                           "Refetching SPCS token and retrying")
            try:
                refresh_spcs_token()
                if global_state._spcs_token:
                    self.set_spcs_token(global_state._spcs_token)
            except Exception as e:
                logger.error(f"Failed to refresh SPCS token: {e}.")
                return response
            response = method_func(*args, **kwargs)

        return response

    def _inject_rbac_params(self, kwargs: dict) -> dict:
        from kumoai import global_state
        if not global_state._rbac_enabled:
            return kwargs

        params = dict(kwargs.get('params', None) or {})
        if global_state._group_id is not None:
            params['group_id'] = global_state._group_id
        if global_state._project_id is not None:
            params['project_id'] = global_state._project_id
        kwargs['params'] = params
        return kwargs

    def _request(self, endpoint: Endpoint, **kwargs: Any) -> requests.Response:
        r"""Send an HTTP request to the specified endpoint."""
        endpoint_str = endpoint.get_path()
        if endpoint.method == HTTPMethod.GET:
            return self._get(endpoint_str, **kwargs)
        if endpoint.method == HTTPMethod.POST:
            return self._post(endpoint_str, **kwargs)
        if endpoint.method == HTTPMethod.PATCH:
            return self._patch(endpoint_str, **kwargs)
        if endpoint.method == HTTPMethod.DELETE:
            return self._delete(endpoint_str, **kwargs)
        raise ValueError(f"Unsupported HTTP method: {endpoint.method}")

    def _get(self, endpoint: str, **kwargs: Any) -> requests.Response:
        kwargs = self._inject_rbac_params(kwargs)
        url = self._format_endpoint_url(endpoint)
        return self._request_with_401_retry(self._session.get, url=url,
                                            verify=self._verify_ssl, **kwargs)

    def _post(self, endpoint: str, **kwargs: Any) -> requests.Response:
        kwargs = self._inject_rbac_params(kwargs)
        url = self._format_endpoint_url(endpoint)
        return self._request_with_401_retry(self._session.post, url=url,
                                            verify=self._verify_ssl, **kwargs)

    def _patch(self, endpoint: str, **kwargs: Any) -> requests.Response:
        kwargs = self._inject_rbac_params(kwargs)
        url = self._format_endpoint_url(endpoint)
        return self._request_with_401_retry(self._session.patch, url=url,
                                            verify=self._verify_ssl, **kwargs)

    def _delete(self, endpoint: str, **kwargs: Any) -> requests.Response:
        kwargs = self._inject_rbac_params(kwargs)
        url = self._format_endpoint_url(endpoint)
        return self._request_with_401_retry(self._session.delete, url=url,
                                            verify=self._verify_ssl, **kwargs)

    def _format_endpoint_url(self, endpoint: str) -> str:
        if not endpoint.startswith("/"):
            raise ValueError("Endpoint path must start with '/'")
        return f"{self._url}{endpoint}"

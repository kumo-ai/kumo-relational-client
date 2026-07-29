# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, NoReturn, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from kumorfm.client.endpoints import Endpoint, HTTPMethod

logger = logging.getLogger('kumorfm')

_AUTH_STATUS_CODES = frozenset({401, 403})


def _json_or_none(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _raise_init_error(url: str, exc: BaseException) -> NoReturn:
    """Translate an init-time requests exception into a user-facing error."""
    if isinstance(exc, requests.exceptions.HTTPError):
        response = exc.response
        status_code = (response.status_code if response is not None else None)
        body = response.text if response is not None else ''
        if status_code in _AUTH_STATUS_CODES:
            raise ValueError(
                f"Client authentication failed for {url!r}. If the NIM is "
                "behind an authenticating gateway, pass a matching "
                f"'api_key'. HTTP {status_code}: {body}") from exc
        raise ValueError(
            f"NIM at {url!r} returned HTTP {status_code} while "
            f"initializing the SDK. Response: {body}") from exc
    if isinstance(exc, requests.exceptions.Timeout):
        raise ValueError(f"Timed out connecting to the NIM at {url!r}. "
                         f"Original error: {exc}") from exc
    if isinstance(exc, requests.exceptions.ConnectionError):
        raise ValueError(
            f"Could not connect to the NIM at {url!r}. Verify the "
            f"server is running and the URL is correct. Original error: "
            f"{exc}") from exc
    raise ValueError(f"Failed to initialize KumoRFM client against {url!r}. "
                     f"Exception: {exc}") from exc


class KumoClient:
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        verify_ssl: bool = True,
    ) -> None:
        r"""Creates a client for KumoRFM requests against a Universal TFM NIM.

        NIMs are unauthenticated by contract. ``api_key`` is optional and only
        needed when a deployment adds its own authenticating gateway; when
        given it is sent as the ``X-API-Key`` header, otherwise no auth header
        is set.
        """
        self._url = url
        self._api_key = api_key
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

    def close(self) -> None:
        r"""Closes the underlying HTTP session and its pooled connections."""
        self._session.close()

    def authenticate(self) -> None:
        """Verify the endpoint is a reachable Universal TFM NIM.

        NIMs are unauthenticated by contract. When a deployment fronts the NIM
        with an authenticating gateway, a 401/403 from either probe surfaces as
        an authentication error rather than being masked as a missing model.
        """
        ready = self._probe('/v1/health/ready')
        ready_data = _json_or_none(ready)
        ready_status = (str(ready_data.get('status', '')).lower()
                        if isinstance(ready_data, dict) else '')
        ready_check = (str(ready_data.get('check', '')).lower()
                       if isinstance(ready_data, dict) else '')
        if not (ready_status == 'ready'
                or (ready_status == 'healthy' and ready_check == 'ready')):
            raise ValueError(
                f"NIM at {self._url!r} is not ready. '/v1/health/ready' "
                f"returned: {ready_data!r}")

        models = self._probe('/v1/models')
        data = _json_or_none(models)
        advertised = isinstance(data, dict) and any(
            isinstance(model, dict) and model.get('id') == 'kumo-rfm'
            for model in data.get('data', []))
        if not advertised:
            raise ValueError(
                f"Endpoint {self._url!r} did not advertise the 'kumo-rfm' "
                "model at '/v1/models'. Point the SDK at a Universal TFM NIM "
                "serving Kumo RFM.")

    def _probe(self, path: str) -> requests.Response:
        try:
            response = self._session.get(self._url + path,
                                         verify=self._verify_ssl, timeout=10)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            _raise_init_error(self._url, e)

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
        url = self._format_endpoint_url(endpoint)
        return self._session.get(url, verify=self._verify_ssl, **kwargs)

    def _post(self, endpoint: str, **kwargs: Any) -> requests.Response:
        url = self._format_endpoint_url(endpoint)
        return self._session.post(url, verify=self._verify_ssl, **kwargs)

    def _patch(self, endpoint: str, **kwargs: Any) -> requests.Response:
        url = self._format_endpoint_url(endpoint)
        return self._session.patch(url, verify=self._verify_ssl, **kwargs)

    def _delete(self, endpoint: str, **kwargs: Any) -> requests.Response:
        url = self._format_endpoint_url(endpoint)
        return self._session.delete(url, verify=self._verify_ssl, **kwargs)

    def _format_endpoint_url(self, endpoint: str) -> str:
        if not endpoint.startswith("/"):
            raise ValueError("Endpoint path must start with '/'")
        return f"{self._url}{endpoint}"

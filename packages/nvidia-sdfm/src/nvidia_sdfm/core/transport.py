# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from nvidia_sdfm.errors import NimRequestError, SdfmError

_PREDICTIONS_PATH = '/v1/predictions'
_HEALTH_READY_PATH = '/v1/health/ready'
_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_FACTOR = 0.5
_RETRY_STATUS = (429, 500, 502, 503, 504)
_LOCAL_HOSTS = frozenset({'localhost', '127.0.0.1', '::1'})


def _validate_url(url: str, api_key: str | None) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        raise SdfmError(
            f'url must start with http:// or https://, got {url!r}',
            code='INVALID_CONFIGURATION',
        )
    host = (parsed.hostname or '').lower()
    if api_key and parsed.scheme == 'http' and host not in _LOCAL_HOSTS:
        raise SdfmError(
            'refusing to send an API key over plaintext HTTP; use an https:// '
            f'URL (or a localhost endpoint), got {url!r}',
            code='INVALID_CONFIGURATION',
        )


class _Session(requests.Session):
    r"""A ``Session`` that drops ``X-API-Key`` on a cross-origin redirect.

    ``requests`` strips only the standard ``Authorization`` header when a
    redirect changes origin; a custom header set on the session is re-sent
    verbatim. Without this the configured endpoint could hand the API key --
    and, on a 307/308, the request body -- to any host it names, defeating
    ``_validate_url``.
    """

    def rebuild_auth(
        self,
        prepared_request: requests.PreparedRequest,
        response: requests.Response,
    ) -> None:
        super().rebuild_auth(prepared_request, response)
        previous_url = response.request.url
        if previous_url and prepared_request.url and self.should_strip_auth(
                previous_url, prepared_request.url):
            prepared_request.headers.pop('X-API-Key', None)


def _build_session(
    api_key: str | None,
    max_retries: int,
    backoff_factor: float,
) -> requests.Session:
    session = _Session()
    if api_key:
        session.headers['X-API-Key'] = api_key
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=_RETRY_STATUS,
        allowed_methods=frozenset({'GET', 'POST'}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


class Transport:
    r"""The HTTP wire layer to one Universal TFM NIM.

    Owns a pooled ``requests.Session`` with retry/backoff on transient
    failures (429/5xx). Not part of the public API surface; a ``SDFMClient``
    holds one of these.
    """

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        *,
        verify_ssl: bool = True,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
    ) -> None:
        _validate_url(url, api_key)
        self._url = url.rstrip('/')
        self._api_key = api_key
        self._verify_ssl = verify_ssl
        self._timeout = timeout
        self._session = _build_session(api_key, max_retries, backoff_factor)

    @property
    def url(self) -> str:
        return self._url

    @property
    def api_key(self) -> str | None:
        return self._api_key

    @property
    def verify_ssl(self) -> bool:
        return self._verify_ssl

    @property
    def timeout(self) -> float:
        return self._timeout

    def close(self) -> None:
        self._session.close()

    def health_ready(self) -> bool:
        response = self._session.get(
            self._url + _HEALTH_READY_PATH,
            timeout=self._timeout,
            verify=self._verify_ssl,
        )
        return response.status_code == 200

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(_PREDICTIONS_PATH, payload)

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(
            'Session endpoints are not wired yet; see the README',
        )

    def session_predict(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError(
            'Session endpoints are not wired yet; see the README',
        )

    def delete_session(self, session_id: str) -> None:
        raise NotImplementedError(
            'Session endpoints are not wired yet; see the README',
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._session.post(
                self._url + path,
                json=payload,
                timeout=self._timeout,
                verify=self._verify_ssl,
            )
        except requests.RequestException as error:
            raise SdfmError(
                f'Request to {self._url + path} failed: {error}',
                code='TRANSPORT_ERROR',
            ) from error
        if response.status_code >= 400:
            raise _to_nim_error(response)
        try:
            body = response.json()
        except ValueError as error:
            raise SdfmError(
                f'Invalid JSON response from {self._url + path}',
                code='TRANSPORT_ERROR',
            ) from error
        if not isinstance(body, dict):
            raise SdfmError(
                f'Expected a JSON object from {self._url + path}, '
                f'got {type(body).__name__}',
                code='TRANSPORT_ERROR',
            )
        return body


def _to_nim_error(response: requests.Response) -> NimRequestError:
    try:
        body = response.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    code = body.get('code')
    message = body.get('detail') or body.get('title') or response.text or (
        f'NIM request failed with status {response.status_code}'
    )
    details = {
        key: value
        for key, value in body.items()
        if key not in ('code', 'detail', 'title', 'type', 'status', 'instance')
    }
    return NimRequestError(
        response.status_code,
        code=code,
        message=message,
        details=details,
    )

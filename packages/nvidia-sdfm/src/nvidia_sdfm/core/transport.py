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
_RETRY_AFTER_MAX_SECONDS = 60
_MAX_BODY_SNIPPET = 512
_LOCAL_HOSTS = frozenset({'localhost', '127.0.0.1', '::1'})


def _validate_url(url: str, api_key: str | None) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        raise SdfmError(
            f'url must start with http:// or https://, got {url!r}',
            code='INVALID_CONFIGURATION',
        )
    if not parsed.hostname:
        raise SdfmError(
            f'url is missing a host, got {url!r}',
            code='INVALID_CONFIGURATION',
        )
    host = parsed.hostname.lower()
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

def _validate_limits(timeout: Any, max_retries: Any) -> None:
    r"""Reject unusable transport limits at construction.

    Without this the offending value first surfaces as a raw ``ValueError``
    from urllib3, on the first request rather than at the call that set it.
    """
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or timeout <= 0):
        raise SdfmError(
            f'timeout must be a positive number of seconds, got {timeout!r}',
            code='INVALID_CONFIGURATION',
        )
    if (isinstance(max_retries, bool) or not isinstance(max_retries, int)
            or max_retries < 0):
        raise SdfmError(
            f'max_retries must be a non-negative integer, got {max_retries!r}',
            code='INVALID_CONFIGURATION',
        )


def _build_retry(max_retries: int, backoff_factor: float) -> Retry:
    r"""The retry policy shared by every request this transport makes.

    ``retry_after_max`` caps how long a server-chosen ``Retry-After`` header
    may park the caller; it needs urllib3 >= 2.3, and older versions fall back
    to urllib3's own (6 hour) cap.
    """
    options: dict[str, Any] = {
        'total': max_retries,
        'connect': max_retries,
        'read': max_retries,
        'status': max_retries,
        'backoff_factor': backoff_factor,
        'status_forcelist': _RETRY_STATUS,
        'allowed_methods': frozenset({'GET', 'POST'}),
        'respect_retry_after_header': True,
        'raise_on_status': False,
    }
    try:
        return Retry(retry_after_max=_RETRY_AFTER_MAX_SECONDS, **options)
    except TypeError:
        return Retry(**options)


def _snippet(text: str) -> str:
    r"""A one-line, length-capped view of a response body, so a multi-megabyte
    HTML error page cannot become a multi-megabyte exception message.
    """
    text = ' '.join(text.split())
    if len(text) <= _MAX_BODY_SNIPPET:
        return text
    return (f'{text[:_MAX_BODY_SNIPPET]}... '
            f'[truncated, {len(text)} chars total]')


def _build_session(
    api_key: str | None,
    max_retries: int,
    backoff_factor: float,
) -> requests.Session:
    session = _Session()
    if api_key:
        session.headers['X-API-Key'] = api_key
    adapter = HTTPAdapter(max_retries=_build_retry(max_retries,
                                                   backoff_factor))
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


class Transport:
    r"""The HTTP wire layer to one Universal TFM NIM.

    Owns a pooled ``requests.Session`` with retry/backoff on transient
    failures (429/5xx). Not part of the public API surface; a ``SDFMClient``
    holds one of these.

    ``timeout`` bounds each individual attempt, not the call as a whole: a
    request that exhausts ``max_retries`` can take up to
    ``(max_retries + 1) * timeout`` plus backoff.
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
        _validate_limits(timeout, max_retries)
        self._url = url.rstrip('/')
        self._api_key = api_key
        self._verify_ssl = verify_ssl
        self._timeout = timeout
        self._closed = False
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
        self._closed = True
        self._session.close()

    def _require_open(self) -> None:
        if self._closed:
            raise SdfmError(
                'this client is closed; construct a new SDFMClient',
                code='INVALID_CONFIGURATION',
            )

    def health_ready(self) -> bool:
        self._require_open()
        try:
            response = self._session.get(
                self._url + _HEALTH_READY_PATH,
                timeout=self._timeout,
                verify=self._verify_ssl,
            )
        except requests.RequestException as error:
            raise SdfmError(
                f'Request to {self._url + _HEALTH_READY_PATH} failed: {error}',
                code='TRANSPORT_ERROR',
            ) from error
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
        self._require_open()
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
    reported = body.get('detail') or body.get('title') or response.text
    message = _snippet(str(reported)) if reported else (
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

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import re
import warnings
from typing import Any
from urllib.parse import quote, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from nemotron_structured.errors import (
    NimRequestError,
    StructuredError,
    format_invalid_params,
)

_PREDICTIONS_PATH = '/v1/predictions'
_SESSIONS_PATH = '/v1/sessions'
_HEALTH_READY_PATH = '/v1/health/ready'
_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_FACTOR = 0.5
_RETRY_STATUS = (429, 500, 502, 503, 504)
_RETRY_AFTER_MAX_SECONDS = 60
_MAX_BODY_SNIPPET = 512
_LOCAL_HOSTS = frozenset({'localhost', '127.0.0.1', '::1'})
# Well above any real prediction response -- a full-precision float64 column
# for a million rows is ~25 MB of JSON -- but far below what a compressed
# hostile body can inflate to.
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_RESPONSE_CHUNK_BYTES = 1024 * 1024
# Deleting a session is a bookkeeping call the server answers immediately, and
# it runs on cleanup paths where the caller is no longer waiting on a result.
# The full request timeout would let an unreachable endpoint stall a teardown.
_DELETE_TIMEOUT_SECONDS = 10.0
_USERINFO_RE = re.compile(r'(?<=//)[^/@\s]+@')


def redact_url(url: str | None) -> str | None:
    r"""Strip any userinfo from ``url`` so it is safe to log or display.

    ``https://user:token@host/path`` is a legal way to point this client at a
    deployment, and the credential is then in the URL itself rather than in
    ``api_key``. Anywhere a URL reaches an error message or a repr it goes
    through here first, so the credential does not travel with the diagnosis
    into a log aggregator or a bug report.

    Mirrors ``nemotron_relational.client.client.redact_url``, which guards the same
    credential on the Nemotron Relational path. The two cannot share one implementation:
    this package must not import the optional driver, and the package both
    depend on is a SQL-connector package with no URL handling. Keep them in
    step.
    """
    if not url:
        return url
    parsed = urlparse(url)
    if not parsed.username and not parsed.password:
        return url
    host = parsed.hostname or ''
    if ':' in host:
        host = f'[{host}]'
    if parsed.port:
        host = f'{host}:{parsed.port}'
    return parsed._replace(netloc=host).geturl()


def scrub_userinfo(text: str) -> str:
    r"""Remove any ``user:secret@`` credential from free text.

    The counterpart to :func:`redact_url` for strings that merely *contain* a
    URL rather than being one -- chiefly the ``requests`` exception messages
    quoted into transport errors, which echo back whatever URL was requested.
    """
    return _USERINFO_RE.sub('', text)


def _warn_if_verification_disabled(url: str, verify_ssl: bool) -> None:
    """Warn when TLS verification is turned off against a real host.

    A local NIM over a self-signed certificate is a normal thing to do, so
    loopback stays quiet. Anywhere else the connection can be intercepted, and
    an example copied out of an internal environment is how that reaches
    production.
    """
    if verify_ssl:
        return
    host = (urlparse(url).hostname or '').lower()
    if host in _LOCAL_HOSTS:
        return
    warnings.warn(
        f'TLS certificate verification is disabled for {redact_url(url)}. '
        'The connection can be intercepted and any API key read off it. '
        'Set verify_ssl=True and trust the endpoint certificate instead.',
        stacklevel=3,
    )


def _validate_url(url: str, api_key: str | None) -> None:
    if not isinstance(url, str):
        raise StructuredError(
            f'url must be a string, got {type(url).__name__}',
            code='INVALID_CONFIGURATION',
        )
    parsed = urlparse(url)
    shown = redact_url(url)
    if parsed.scheme not in ('http', 'https'):
        raise StructuredError(
            f'url must start with http:// or https://, got {shown!r}',
            code='INVALID_CONFIGURATION',
        )
    if not parsed.hostname:
        raise StructuredError(
            f'url is missing a host, got {shown!r}',
            code='INVALID_CONFIGURATION',
        )
    host = parsed.hostname.lower()
    if api_key and parsed.scheme == 'http' and host not in _LOCAL_HOSTS:
        raise StructuredError(
            'refusing to send an API key over plaintext HTTP; use an https:// '
            f'URL (or a localhost endpoint), got {shown!r}',
            code='INVALID_CONFIGURATION',
        )


class _Session(requests.Session):
    r"""A ``Session`` that drops ``X-API-Key`` on a cross-origin redirect.

    ``requests`` strips only the standard ``Authorization`` header when a
    redirect changes origin; a custom header set on the session is re-sent
    verbatim, so without this the configured endpoint could hand the API key
    to any host it names, defeating ``_validate_url``.

    Redirects are still followed, and this covers the credential only. A
    ``307``/``308`` therefore re-sends the request *body* -- for Nemotron Tabular, the
    caller's whole labelled context table -- to the redirect target, including
    across an ``https`` to ``http`` downgrade, which ``_validate_url`` never
    sees because it runs at construction against the configured URL. That is
    an accepted trade: refusing redirects outright breaks legitimate ``308``
    normalisation, and the endpoint being redirected *from* was already trusted
    with the same payload. A caller who cannot accept it should point the
    client at a URL that does not redirect.
    """

    def rebuild_auth(
        self,
        prepared_request: requests.PreparedRequest,
        response: requests.Response,
    ) -> None:
        super().rebuild_auth(prepared_request, response)
        previous_url = response.request.url
        if (
            previous_url
            and prepared_request.url
            and self.should_strip_auth(previous_url, prepared_request.url)
        ):
            prepared_request.headers.pop('X-API-Key', None)


def _validate_limits(timeout: Any, max_retries: Any) -> None:
    r"""Reject unusable transport limits at construction.

    Without this the offending value first surfaces as a raw ``ValueError``
    from urllib3, on the first request rather than at the call that set it.

    ``inf`` and ``nan`` are rejected explicitly: both compare ``False`` against
    ``<= 0`` and so pass an ordinary positivity test, then fail on the first
    request with ``OverflowError``/``ValueError`` from the socket layer. ``inf``
    is the natural way to reach for "no timeout" here, since this constructor
    does not accept ``None``.
    """
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise StructuredError(
            f'timeout must be a positive, finite number of seconds, got '
            f'{timeout!r}',
            code='INVALID_CONFIGURATION',
        )
    if (
        isinstance(max_retries, bool)
        or not isinstance(max_retries, int)
        or max_retries < 0
    ):
        raise StructuredError(
            f'max_retries must be a non-negative integer, got {max_retries!r}',
            code='INVALID_CONFIGURATION',
        )


def _build_retry(
    max_retries: int,
    backoff_factor: float,
    *,
    retry_post: bool = True,
) -> Retry:
    r"""The retry policy this transport's requests run under.

    ``retry_after_max`` caps how long a server-chosen ``Retry-After`` header
    may park the caller; it needs urllib3 >= 2.3, and older versions fall back
    to urllib3's own (6 hour) cap.

    ``retry_post=False`` drops ``POST`` from the retryable methods, which
    switches off the read and status retries for a route whose side effect the
    client cannot reconcile. Connection failures stay retryable either way:
    urllib3 gates read and status retries on the method but not connect ones,
    and a request that never reached the server cannot have had an effect.
    """
    methods = {'GET', 'POST'} if retry_post else {'GET'}
    options: dict[str, Any] = {
        'total': max_retries,
        'connect': max_retries,
        'read': max_retries,
        'status': max_retries,
        'backoff_factor': backoff_factor,
        'status_forcelist': _RETRY_STATUS,
        'allowed_methods': frozenset(methods),
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
    return f'{text[:_MAX_BODY_SNIPPET]}... [truncated, {len(text)} chars total]'


def _read_capped(response: requests.Response, url: str) -> bytes:
    r"""Reads a streamed response body, refusing anything past the cap.

    ``requests`` inflates ``Content-Encoding: gzip`` with no ratio limit, so
    an unbounded read lets a small compressed body expand into hundreds of
    megabytes of client memory before anything is parsed.
    """
    body = bytearray()
    for chunk in response.iter_content(_RESPONSE_CHUNK_BYTES):
        if len(body) + len(chunk) > _MAX_RESPONSE_BYTES:
            raise StructuredError(
                f'Response body from {redact_url(url)} exceeds the '
                f'{_MAX_RESPONSE_BYTES} byte limit',
                code='TRANSPORT_ERROR',
            )
        body.extend(chunk)
    return bytes(body)


def _path_segment(value: str) -> str:
    r"""Escapes a server-chosen id before it is spliced into a URL path, so a
    hostile or malformed ``session_id`` cannot redirect the call to another
    route.
    """
    return quote(value, safe='')


def _build_session(
    url: str,
    api_key: str | None,
    max_retries: int,
    backoff_factor: float,
) -> requests.Session:
    r"""The pooled session, with ``POST /v1/sessions`` held out of the retries.

    Creating a session fits and pins the context in the worker's memory before
    the response is written, so a re-sent ``POST`` leaves one orphaned pinned
    context per attempt: the client keeps only the last ``session_id`` and can
    never ``DELETE`` the earlier ones. That is a side effect the client cannot
    reconcile, unlike ``/v1/predictions``, which the same policy may safely
    replay.

    ``requests`` resolves an adapter by longest matching URL prefix, so the
    stricter policy is mounted on the exact create-session URL and the ordinary
    one re-mounted on the routes below it: scoring against a pinned context and
    releasing one are both replayable and keep the full policy.
    """
    session = _Session()
    if api_key:
        session.headers['X-API-Key'] = api_key
    adapter = HTTPAdapter(max_retries=_build_retry(max_retries, backoff_factor))
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.mount(
        url + _SESSIONS_PATH,
        HTTPAdapter(
            max_retries=_build_retry(
                max_retries, backoff_factor, retry_post=False
            )
        ),
    )
    session.mount(url + _SESSIONS_PATH + '/', adapter)
    return session


class Transport:
    r"""The HTTP wire layer to one Universal TFM NIM.

    Owns a pooled ``requests.Session`` with retry/backoff on transient
    failures (429/5xx). Not part of the public API surface; a ``StructuredClient``
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
        _warn_if_verification_disabled(url, verify_ssl)
        self._url = url.rstrip('/')
        self._api_key = api_key
        self._verify_ssl = verify_ssl
        self._timeout = timeout
        self._closed = False
        self._max_retries = max_retries
        self._session = _build_session(
            self._url, api_key, max_retries, backoff_factor
        )

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

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def close(self) -> None:
        self._closed = True
        self._session.close()

    def _require_open(self) -> None:
        if self._closed:
            raise StructuredError(
                'this client is closed; construct a new StructuredClient',
                code='INVALID_CONFIGURATION',
            )

    def health_ready(self) -> bool:
        self._require_open()
        try:
            # Streamed and closed unread: only the status matters, and the
            # body is attacker-controlled on a misconfigured endpoint.
            with self._session.get(
                self._url + _HEALTH_READY_PATH,
                timeout=self._timeout,
                verify=self._verify_ssl,
                stream=True,
            ) as response:
                return response.status_code == 200
        except requests.RequestException as error:
            raise StructuredError(
                f'Request to '
                f'{redact_url(self._url + _HEALTH_READY_PATH)} failed: '
                f'{scrub_userinfo(str(error))}',
                code='TRANSPORT_ERROR',
            ) from error

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(_PREDICTIONS_PATH, payload)

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        r"""Pins a context server-side; ``payload`` carries the context-only
        sections of a prediction request.
        """
        return self._post(_SESSIONS_PATH, payload)

    def session_predict(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        r"""Scores rows against a pinned context; ``payload`` carries only the
        per-call sections.
        """
        return self._post(
            f'{_SESSIONS_PATH}/{_path_segment(session_id)}/predictions', payload
        )

    def delete_session(self, session_id: str) -> None:
        r"""Releases a pinned context. Idempotent server-side: the NIM answers
        204 whether or not the session is still there.
        """
        self._require_open()
        url = f'{self._url}{_SESSIONS_PATH}/{_path_segment(session_id)}'
        try:
            with self._session.delete(
                url,
                timeout=min(self._timeout, _DELETE_TIMEOUT_SECONDS),
                verify=self._verify_ssl,
                stream=True,
            ) as response:
                status_code = response.status_code
                content = _read_capped(response, url)
        except requests.RequestException as error:
            raise StructuredError(
                f'Request to {redact_url(url)} failed: '
                f'{scrub_userinfo(str(error))}',
                code='TRANSPORT_ERROR',
            ) from error
        if status_code >= 400:
            raise _to_nim_error(status_code, content)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_open()
        url = self._url + path
        try:
            with self._session.post(
                url,
                json=payload,
                timeout=self._timeout,
                verify=self._verify_ssl,
                stream=True,
            ) as response:
                status_code = response.status_code
                content = _read_capped(response, url)
        except requests.RequestException as error:
            raise StructuredError(
                f'Request to {redact_url(url)} failed: '
                f'{scrub_userinfo(str(error))}',
                code='TRANSPORT_ERROR',
            ) from error
        if status_code >= 400:
            raise _to_nim_error(status_code, content)
        try:
            # A deeply nested body raises RecursionError, which is a
            # RuntimeError and would otherwise escape the StructuredError contract.
            body = json.loads(content)
        except (ValueError, RecursionError) as error:
            raise StructuredError(
                f'Invalid JSON response from {redact_url(url)}',
                code='TRANSPORT_ERROR',
            ) from error
        if not isinstance(body, dict):
            raise StructuredError(
                f'Expected a JSON object from {redact_url(url)}, got '
                f'{type(body).__name__}',
                code='TRANSPORT_ERROR',
            )
        return body


def _to_nim_error(status_code: int, content: bytes) -> NimRequestError:
    try:
        body = json.loads(content)
    except (ValueError, RecursionError):
        body = {}
    if not isinstance(body, dict):
        body = {}
    code = body.get('code')
    reported = (
        body.get('detail')
        or body.get('title')
        or content.decode('utf-8', 'replace')
    )
    message = (
        _snippet(str(reported))
        if reported
        else (f'NIM request failed with status {status_code}')
    )
    # A validation failure's top-level detail is often only "Request validation
    # failed."; the per-field diagnosis is in invalid_params, so render it into
    # the message rather than leaving it for the caller to dig out of details.
    message += format_invalid_params(body.get('invalid_params'))
    details = {
        key: value
        for key, value in body.items()
        if key not in ('code', 'detail', 'title', 'type', 'status', 'instance')
    }
    return NimRequestError(
        status_code,
        code=code,
        message=message,
        details=details,
    )

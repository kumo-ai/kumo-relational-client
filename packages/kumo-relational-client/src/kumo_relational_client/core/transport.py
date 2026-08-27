# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import re
import warnings
from typing import Any
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from kumo_relational_client.errors import (
    RelationalError,
)

_HEALTH_READY_PATH = '/v1/health/ready'
_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_FACTOR = 0.5
_RETRY_STATUS = (429, 500, 502, 503, 504)
_RETRY_AFTER_MAX_SECONDS = 60
_LOCAL_HOSTS = frozenset({'localhost', '127.0.0.1', '::1'})
# Well above any real prediction response -- a full-precision float64 column
# for a million rows is ~25 MB of JSON -- but far below what a compressed
# hostile body can inflate to.
_USERINFO_RE = re.compile(r'(?<=//)[^/@\s]+@')


def redact_url(url: str | None) -> str | None:
    r"""Strip any userinfo from ``url`` so it is safe to log or display.

    ``https://user:token@host/path`` is a legal way to point this client at a
    deployment, and the credential is then in the URL itself rather than in
    ``api_key``. Anywhere a URL reaches an error message or a repr it goes
    through here first, so the credential does not travel with the diagnosis
    into a log aggregator or a bug report.

    Mirrors ``kumo_relational_engine.client.client.redact_url``, which guards the same
    credential on the Kumo Relational path. The two cannot share one implementation:
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
        raise RelationalError(
            f'url must be a string, got {type(url).__name__}',
            code='INVALID_CONFIGURATION',
        )
    parsed = urlparse(url)
    shown = redact_url(url)
    if parsed.scheme not in ('http', 'https'):
        raise RelationalError(
            f'url must start with http:// or https://, got {shown!r}',
            code='INVALID_CONFIGURATION',
        )
    if not parsed.hostname:
        raise RelationalError(
            f'url is missing a host, got {shown!r}',
            code='INVALID_CONFIGURATION',
        )
    host = parsed.hostname.lower()
    if api_key and parsed.scheme == 'http' and host not in _LOCAL_HOSTS:
        raise RelationalError(
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

    Redirects are still followed, and their bodies are discarded unread rather
    than consumed, so a redirect chain cannot be used to make this layer read an
    unbounded response. This covers the credential only. A
    ``307``/``308`` therefore re-sends the request *body* -- the caller's whole
    context -- to the redirect target, including
    across an ``https`` to ``http`` downgrade, which ``_validate_url`` never
    sees because it runs at construction against the configured URL. That is
    an accepted trade: refusing redirects outright breaks legitimate ``308``
    normalisation, and the endpoint being redirected *from* was already trusted
    with the same payload. A caller who cannot accept it should point the
    client at a URL that does not redirect.
    """

    def resolve_redirects(  # type: ignore[override]
        self,
        resp: requests.Response,
        req: requests.PreparedRequest,
        **kwargs: Any,
    ) -> Any:
        r"""Follow redirects without reading the bodies they carry.

        ``requests`` consumes each hop's body (``resp.content``) to release the
        socket, with no cap. This layer never needs those bytes -- the only
        request it makes reports a status -- and they are attacker-controlled on
        a misconfigured endpoint, so the socket is released by closing it
        instead of by reading it.
        """
        # Close rather than drain: the socket is given up instead of being
        # read to the end so it can be reused. Redirects are rare here, so
        # trading connection reuse for a bounded read is the right way round.
        resp.raw.close()
        resp._content = b''
        resp._content_consumed = True  # type: ignore[attr-defined]
        return super().resolve_redirects(resp, req, **kwargs)

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
        raise RelationalError(
            f'timeout must be a positive, finite number of seconds, got '
            f'{timeout!r}',
            code='INVALID_CONFIGURATION',
        )
    if (
        isinstance(max_retries, bool)
        or not isinstance(max_retries, int)
        or max_retries < 0
    ):
        raise RelationalError(
            f'max_retries must be a non-negative integer, got {max_retries!r}',
            code='INVALID_CONFIGURATION',
        )


def _build_retry(max_retries: int, backoff_factor: float) -> Retry:
    r"""The retry policy this transport's requests run under.

    ``retry_after_max`` caps how long a server-chosen ``Retry-After`` header
    may park the caller; it needs urllib3 >= 2.3, and older versions fall back
    to urllib3's own (6 hour) cap.

    Only ``GET`` is retryable, which is every request this transport makes.
    Connection failures stay retryable regardless: urllib3 gates read and status
    retries on the method but not connect ones, and a request that never reached
    the server cannot have had an effect.
    """
    methods = {'GET'}
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
    return session


class Transport:
    r"""The HTTP wire layer to one Universal TFM NIM.

    Owns a pooled ``requests.Session`` with retry/backoff on transient
    failures (429/5xx). Not part of the public API surface; a ``RelationalClient``
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
            raise RelationalError(
                'this client is closed; construct a new RelationalClient',
                code='INVALID_CONFIGURATION',
            )

    def health_ready(self) -> bool:
        self._require_open()
        try:
            # Streamed and closed unread: only the status matters, and the
            # body is attacker-controlled on a misconfigured endpoint.
            # Redirect hops are released by closing rather than reading, in
            # `_Session.resolve_redirects`. One gap remains: urllib3 drains a
            # retryable 4xx/5xx body before retrying it, so a hostile endpoint
            # can still be read up to `max_retries` times. Closing that would
            # mean not retrying a status here at all, which is a change to the
            # documented retry contract rather than an implementation detail.
            with self._session.get(
                self._url + _HEALTH_READY_PATH,
                timeout=self._timeout,
                verify=self._verify_ssl,
                stream=True,
            ) as response:
                return response.status_code == 200
        except requests.RequestException as error:
            raise RelationalError(
                f'Request to '
                f'{redact_url(self._url + _HEALTH_READY_PATH)} failed: '
                f'{scrub_userinfo(str(error))}',
                code='TRANSPORT_ERROR',
            ) from error

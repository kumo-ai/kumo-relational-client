# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Transport behaviour that only a real socket can show.

``requests_mock`` mounts its own adapter, which bypasses the urllib3 ``Retry``
policy and the socket timeout entirely, so retry and timeout tests here talk to
a local ``http.server`` instead.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import urllib3.util.retry

from kumo_relational_client.core.transport import (
    _RETRY_AFTER_MAX_SECONDS,
    Transport,
    _build_retry,
)
from kumo_relational_client.errors import RelationalError

_REFUSED_URL = 'http://127.0.0.1:1'
_RETRY_AFTER_SECONDS = 3600
_RETRY_AFTER_CAP_APPLIES = (
    getattr(_build_retry(1, 0.0), 'retry_after_max', None)
    == _RETRY_AFTER_MAX_SECONDS
)


def _predictions_body(size: int) -> bytes:
    r"""A syntactically valid prediction response of roughly ``size`` bytes."""
    filler = b'0,' * (max(size - 18, 0) // 2)
    return b'{"predictions":[' + filler + b'0]}'


@dataclass
class _Reply:
    status: int = 200
    body: str = '{"predictions": []}'
    raw: bytes | None = None
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0.0


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_GET(self) -> None:
        self._reply()

    def do_POST(self) -> None:
        self._reply()

    def log_message(self, *args: object) -> None:
        pass

    def _reply(self) -> None:
        length = int(self.headers.get('Content-Length') or 0)
        if length:
            self.rfile.read(length)
        script = self.server.script
        reply = script.pop(0) if len(script) > 1 else script[0]
        self.server.received.append(self.path)
        if reply.delay:
            time.sleep(reply.delay)
        body = reply.raw if reply.raw is not None else reply.body.encode()
        try:
            self.send_response(reply.status)
            for name, value in reply.headers.items():
                self.send_header(name, value)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            pass


@contextmanager
def _serve(*script: _Reply) -> Iterator[tuple[ThreadingHTTPServer, str]]:
    server = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
    server.script = list(script)
    server.received = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()


def test_retry_policy_retries_transient_status_on_a_real_socket():
    with _serve(_Reply(status=503), _Reply(status=503), _Reply()) as (
        server,
        url,
    ):
        transport = Transport(url, max_retries=3, backoff_factor=0.0)
        assert transport.health_ready() is True
    assert server.received == ['/v1/health/ready'] * 3


def test_retry_after_header_is_capped(monkeypatch):
    r"""A server-chosen ``Retry-After`` must not park the caller for an hour.

    ``Transport._build_retry`` can only cap it on urllib3 >= 2.3; older
    releases have no ``retry_after_max`` and honour the header in full, so the
    expected sleep follows the policy the transport actually built.
    """
    slept: list[float] = []
    monkeypatch.setattr(urllib3.util.retry.time, 'sleep', slept.append)

    with _serve(
        _Reply(status=503, headers={'Retry-After': str(_RETRY_AFTER_SECONDS)}),
        _Reply(),
    ) as (_, url):
        Transport(url, max_retries=1, backoff_factor=0.0).health_ready()

    assert slept == [
        _RETRY_AFTER_MAX_SECONDS
        if _RETRY_AFTER_CAP_APPLIES
        else _RETRY_AFTER_SECONDS
    ]


def test_timeout_bounds_a_slow_response():
    with _serve(_Reply(delay=5.0)) as (_, url):
        transport = Transport(url, timeout=0.5, max_retries=0)
        started = time.monotonic()
        with pytest.raises(RelationalError) as excinfo:
            transport.health_ready()
        elapsed = time.monotonic() - started
    assert excinfo.value.code == 'TRANSPORT_ERROR'
    assert elapsed < 4.0


def test_health_ready_wraps_unreachable_host():
    transport = Transport(_REFUSED_URL, timeout=2.0, max_retries=0)
    with pytest.raises(RelationalError) as excinfo:
        transport.health_ready()
    assert excinfo.value.code == 'TRANSPORT_ERROR'


def test_closed_transport_stops_issuing_requests():
    with _serve(_Reply()) as (server, url):
        transport = Transport(url, max_retries=0)
        transport.health_ready()
        transport.close()
        with pytest.raises(RelationalError) as excinfo:
            transport.health_ready()
    assert excinfo.value.code == 'INVALID_CONFIGURATION'
    assert server.received == ['/v1/health/ready']

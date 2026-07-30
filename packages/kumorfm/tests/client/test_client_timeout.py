# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The socket timeout the SDK client configures must reach the wire.

``requests_mock`` mounts its own adapter and never opens a socket, so these
tests talk to a local ``http.server`` that stalls before replying.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

import pytest
import requests

import kumorfm
import kumorfm.rfm as rfm_engine
from kumorfm.client import KumoClient

_READY_BODY = json.dumps({'status': 'healthy', 'check': 'ready'})
_MODELS_BODY = json.dumps({'object': 'list', 'data': [{'id': 'kumo-rfm'}]})


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
        time.sleep(self.server.delay)
        body = self.server.bodies.get(self.path, '{}').encode()
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            pass


@contextmanager
def _serve(delay: float = 0.0) -> Iterator[str]:
    server = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
    server.delay = delay
    server.bodies = {
        '/v1/health/ready': _READY_BODY,
        '/v1/models': _MODELS_BODY,
    }
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()


def test_post_honours_the_configured_timeout():
    with _serve(delay=5.0) as url:
        client = KumoClient(url, timeout=0.5)
        started = time.monotonic()
        with pytest.raises(requests.Timeout):
            client._post('/v1/predictions', json={})
        elapsed = time.monotonic() - started
    assert elapsed < 4.0


def test_per_call_timeout_overrides_the_client_default():
    with _serve(delay=5.0) as url:
        client = KumoClient(url, timeout=30.0)
        with pytest.raises(requests.Timeout):
            client._post('/v1/predictions', json={}, timeout=0.5)


def test_init_forwards_the_timeout_to_the_request_client():
    try:
        with _serve() as url:
            rfm_engine.init(url=url, timeout=7.0,
                            _token=rfm_engine._SDFM_CLIENT_TOKEN)
            assert kumorfm.global_state.client._timeout == 7.0
    finally:
        rfm_engine.global_state.reset()
        kumorfm.global_state.clear()

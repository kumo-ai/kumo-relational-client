# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""``KumoClient`` carries every KumoRFM prediction, so it must enforce the
credential guard the other client documents, and importing the package must
not reach the network. See
``bugs/security-kumoclient-import-side-effect-and-missing-plaintext-guard.md``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

import pytest
import requests

from kumorfm.client.client import KumoClient, _raise_init_error

_READY = b'{"status": "ready"}'
_MODELS = b'{"object": "list", "data": [{"id": "kumo-rfm"}]}'


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_GET(self) -> None:
        self.server.received.append(self.path)
        body = _MODELS if self.path.endswith('/v1/models') else _READY
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@contextmanager
def _serve() -> Iterator[tuple[ThreadingHTTPServer, str]]:
    server = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
    server.received = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()


def test_plaintext_url_with_an_api_key_is_refused() -> None:
    with pytest.raises(ValueError, match='plaintext HTTP'):
        KumoClient('http://nim.test:8000', api_key='SECRET')


@pytest.mark.parametrize('url', [
    'https://nim.test:8000',
    'http://localhost:8000',
    'http://127.0.0.1:8000',
])
def test_supported_endpoints_still_accept_an_api_key(url: str) -> None:
    assert KumoClient(url, api_key='SECRET')._url == url


def test_plaintext_url_without_an_api_key_is_allowed() -> None:
    assert KumoClient('http://nim.test:8000')._url == 'http://nim.test:8000'


@pytest.mark.parametrize('url', ['nim.test:8000', 'ftp://nim.test', 'http://'])
def test_unusable_urls_are_refused(url: str) -> None:
    with pytest.raises(ValueError):
        KumoClient(url)


def test_init_error_body_is_truncated() -> None:
    response = requests.Response()
    response.status_code = 500
    response._content = b'x' * 8_000_000
    error = requests.exceptions.HTTPError(response=response)

    with pytest.raises(ValueError) as excinfo:
        _raise_init_error('https://nim.test', error)

    assert len(str(excinfo.value)) < 2_000
    assert 'truncated' in str(excinfo.value)


def test_importing_the_package_does_not_connect() -> None:
    with _serve() as (server, url):
        env = dict(os.environ)
        env['KUMO_API_ENDPOINT'] = url
        env['PYTHONPATH'] = os.pathsep.join(sys.path)
        result = subprocess.run(
            [sys.executable, '-c', 'import kumorfm'],
            env=env,
            capture_output=True,
            timeout=120,
        )

    assert result.returncode == 0, result.stderr.decode()
    assert server.received == []


def test_endpoint_env_var_still_initializes_on_first_use() -> None:
    r"""Dropping the import-time ``init()`` must not drop the env-var contract.

    Callers who set ``KUMO_API_ENDPOINT`` and never call ``init()`` explicitly
    kept working before; the connection just moves from import time to first
    use, where a network call is expected.
    """
    with _serve() as (server, url):
        env = dict(os.environ)
        env['KUMO_API_ENDPOINT'] = url
        env['PYTHONPATH'] = os.pathsep.join(sys.path)
        result = subprocess.run(
            [sys.executable, '-c',
             'import kumorfm; print(kumorfm.global_state.client._url)'],
            env=env,
            capture_output=True,
            timeout=120,
        )
        received = list(server.received)

    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.decode().strip() == url
    assert received, 'first use should have authenticated'


def test_blank_endpoint_env_var_does_not_start_a_connection() -> None:
    r"""An empty ``KUMO_API_ENDPOINT`` means unconfigured, not a URL.

    A key-presence check treats ``KUMO_API_ENDPOINT=""`` as configured and
    routes the caller into a URL-parse error instead of saying no endpoint was
    given. A non-blank value that is not a URL still reaches URL validation,
    which names the offending value.
    """
    env = dict(os.environ)
    env['KUMO_API_ENDPOINT'] = ''
    env['PYTHONPATH'] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [sys.executable, '-c', 'import kumorfm; kumorfm.global_state.client'],
        env=env,
        capture_output=True,
        timeout=120,
    )

    stderr = result.stderr.decode()
    assert result.returncode != 0
    assert 'must start with http' not in stderr, stderr
    assert 'Client creation or authentication failed' in stderr, stderr


def test_explicit_init_with_a_blank_endpoint_names_the_env_var() -> None:
    r"""``init()`` resolves the variable itself, so the same blank value has
    to reach its own missing-endpoint message rather than URL validation.
    """
    env = dict(os.environ)
    env['KUMO_API_ENDPOINT'] = ''
    env['PYTHONPATH'] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [sys.executable, '-c', 'import kumorfm; kumorfm.init()'],
        env=env,
        capture_output=True,
        timeout=120,
    )

    stderr = result.stderr.decode()
    assert result.returncode != 0
    assert 'no endpoint URL was provided' in stderr, stderr


def test_ambient_endpoint_does_not_connect_under_pytest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""A test run must never be connected to a real endpoint by accident.

    The removed import-time ``init()`` skipped itself under pytest; the lazy
    path has to keep that guard, or an exported ``KUMO_API_ENDPOINT`` would
    silently point somebody's test suite at a live NIM.
    """
    import kumorfm

    with _serve() as (server, url):
        monkeypatch.setenv('KUMO_API_ENDPOINT', url)
        monkeypatch.setattr(kumorfm.global_state, '_url', None, raising=False)
        with pytest.raises(ValueError, match='Client creation'):
            kumorfm.global_state.client
        assert server.received == []

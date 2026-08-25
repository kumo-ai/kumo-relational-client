# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import threading
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest
from urllib3.util.retry import Retry

from nemotron_structured import StructuredClient
from nemotron_structured.core import transport as transport_module
from nemotron_structured.core.transport import _PREDICTIONS_PATH, Transport
from nemotron_structured.errors import NimRequestError, StructuredError
from nemotron_structured.requests import ModelRequest, NemotronTabularRequest

_URL = 'http://nim.example.com:8000'


def _canned_response(request_id: str = 'pred_123') -> dict:
    return {
        'id': request_id,
        'model': 'kumo-tabular',
        'predictions': [
            {
                'row_index': 0,
                'prediction': 'yes',
                'probabilities': {'yes': 0.8, 'no': 0.2},
            },
            {
                'row_index': 1,
                'prediction': 'no',
                'probabilities': {'yes': 0.3, 'no': 0.7},
            },
        ],
        'metadata': {'task_kind': 'classification'},
    }


def test_predict_end_to_end_round_trip(requests_mock, context_df, predict_df):
    requests_mock.post(_URL + '/v1/predictions', json=_canned_response())

    with StructuredClient(url=_URL) as client:
        frame = client._predict(
            NemotronTabularRequest(
                context=context_df,
                predict=predict_df,
                task='classification',
                target='target_col',
                outputs=['prediction', 'probabilities'],
            )
        )

    assert list(frame['row_index']) == [0, 1]
    assert list(frame['prediction']) == ['yes', 'no']

    sent_payload = requests_mock.last_request.json()
    assert sent_payload['model'] == 'kumo-tabular'
    assert sent_payload['task']['target']['column_name'] == 'target_col'


def test_predict_unknown_model_raises():
    @dataclass
    class _UnknownRequest(ModelRequest):
        model: ClassVar[str] = 'not-a-real-model'

    with pytest.raises(StructuredError):
        StructuredClient(url=_URL)._predict(_UnknownRequest())


def test_predict_wrong_request_type_raises():
    @dataclass
    class _MislabelledRequest(ModelRequest):
        model: ClassVar[str] = 'kumo-tabular'

    with pytest.raises(StructuredError) as excinfo:
        StructuredClient(url=_URL)._predict(_MislabelledRequest())
    assert excinfo.value.code == 'INVALID_REQUEST'


def test_predict_propagates_nim_error(requests_mock, context_df, predict_df):
    requests_mock.post(
        _URL + '/v1/predictions',
        status_code=422,
        json={
            'type': 'about:blank',
            'status': 422,
            'title': 'Unprocessable Entity',
            'detail': 'schema validation failed',
            'code': 'INVALID_SCHEMA',
        },
    )

    with pytest.raises(NimRequestError) as excinfo:
        StructuredClient(url=_URL)._predict(
            NemotronTabularRequest(
                context=context_df,
                predict=predict_df,
                task='classification',
                target='target_col',
                outputs=['prediction'],
            )
        )
    assert excinfo.value.status_code == 422
    assert excinfo.value.code == 'INVALID_SCHEMA'


def test_health_ready_true_on_200(requests_mock):
    requests_mock.get(_URL + '/v1/health/ready', status_code=200, json={})
    client = Transport(_URL)
    assert client.health_ready() is True


def test_health_ready_false_on_503(requests_mock):
    requests_mock.get(_URL + '/v1/health/ready', status_code=503, json={})
    client = Transport(_URL)
    assert client.health_ready() is False


def test_client_sends_api_key_header(requests_mock):
    https_url = 'https://nim.example.com:8000'
    requests_mock.post(https_url + '/v1/predictions', json=_canned_response())
    client = Transport(https_url, api_key='secret')
    client.predict({'model': 'kumo-tabular'})
    assert requests_mock.last_request.headers['X-API-Key'] == 'secret'


def test_api_key_over_plaintext_http_is_rejected():
    with pytest.raises(StructuredError) as excinfo:
        Transport('http://nim.example.com:8000', api_key='secret')
    assert excinfo.value.code == 'INVALID_CONFIGURATION'


def test_api_key_over_localhost_http_is_allowed():
    assert Transport('http://localhost:8000', api_key='secret').url


def test_api_key_over_https_is_allowed():
    assert Transport('https://nim.example.com', api_key='secret').url


def test_no_api_key_over_http_is_allowed():
    assert Transport('http://nim.example.com:8000').url


def test_non_http_scheme_is_rejected():
    with pytest.raises(StructuredError) as excinfo:
        Transport('ftp://nim.example.com')
    assert excinfo.value.code == 'INVALID_CONFIGURATION'


def test_non_string_url_is_rejected_at_construction():
    with pytest.raises(StructuredError) as excinfo:
        Transport(123)  # type: ignore[arg-type]
    assert excinfo.value.code == 'INVALID_CONFIGURATION'
    assert 'url must be a string' in str(excinfo.value)


def test_invalid_json_success_response_raises_transport_error(requests_mock):
    requests_mock.post(_URL + '/v1/predictions', text='not json')
    client = Transport(_URL)
    with pytest.raises(StructuredError) as excinfo:
        client.predict({'model': 'kumo-tabular'})
    assert excinfo.value.code == 'TRANSPORT_ERROR'


def test_non_object_json_success_response_raises_transport_error(requests_mock):
    requests_mock.post(_URL + '/v1/predictions', json=['a', 'b'])
    client = Transport(_URL)
    with pytest.raises(StructuredError) as excinfo:
        client.predict({'model': 'kumo-tabular'})
    assert excinfo.value.code == 'TRANSPORT_ERROR'


def test_non_object_json_error_body_is_handled(requests_mock):
    requests_mock.post(_URL + '/v1/predictions', status_code=500, json=['boom'])
    client = Transport(_URL)
    with pytest.raises(NimRequestError) as excinfo:
        client.predict({'model': 'kumo-tabular'})
    assert excinfo.value.status_code == 500


def test_session_endpoints_use_the_contract_routes(requests_mock):
    requests_mock.post(
        _URL + '/v1/sessions', json={'session_id': 'sess-1'}, status_code=201
    )
    requests_mock.post(
        _URL + '/v1/sessions/sess-1/predictions', json={'predictions': []}
    )
    requests_mock.delete(_URL + '/v1/sessions/sess-1', status_code=204)

    transport = Transport(_URL)
    assert transport.create_session({'context': {}})['session_id'] == 'sess-1'
    assert transport.session_predict('sess-1', {'predict': {}}) == {
        'predictions': []
    }
    assert transport.delete_session('sess-1') is None


def test_session_id_cannot_escape_its_path_segment(requests_mock):
    r"""A server-chosen id is spliced into a URL; without escaping, one holding
    ``../`` would send the next call to a different route entirely.
    """
    requests_mock.delete(
        _URL + '/v1/sessions/..%2F..%2Fv1%2Fpredictions', status_code=204
    )

    Transport(_URL).delete_session('../../v1/predictions')

    assert requests_mock.last_request.path.lower() == (
        '/v1/sessions/..%2f..%2fv1%2fpredictions'
    )


def test_delete_session_reports_a_server_error(requests_mock):
    requests_mock.delete(
        _URL + '/v1/sessions/sess-1',
        status_code=500,
        json={'code': 'INTERNAL_ERROR', 'detail': 'boom'},
    )

    with pytest.raises(NimRequestError) as excinfo:
        Transport(_URL).delete_session('sess-1')

    assert excinfo.value.status_code == 500


def test_transport_mounts_retry_policy():
    transport = Transport(_URL, max_retries=5)
    retry = transport._session.get_adapter(_URL).max_retries
    assert retry.total == 5
    assert 429 in retry.status_forcelist
    assert 503 in retry.status_forcelist
    assert 'POST' in retry.allowed_methods


def test_retry_policy_survives_urllib3_without_retry_after_max(monkeypatch):
    r"""urllib3 < 2.3 rejects ``retry_after_max``; the policy must still build.

    CI runs a urllib3 that accepts the keyword, so the fallback in
    ``_build_retry`` is only ever reached by forcing the ``TypeError`` here.
    """

    def _reject_cap(**options):
        if 'retry_after_max' in options:
            raise TypeError(
                '__init__() got an unexpected keyword argument '
                "'retry_after_max'"
            )
        return Retry(**options)

    monkeypatch.setattr(transport_module, 'Retry', _reject_cap)
    retry = transport_module._build_retry(3, 0.5)

    assert retry.total == 3
    assert 503 in retry.status_forcelist
    assert retry.respect_retry_after_header is True


def test_client_forwards_timeout_and_max_retries():
    client = StructuredClient(_URL, timeout=12.0, max_retries=7)
    transport = client._transport
    assert transport.timeout == 12.0
    assert transport._session.get_adapter(_URL).max_retries.total == 7


# `X-API-Key` must not follow a redirect to another origin. `requests_mock`
# cannot exercise redirect resolution, so these run over real sockets.


def _make_handler(state: dict) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def do_POST(self) -> None:
            state['headers'].append(dict(self.headers))
            self.rfile.read(int(self.headers.get('Content-Length') or 0))
            location = state.get('redirect_to')
            if location and not state['redirected']:
                state['redirected'] = True
                self.send_response(307)
                self.send_header('Location', location)
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            body = json.dumps({'predictions': []}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    return _Handler


@pytest.fixture
def servers() -> Iterator[tuple[dict, dict]]:
    states: list[dict] = []
    running: list[ThreadingHTTPServer] = []
    for _ in range(2):
        state: dict = {'headers': [], 'redirected': False}
        server = ThreadingHTTPServer(('127.0.0.1', 0), _make_handler(state))
        state['url'] = f'http://localhost:{server.server_address[1]}'
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        states.append(state)
        running.append(server)
    try:
        yield states[0], states[1]
    finally:
        for server in running:
            server.shutdown()
            server.server_close()


def test_api_key_is_not_forwarded_across_a_cross_origin_redirect(servers):
    origin, target = servers
    origin['redirect_to'] = target['url'] + _PREDICTIONS_PATH

    Transport(origin['url'], api_key='secret').predict(
        {'model': 'kumo-relational'}
    )

    assert origin['headers'][0]['X-API-Key'] == 'secret'
    assert target['headers'][0].get('X-API-Key') is None


def test_api_key_is_kept_on_a_same_origin_redirect(servers):
    origin, _ = servers
    origin['redirect_to'] = origin['url'] + '/v1/predictions/'

    Transport(origin['url'], api_key='secret').predict(
        {'model': 'kumo-relational'}
    )

    assert len(origin['headers']) == 2
    assert origin['headers'][1]['X-API-Key'] == 'secret'


def test_redirects_are_still_followed(servers):
    origin, target = servers
    origin['redirect_to'] = target['url'] + _PREDICTIONS_PATH

    body = Transport(origin['url']).predict({'model': 'kumo-relational'})

    assert body == {'predictions': []}
    assert len(target['headers']) == 1


def test_nim_error_string_carries_the_http_status(requests_mock):
    requests_mock.post(
        _URL + '/v1/predictions',
        status_code=422,
        json={'detail': 'schema validation failed', 'code': 'INVALID_SCHEMA'},
    )

    with pytest.raises(NimRequestError) as excinfo:
        Transport(_URL).predict({'model': 'kumo-tabular'})
    assert str(excinfo.value) == (
        '[422 INVALID_SCHEMA] schema validation failed'
    )


def test_nim_error_string_carries_the_status_without_a_code(requests_mock):
    requests_mock.post(
        _URL + '/v1/predictions', status_code=403, text='Forbidden'
    )

    with pytest.raises(NimRequestError) as excinfo:
        Transport(_URL).predict({'model': 'kumo-tabular'})
    assert str(excinfo.value) == '[403] Forbidden'


def test_nim_error_truncates_a_huge_response_body(requests_mock):
    requests_mock.post(
        _URL + '/v1/predictions',
        status_code=502,
        text='<html>' + 'x' * 3_000_000 + '</html>',
    )

    with pytest.raises(NimRequestError) as excinfo:
        Transport(_URL).predict({'model': 'kumo-tabular'})
    assert len(str(excinfo.value)) < 1024
    assert 'truncated' in str(excinfo.value)
    assert excinfo.value.status_code == 502


@pytest.mark.parametrize(
    'timeout', [-1, 0, 'sixty', None, float('inf'), float('nan'), float('-inf')]
)
def test_invalid_timeout_is_rejected_at_construction(timeout):
    r"""``inf`` and ``nan`` too: client-non-finite-timeout-escapes-the-limit-
    guard.md. Both compare ``False`` against ``<= 0``, so they used to be
    accepted here and then surface on the first request as ``OverflowError:
    timestamp out of range for platform time_t``, which names nothing the
    caller typed. ``inf`` is the reachable one -- this constructor rejects
    ``None``, so it is the only way left to spell "no timeout".
    """
    with pytest.raises(StructuredError) as excinfo:
        Transport(_URL, timeout=timeout)
    assert excinfo.value.code == 'INVALID_CONFIGURATION'


@pytest.mark.parametrize('timeout', [1, 0.5, 60, 3600.0])
def test_usable_timeouts_are_still_accepted(timeout):
    assert Transport(_URL, timeout=timeout).timeout == timeout


@pytest.mark.parametrize('max_retries', [-1, 2.5, 'three'])
def test_invalid_max_retries_is_rejected_at_construction(max_retries):
    with pytest.raises(StructuredError) as excinfo:
        Transport(_URL, max_retries=max_retries)
    assert excinfo.value.code == 'INVALID_CONFIGURATION'


@pytest.mark.parametrize('api_key', [None, 'secret'])
def test_url_without_a_host_is_rejected_at_construction(api_key):
    with pytest.raises(StructuredError) as excinfo:
        Transport('http://', api_key=api_key)
    assert excinfo.value.code == 'INVALID_CONFIGURATION'
    assert 'missing a host' in str(excinfo.value)


def test_predict_after_close_is_rejected(requests_mock, context_df, predict_df):
    requests_mock.post(_URL + '/v1/predictions', json=_canned_response())
    client = StructuredClient(url=_URL)
    client.close()

    with pytest.raises(StructuredError) as excinfo:
        client._predict(
            NemotronTabularRequest(
                context=context_df,
                predict=predict_df,
                task='classification',
                target='target_col',
                outputs=['prediction'],
            )
        )
    assert excinfo.value.code == 'INVALID_CONFIGURATION'
    assert requests_mock.call_count == 0


# ---------------------------------------------------------------------------
# Both transports must satisfy the same internal contract.
#
# These exist because a merge once left ServingTarget without _require_open
# while StructuredClient._predict had started calling it. Nothing conflicted
# textually and the whole suite stayed green -- the break only surfaced as an
# AttributeError against a live serving endpoint. Anything _predict calls on
# self._transport belongs here, parametrised over both, so the next divergence
# fails locally instead.
# ---------------------------------------------------------------------------


def _serving_target():
    from nemotron_structured.core.serving import DatabricksServingTarget

    return DatabricksServingTarget('an-endpoint', object())


def _http_transport():
    return Transport(url=_URL)


_TRANSPORTS = [
    pytest.param(_http_transport, id='http'),
    pytest.param(_serving_target, id='serving'),
]


@pytest.mark.parametrize('build', _TRANSPORTS)
def test_transport_exposes_what_predict_calls(build):
    # Named explicitly rather than derived from Transport: the point is to pin
    # the surface _predict depends on, not to mirror whatever Transport grows.
    target = build()
    for name in ('_require_open', 'close', 'health_ready', 'predict'):
        assert callable(getattr(target, name, None)), (
            f'{type(target).__name__} is missing {name}()'
        )


@pytest.mark.parametrize('build', _TRANSPORTS)
def test_require_open_passes_while_open(build):
    build()._require_open()


@pytest.mark.parametrize('build', _TRANSPORTS)
def test_require_open_refuses_after_close(build):
    target = build()
    target.close()
    with pytest.raises(StructuredError) as excinfo:
        target._require_open()
    assert excinfo.value.code == 'INVALID_CONFIGURATION'
    assert 'closed' in str(excinfo.value)


def test_disabling_tls_verification_warns_for_a_real_host():
    with pytest.warns(UserWarning, match='verification is disabled'):
        Transport('https://nim.example.com', verify_ssl=False).close()


def test_disabling_tls_verification_is_quiet_on_loopback():
    """A local NIM behind a self-signed certificate is ordinary."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        Transport('https://localhost:8000', verify_ssl=False).close()


def test_the_warning_does_not_carry_the_credential():
    with pytest.warns(UserWarning) as caught:
        Transport(
            'https://user:secret@nim.example.com', verify_ssl=False
        ).close()
    assert 'secret' not in str(caught[0].message)

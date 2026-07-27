from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from nvidia_sdfm import SDFMClient
from nvidia_sdfm.requests import TabICLRequest
from nvidia_sdfm.core.transport import Transport
from nvidia_sdfm.errors import NimRequestError, SdfmError
from nvidia_sdfm.requests import ModelRequest

_URL = 'http://nim.example.com:8000'


def _canned_response(request_id: str = 'pred_123') -> dict:
    return {
        'id': request_id,
        'model': 'tabicl',
        'predictions': [
            {'row_index': 0, 'prediction': 'yes',
             'probabilities': {'yes': 0.8, 'no': 0.2}},
            {'row_index': 1, 'prediction': 'no',
             'probabilities': {'yes': 0.3, 'no': 0.7}},
        ],
        'metadata': {'task_kind': 'classification'},
    }


def test_predict_end_to_end_round_trip(requests_mock, context_df, predict_df):
    requests_mock.post(_URL + '/v1/predictions', json=_canned_response())

    with SDFMClient(url=_URL) as client:
        frame = client._predict(TabICLRequest(
            context=context_df,
            predict=predict_df,
            task='classification',
            target='target_col',
            outputs=['prediction', 'probabilities'],
        ))

    assert list(frame['row_index']) == [0, 1]
    assert list(frame['prediction']) == ['yes', 'no']

    sent_payload = requests_mock.last_request.json()
    assert sent_payload['model'] == 'tabicl'
    assert sent_payload['task']['target']['column_name'] == 'target_col'


def test_predict_unknown_model_raises():
    @dataclass
    class _UnknownRequest(ModelRequest):
        model: ClassVar[str] = 'not-a-real-model'

    with pytest.raises(SdfmError):
        SDFMClient(url=_URL)._predict(_UnknownRequest())


def test_predict_wrong_request_type_raises():
    @dataclass
    class _MislabelledRequest(ModelRequest):
        model: ClassVar[str] = 'tabicl'

    with pytest.raises(SdfmError) as excinfo:
        SDFMClient(url=_URL)._predict(_MislabelledRequest())
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
        SDFMClient(url=_URL)._predict(TabICLRequest(
            context=context_df,
            predict=predict_df,
            task='classification',
            target='target_col',
            outputs=['prediction'],
        ))
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
    client.predict({'model': 'tabicl'})
    assert requests_mock.last_request.headers['X-API-Key'] == 'secret'


def test_api_key_over_plaintext_http_is_rejected():
    with pytest.raises(SdfmError) as excinfo:
        Transport('http://nim.example.com:8000', api_key='secret')
    assert excinfo.value.code == 'INVALID_CONFIGURATION'


def test_api_key_over_localhost_http_is_allowed():
    assert Transport('http://localhost:8000', api_key='secret').url


def test_api_key_over_https_is_allowed():
    assert Transport('https://nim.example.com', api_key='secret').url


def test_no_api_key_over_http_is_allowed():
    assert Transport('http://nim.example.com:8000').url


def test_non_http_scheme_is_rejected():
    with pytest.raises(SdfmError) as excinfo:
        Transport('ftp://nim.example.com')
    assert excinfo.value.code == 'INVALID_CONFIGURATION'


def test_invalid_json_success_response_raises_transport_error(requests_mock):
    requests_mock.post(_URL + '/v1/predictions', text='not json')
    client = Transport(_URL)
    with pytest.raises(SdfmError) as excinfo:
        client.predict({'model': 'tabicl'})
    assert excinfo.value.code == 'TRANSPORT_ERROR'


def test_non_object_json_success_response_raises_transport_error(requests_mock):
    requests_mock.post(_URL + '/v1/predictions', json=['a', 'b'])
    client = Transport(_URL)
    with pytest.raises(SdfmError) as excinfo:
        client.predict({'model': 'tabicl'})
    assert excinfo.value.code == 'TRANSPORT_ERROR'


def test_non_object_json_error_body_is_handled(requests_mock):
    requests_mock.post(_URL + '/v1/predictions', status_code=500, json=['boom'])
    client = Transport(_URL)
    with pytest.raises(NimRequestError) as excinfo:
        client.predict({'model': 'tabicl'})
    assert excinfo.value.status_code == 500


def test_session_endpoints_not_implemented():
    client = Transport(_URL)
    with pytest.raises(NotImplementedError):
        client.create_session({})
    with pytest.raises(NotImplementedError):
        client.session_predict('sess-1', {})
    with pytest.raises(NotImplementedError):
        client.delete_session('sess-1')


def test_transport_mounts_retry_policy():
    transport = Transport(_URL, max_retries=5)
    retry = transport._session.get_adapter(_URL).max_retries
    assert retry.total == 5
    assert 429 in retry.status_forcelist
    assert 503 in retry.status_forcelist
    assert 'POST' in retry.allowed_methods


def test_client_forwards_timeout_and_max_retries():
    client = SDFMClient(_URL, timeout=12.0, max_retries=7)
    transport = client._transport
    assert transport.timeout == 12.0
    assert transport._session.get_adapter(_URL).max_retries.total == 7

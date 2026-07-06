from __future__ import annotations

import pytest

import nvidia_sdfm
from nvidia_sdfm.core.transport import TFMClient
from nvidia_sdfm.errors import NimRequestError, NotInitializedError, SdfmError

_URL = 'http://nim.example.com:8000'


@pytest.fixture(autouse=True)
def _reset_client():
    nvidia_sdfm._client = None
    yield
    nvidia_sdfm._client = None


def _canned_response(request_id: str = 'pred_123') -> dict:
    return {
        'id': request_id,
        'model': 'tabicl',
        'predictions': [
            {'row_index': 0, 'prediction': 'yes', 'probabilities': {'yes': 0.8, 'no': 0.2}},
            {'row_index': 1, 'prediction': 'no', 'probabilities': {'yes': 0.3, 'no': 0.7}},
        ],
        'metadata': {'task_kind': 'classification'},
    }


def test_predict_end_to_end_round_trip(requests_mock, context_df, predict_df):
    requests_mock.post(_URL + '/v1/predictions', json=_canned_response())

    nvidia_sdfm.init(url=_URL)
    frame = nvidia_sdfm.predict(
        model='tabicl',
        context=context_df,
        predict=predict_df,
        task='classification',
        target='target_col',
        outputs=['prediction', 'probabilities'],
    )

    assert list(frame['row_index']) == [0, 1]
    assert list(frame['prediction']) == ['yes', 'no']

    sent_payload = requests_mock.last_request.json()
    assert sent_payload['model'] == 'tabicl'
    assert sent_payload['task']['target']['column_name'] == 'target_col'


def test_predict_without_init_raises():
    with pytest.raises(NotInitializedError):
        nvidia_sdfm.predict(model='tabicl')


def test_predict_unknown_model_raises():
    nvidia_sdfm.init(url=_URL)
    with pytest.raises(SdfmError):
        nvidia_sdfm.predict(model='not-a-real-model')


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

    nvidia_sdfm.init(url=_URL)
    with pytest.raises(NimRequestError) as excinfo:
        nvidia_sdfm.predict(
            model='tabicl',
            context=context_df,
            predict=predict_df,
            task='classification',
            target='target_col',
            outputs=['prediction'],
        )
    assert excinfo.value.status_code == 422
    assert excinfo.value.code == 'INVALID_SCHEMA'


def test_health_ready_true_on_200(requests_mock):
    requests_mock.get(_URL + '/v1/health/ready', status_code=200, json={})
    client = TFMClient(_URL)
    assert client.health_ready() is True


def test_health_ready_false_on_503(requests_mock):
    requests_mock.get(_URL + '/v1/health/ready', status_code=503, json={})
    client = TFMClient(_URL)
    assert client.health_ready() is False


def test_client_sends_api_key_header(requests_mock):
    requests_mock.post(_URL + '/v1/predictions', json=_canned_response())
    client = TFMClient(_URL, api_key='secret')
    client.predict({'model': 'tabicl'})
    assert requests_mock.last_request.headers['X-API-Key'] == 'secret'


def test_invalid_json_success_response_raises_transport_error(requests_mock):
    requests_mock.post(_URL + '/v1/predictions', text='not json')
    client = TFMClient(_URL)
    with pytest.raises(SdfmError) as excinfo:
        client.predict({'model': 'tabicl'})
    assert excinfo.value.code == 'TRANSPORT_ERROR'


def test_non_object_json_success_response_raises_transport_error(requests_mock):
    requests_mock.post(_URL + '/v1/predictions', json=['a', 'b'])
    client = TFMClient(_URL)
    with pytest.raises(SdfmError) as excinfo:
        client.predict({'model': 'tabicl'})
    assert excinfo.value.code == 'TRANSPORT_ERROR'


def test_non_object_json_error_body_is_handled(requests_mock):
    requests_mock.post(_URL + '/v1/predictions', status_code=500, json=['boom'])
    client = TFMClient(_URL)
    with pytest.raises(NimRequestError) as excinfo:
        client.predict({'model': 'tabicl'})
    assert excinfo.value.status_code == 500


def test_session_endpoints_not_implemented():
    client = TFMClient(_URL)
    with pytest.raises(NotImplementedError):
        client.create_session({})
    with pytest.raises(NotImplementedError):
        client.session_predict('sess-1', {})
    with pytest.raises(NotImplementedError):
        client.delete_session('sess-1')

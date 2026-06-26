from __future__ import annotations

from typing import Any

import pytest
import requests

from kumoai.client import KumoClient
from kumoai.client.generated.tfm_api import TFMOperations
from kumoai.client.rfm import RFMAPI
from kumoai.exceptions import HTTPException

from rfm_nim_payloads import (
    NIM_V0_PREDICTION_PATH,
    SDK_V1_PREDICTION_PATH,
    nim_v0_session_create_payload,
    nim_v0_session_predict_minimal_payload,
    nim_v0_smoke_payload,
    sdk_v1_smoke_payload,
)

MOCK_URL = 'http://kumo.ai'


class JsonRequestCapture:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None
        self.headers: requests.structures.CaseInsensitiveDict[str] | None = None

    def __call__(self, request: Any) -> bool:
        self.payload = request.json()
        self.headers = request.headers
        return True


def test_rfm_api_predict_posts_sdk_v1_payload_and_parses_response(
    mock_api: Any,
) -> None:
    capture = JsonRequestCapture()
    mock_api.post(
        f'{MOCK_URL}{SDK_V1_PREDICTION_PATH}',
        additional_matcher=capture,
        json={
            'id': 'pred-contract-test',
            'model': 'kumo-rfm',
            'predictions': [{
                'id': '601',
                'prediction': True,
                'probabilities': {
                    'True': 0.65,
                    'False': 0.35,
                },
            }],
            'metadata': {
                'version': 'v1',
                'adapter': 'mock',
            },
        },
    )

    api = RFMAPI(KumoClient(MOCK_URL, api_key='DISABLED'))
    result = api.predict(sdk_v1_smoke_payload())

    assert capture.payload == sdk_v1_smoke_payload()
    assert capture.headers is not None
    assert capture.headers['Content-Type'] == 'application/json'
    assert result.prediction == {
        'columns': ['ENTITY', 'prediction', 'True_PROB', 'False_PROB'],
        'data': [[601, True, 0.65, 0.35]],
    }


def test_sdk_prediction_endpoint_is_v1_but_current_nim_smoke_is_v0() -> None:
    client = KumoClient(MOCK_URL, api_key='DISABLED')
    endpoint = TFMOperations.create_prediction.endpoint

    assert endpoint.get_path() == '/predictions'
    assert client._format_endpoint_url(endpoint.get_path()) == (
        f'{MOCK_URL}{SDK_V1_PREDICTION_PATH}')
    assert NIM_V0_PREDICTION_PATH == '/v0/predictions'
    assert SDK_V1_PREDICTION_PATH != NIM_V0_PREDICTION_PATH


def test_sdk_predict_surfaces_current_nim_v1_prediction_404(
    mock_api: Any,
) -> None:
    mock_api.post(
        f'{MOCK_URL}{SDK_V1_PREDICTION_PATH}',
        status_code=404,
        json={'detail': 'Not Found'},
    )

    api = RFMAPI(KumoClient(MOCK_URL, api_key='DISABLED'))
    with pytest.raises(HTTPException) as exc_info:
        api.predict(sdk_v1_smoke_payload())

    assert exc_info.value.status_code == 404
    assert mock_api.request_history[0].url == (
        f'{MOCK_URL}{SDK_V1_PREDICTION_PATH}')


def test_container_smoke_payload_tracks_current_v0_wire_shape() -> None:
    payload = nim_v0_smoke_payload()

    assert set(payload) == {
        'model',
        'task',
        'schema',
        'context',
        'predict',
        'output',
        'inference',
    }
    assert 'version' not in payload
    assert 'metadata' not in payload
    assert payload['model'] == 'kumo-rfm'
    assert payload['task']['kind'] == 'classification'
    assert payload['output']['fields'] == ['prediction', 'probabilities']
    assert payload['inference'] == {'run_mode': 'best'}


def test_session_payload_helpers_match_current_v0_session_contract() -> None:
    create = nim_v0_session_create_payload()
    predict = nim_v0_session_predict_minimal_payload()

    assert set(create) == {'model', 'task', 'schema', 'context'}
    assert 'predict' not in create
    assert 'output' not in create
    assert 'inference' not in create

    assert set(predict) == {'predict', 'output'}
    assert predict['output']['fields'] == ['prediction', 'probabilities']

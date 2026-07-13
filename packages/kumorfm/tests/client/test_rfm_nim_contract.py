from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import requests

from kumorfm.client import KumoClient
from kumorfm.client.endpoints import HTTPMethod
from kumorfm.client.generated.tfm_api import (
    TFMOperations,
    TFM_ENDPOINTS_BY_OPERATION_ID,
)
from kumorfm.client.rfm import RFMAPI

from rfm_nim_payloads import (
    NIM_HEALTH_READY_PATH,
    NIM_V1_PREDICTION_PATH,
    NIM_V1_SESSIONS_PATH,
    nim_v1_session_create_payload,
    nim_v1_session_predict_minimal_payload,
    nim_v1_smoke_payload,
)

MOCK_URL = 'http://kumo.ai'
SDK_V1_MODELS_PATH = '/v1/models'


class JsonRequestCapture:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None
        self.headers: requests.structures.CaseInsensitiveDict[str] | None = None

    def __call__(self, request: Any) -> bool:
        self.payload = request.json()
        self.headers = request.headers
        return True


def test_rfm_api_predict_posts_current_v1_payload_and_parses_response(
    mock_api: Any,
) -> None:
    capture = JsonRequestCapture()
    mock_api.post(
        f'{MOCK_URL}{NIM_V1_PREDICTION_PATH}',
        additional_matcher=capture,
        json={
            'id': 'pred-contract-test',
            'model': 'kumo-rfm',
            'predictions': [{
                'id': '601',
                'row_index': 0,
                'prediction': True,
                'probabilities': {
                    'True': 0.65,
                    'False': 0.35,
                },
            }],
            'metadata': {
                'adapter': 'mock',
            },
        },
    )

    api = RFMAPI(KumoClient(MOCK_URL, api_key='DISABLED'))
    payload = nim_v1_smoke_payload()
    result = api.predict(
        payload,
        entity_ids=[601],
        instance_ids=[601],
    )

    assert capture.payload == payload
    assert capture.headers is not None
    assert capture.headers['Content-Type'] == 'application/json'
    assert result.prediction == {
        'columns': ['ENTITY', 'prediction', 'True_PROB', 'False_PROB'],
        'data': [[601, True, 0.65, 0.35]],
    }


def test_sdk_prediction_endpoint_matches_current_nim_v1_route() -> None:
    client = KumoClient(MOCK_URL, api_key='DISABLED')
    endpoint = TFMOperations.run_prediction.endpoint

    assert endpoint.get_path() == NIM_V1_PREDICTION_PATH
    assert client._format_endpoint_url(endpoint.get_path()) == (
        f'{MOCK_URL}{NIM_V1_PREDICTION_PATH}')


def test_sdk_health_endpoint_matches_current_nim_route() -> None:
    client = KumoClient(MOCK_URL, api_key='DISABLED')

    assert NIM_HEALTH_READY_PATH == '/v1/health/ready'
    assert client._format_endpoint_url(NIM_HEALTH_READY_PATH) == (
        f'{MOCK_URL}{NIM_HEALTH_READY_PATH}')


@pytest.mark.parametrize(
    ('operation_id', 'method', 'current_nim_path'),
    [
        ('createSession', HTTPMethod.POST, NIM_V1_SESSIONS_PATH),
        (
            'runSessionPrediction',
            HTTPMethod.POST,
            f'{NIM_V1_SESSIONS_PATH}/{{session_id}}/predictions',
        ),
        (
            'deleteSession',
            HTTPMethod.DELETE,
            f'{NIM_V1_SESSIONS_PATH}/{{session_id}}',
        ),
    ],
)
def test_sdk_generated_session_endpoints_match_current_nim_routes(
    operation_id: str,
    method: HTTPMethod,
    current_nim_path: str,
) -> None:
    client = KumoClient(MOCK_URL, api_key='DISABLED')
    endpoint = TFM_ENDPOINTS_BY_OPERATION_ID[operation_id]

    assert endpoint.method == method
    assert client._format_endpoint_url(endpoint.get_path()) == (
        f'{MOCK_URL}{current_nim_path}')


def test_sdk_rest_authenticate_accepts_current_nim_surface(
    mock_api: Any,
) -> None:
    mock_api.get(f'{MOCK_URL}{NIM_HEALTH_READY_PATH}', json={'status': 'ready'})
    mock_api.get(
        f'{MOCK_URL}{SDK_V1_MODELS_PATH}',
        json={'data': [{
            'id': 'kumo-rfm',
        }]},
    )

    client = KumoClient(MOCK_URL)
    client.authenticate()


def test_rfm_api_predict_does_not_mutate_request_payload(
    mock_api: Any,
) -> None:
    mock_api.post(
        f'{MOCK_URL}{NIM_V1_PREDICTION_PATH}',
        json={
            'id': 'pred-contract-test',
            'model': 'kumo-rfm',
            'predictions': [{
                'id': '601',
                'row_index': 0,
            }],
        },
    )
    payload = nim_v1_smoke_payload()
    original = deepcopy(payload)

    api = RFMAPI(KumoClient(MOCK_URL, api_key='DISABLED'))
    api.predict(
        payload,
        entity_ids=[601],
        instance_ids=[601],
    )

    assert payload == original


def test_rfm_api_predict_maps_varied_prediction_item_shapes(
    mock_api: Any,
) -> None:
    mock_api.post(
        f'{MOCK_URL}{NIM_V1_PREDICTION_PATH}',
        json={
            'id': 'pred-varied-test',
            'model': 'kumo-rfm',
            'predictions': [
                {
                    'id': '601',
                    'row_index': 0,
                    'prediction': 'gold',
                    'scores': ['0.5', 1],
                    'rankings': [{
                        'entity_id': 'item-1',
                        'score': 0.9,
                    }],
                    'embeddings': ['0.1', 0.2],
                    'quantiles': {
                        'p50': '12.5',
                    },
                    'explanation': {
                        'reason': 'fixture',
                    },
                },
                {
                    'id': '602',
                    'row_index': 1,
                    'prediction': 3.14,
                },
                {
                    'id': '603',
                    'row_index': 2,
                    'probabilities': {
                        'A': '0.25',
                        'B': 0.75,
                    },
                },
            ],
            'metadata': {
                'adapter': 'mock',
            },
        },
    )

    api = RFMAPI(KumoClient(MOCK_URL, api_key='DISABLED'))
    payload = nim_v1_smoke_payload()
    payload['predict']['instance_table']['rows'] = [
        [601, '2025-02-01T00:00:00Z'],
        [602, '2025-02-02T00:00:00Z'],
        [603, '2025-02-03T00:00:00Z'],
    ]
    result = api.predict(
        payload,
        entity_ids=['account-a', 'account-a', 'account-b'],
        instance_ids=[601, 602, 603],
    )

    assert result.prediction == {
        'columns': [
            'ENTITY',
            'prediction',
            'scores',
            'rankings',
            'embeddings',
            'q_p50',
            'explanation',
            'A_PROB',
            'B_PROB',
        ],
        'data': [
            [
                'account-a',
                'gold',
                [0.5, 1.0],
                [{
                    'entity_id': 'item-1',
                    'score': 0.9,
                }],
                [0.1, 0.2],
                12.5,
                {
                    'reason': 'fixture',
                },
                None,
                None,
            ],
            [
                'account-a', 3.14, None, None, None, None, None, None,
                None,
            ],
            [
                'account-b', None, None, None, None, None, None, 0.25,
                0.75,
            ],
        ],
    }


def test_prediction_response_rejects_bad_probability_shape(
    mock_api: Any,
) -> None:
    mock_api.post(
        f'{MOCK_URL}{NIM_V1_PREDICTION_PATH}',
        json={
            'id': 'pred-bad-probabilities',
            'model': 'kumo-rfm',
            'predictions': [{
                'id': '1',
                'probabilities': ['not', 'a', 'mapping'],
            }],
        },
    )

    api = RFMAPI(KumoClient(MOCK_URL, api_key='DISABLED'))
    with pytest.raises(TypeError, match='Expected mapping value'):
        api.predict(
            nim_v1_smoke_payload(),
            entity_ids=[601],
            instance_ids=[601],
        )


def test_payload_factories_return_isolated_deep_copies() -> None:
    first = nim_v1_smoke_payload()
    second = nim_v1_smoke_payload()

    first['context']['instance_table']['rows'][0][1] = False
    first['schema']['related_tables']['accounts']['columns']['segment'][
        'dtype'] = 'mutated'

    assert second['context']['instance_table']['rows'][0][1] is True
    assert second['schema']['related_tables']['accounts']['columns'][
        'segment']['dtype'] == 'string'


def test_container_smoke_payload_tracks_current_v1_wire_shape() -> None:
    payload = nim_v1_smoke_payload()

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


def test_session_payload_helpers_match_current_v1_session_contract() -> None:
    create = nim_v1_session_create_payload()
    predict = nim_v1_session_predict_minimal_payload()

    assert set(create) == {'model', 'task', 'schema', 'context'}
    assert 'predict' not in create
    assert 'output' not in create
    assert 'inference' not in create

    assert set(predict) == {'predict', 'output'}
    assert predict['output']['fields'] == ['prediction', 'probabilities']

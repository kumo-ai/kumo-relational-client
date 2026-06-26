from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import requests

from kumoai.client import KumoClient
from kumoai.client.endpoints import HTTPMethod, RFMEndpoints
from kumoai.client.generated.tfm_api import TFMOperations
from kumoai.client.generated.tfm_api import TFM_ENDPOINTS_BY_OPERATION_ID
from kumoai.client.rfm import RFMAPI
from kumoai.exceptions import HTTPException

from rfm_nim_payloads import (
    NIM_HEALTH_READY_PATH,
    NIM_V0_PREDICTION_PATH,
    SDK_V1_PREDICTION_PATH,
    SDK_V1_CONNECTORS_PATH,
    SDK_V1_RFM_PARSE_QUERY_PATH,
    SDK_V1_RFM_VALIDATE_QUERY_PATH,
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        'Known SDK/container compatibility bug: RFMAPI.predict currently posts '
        'to /v1/predictions, while this Kumo RFM NIM serves /v0/predictions.'),
)
def test_sdk_prediction_endpoint_should_match_current_nim_route() -> None:
    client = KumoClient(MOCK_URL, api_key='DISABLED')
    endpoint = TFMOperations.create_prediction.endpoint

    assert client._format_endpoint_url(endpoint.get_path()) == (
        f'{MOCK_URL}{NIM_V0_PREDICTION_PATH}')


@pytest.mark.xfail(
    strict=True,
    reason=(
        'Known SDK/container compatibility bug: SDK prediction payloads include '
        'the v1 body-level version field, while the current v0 NIM contract '
        'uses path versioning and rejects unknown top-level fields.'),
)
def test_sdk_prediction_payload_should_match_current_nim_v0_envelope() -> None:
    sdk_payload = sdk_v1_smoke_payload()
    nim_payload = nim_v0_smoke_payload()

    assert 'version' not in sdk_payload
    assert set(nim_payload) <= set(sdk_payload)


@pytest.mark.parametrize(
    ('operation', 'current_nim_path'),
    [
        (TFMOperations.get_health_live, '/health/live'),
        (TFMOperations.get_health_ready, '/health/ready'),
    ],
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        'Known SDK/container compatibility bug: generated TFM health endpoints '
        'are version-prefixed by KumoClient, while this NIM exposes unversioned '
        'health routes.'),
)
def test_sdk_generated_health_endpoints_should_match_current_nim_routes(
    operation: Any,
    current_nim_path: str,
) -> None:
    client = KumoClient(MOCK_URL, api_key='DISABLED')

    assert client._format_endpoint_url(operation.endpoint.get_path()) == (
        f'{MOCK_URL}{current_nim_path}')


@pytest.mark.parametrize(
    ('operation_id', 'method', 'current_nim_path'),
    [
        ('createSession', HTTPMethod.POST, '/v0/sessions'),
        (
            'runSessionPrediction',
            HTTPMethod.POST,
            '/v0/sessions/{session_id}/predictions',
        ),
        ('deleteSession', HTTPMethod.DELETE, '/v0/sessions/{session_id}'),
    ],
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        'Known SDK/container compatibility bug: generated TFM metadata does '
        'not include the current v0 session routes exposed by this NIM.'),
)
def test_sdk_generated_session_endpoints_should_match_current_nim_routes(
    operation_id: str,
    method: HTTPMethod,
    current_nim_path: str,
) -> None:
    client = KumoClient(MOCK_URL, api_key='DISABLED')
    endpoint = TFM_ENDPOINTS_BY_OPERATION_ID[operation_id]

    assert endpoint.method == method
    assert client._format_endpoint_url(endpoint.get_path()) == (
        f'{MOCK_URL}{current_nim_path}')


@pytest.mark.parametrize(
    ('endpoint', 'legacy_path'),
    [
        (RFMEndpoints.parse_query, SDK_V1_RFM_PARSE_QUERY_PATH),
        (RFMEndpoints.validate_query, SDK_V1_RFM_VALIDATE_QUERY_PATH),
    ],
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        'Known SDK/container compatibility bug: public RFM query helpers still '
        'post to legacy /v1/rfm/* routes, but the current NIM exposes only the '
        'Universal TFM /v0 API.'),
)
def test_sdk_rfm_query_helpers_should_not_target_legacy_nim_absent_routes(
    endpoint: Any,
    legacy_path: str,
) -> None:
    client = KumoClient(MOCK_URL, api_key='DISABLED')

    assert client._format_endpoint_url(endpoint.get_path()) != (
        f'{MOCK_URL}{legacy_path}')


@pytest.mark.xfail(
    strict=True,
    reason=(
        'Known SDK/container compatibility bug: KumoClient.authenticate probes '
        '/v1/connectors, which is absent from the current NIM; NIM-compatible '
        'init should use the available readiness/model metadata surface or '
        'skip legacy auth.'),
)
def test_sdk_rest_authenticate_should_work_against_current_nim_surface(
    mock_api: Any,
) -> None:
    mock_api.get(f'{MOCK_URL}{NIM_HEALTH_READY_PATH}', json={'status': 'ready'})
    mock_api.get(
        f'{MOCK_URL}{SDK_V1_CONNECTORS_PATH}',
        status_code=404,
        json={
            'type': 'about:blank',
            'title': 'Not Found',
            'status': 404,
            'detail': 'Not Found',
        },
    )

    client = KumoClient(MOCK_URL, api_key='DISABLED')
    client.authenticate()


def test_sdk_predict_surfaces_current_nim_v1_prediction_404(
    mock_api: Any,
) -> None:
    error_body = {
        'type': 'about:blank',
        'title': 'Not Found',
        'status': 404,
        'detail': 'Not Found',
    }
    mock_api.post(
        f'{MOCK_URL}{SDK_V1_PREDICTION_PATH}',
        status_code=404,
        json=error_body,
    )

    api = RFMAPI(KumoClient(MOCK_URL, api_key='DISABLED'))
    with pytest.raises(HTTPException) as exc_info:
        api.predict(sdk_v1_smoke_payload())

    assert exc_info.value.status_code == 404
    assert 'Not Found' in exc_info.value.detail
    assert 'Not Found' in str(exc_info.value)
    assert mock_api.request_history[0].url == (
        f'{MOCK_URL}{SDK_V1_PREDICTION_PATH}')


def test_rfm_api_predict_does_not_mutate_request_payload(
    mock_api: Any,
) -> None:
    mock_api.post(
        f'{MOCK_URL}{SDK_V1_PREDICTION_PATH}',
        json={
            'id': 'pred-contract-test',
            'model': 'kumo-rfm',
            'predictions': [],
        },
    )
    payload = sdk_v1_smoke_payload()
    original = deepcopy(payload)

    api = RFMAPI(KumoClient(MOCK_URL, api_key='DISABLED'))
    api.predict(payload)

    assert payload == original


def test_rfm_api_predict_maps_varied_prediction_item_shapes(
    mock_api: Any,
) -> None:
    mock_api.post(
        f'{MOCK_URL}{SDK_V1_PREDICTION_PATH}',
        json={
            'id': 'pred-varied-test',
            'model': 'kumo-rfm',
            'predictions': [
                {
                    'id': 'abc-123',
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
                    'row_index': '7',
                    'prediction': 3.14,
                },
                {
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
    result = api.predict(sdk_v1_smoke_payload())

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
                'abc-123',
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
            [7, 3.14, None, None, None, None, None, None, None],
            [2, None, None, None, None, None, None, 0.25, 0.75],
        ],
    }


def test_prediction_response_rejects_bad_probability_shape(
    mock_api: Any,
) -> None:
    mock_api.post(
        f'{MOCK_URL}{SDK_V1_PREDICTION_PATH}',
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
        api.predict(sdk_v1_smoke_payload())


def test_payload_factories_return_isolated_deep_copies() -> None:
    first = nim_v0_smoke_payload()
    second = nim_v0_smoke_payload()

    first['context']['instance_table']['rows'][0][1] = False
    first['schema']['related_tables']['accounts']['columns']['segment'][
        'dtype'] = 'mutated'

    assert second['context']['instance_table']['rows'][0][1] is True
    assert second['schema']['related_tables']['accounts']['columns'][
        'segment']['dtype'] == 'string'


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

from typing import Any

import requests
from kumoapi.pquery import ValidatedPredictiveQuery

from kumoai.client import KumoClient
from kumoai.client.rfm import RFMAPI
from kumoai.rfm import Graph, KumoRFM
from kumoai.rfm.rfm import Explanation

from test.conftest import MOCK_URL


class JsonPayloadReceptor:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None
        self.headers: requests.structures.CaseInsensitiveDict[str] | None = None

    def __call__(self, request: Any) -> bool:
        self.payload = request.json()
        self.headers = request.headers
        return True


def test_predict_posts_universal_json_payload(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
    mock_api: Any,
) -> None:
    receptor = JsonPayloadReceptor()
    mock_api.post(
        f'{MOCK_URL}/v1/predictions',
        additional_matcher=receptor,
        json={
            'id': 'pred-test',
            'model': 'kumo-rfm',
            'predictions': [{
                'id': '0',
                'prediction': 0.5,
                'embeddings': [0.1, 0.2],
            }],
            'metadata': {
                'version': 'v1',
                'task_kind': 'regression',
            },
        },
    )

    model = KumoRFM(user_store_graph, verbose=False)
    model._client = RFMAPI(KumoClient(MOCK_URL, api_key='DISABLED'))  # type: ignore

    result = model.predict(
        ltv,
        indices=[0],
        inference_config={'output_type': 'quantiles'},
        return_embeddings=True,
        verbose=False,
    )

    assert result.to_dict('records') == [{
        'ENTITY': 0,
        'prediction': 0.5,
        'embeddings': [0.1, 0.2],
    }]
    assert receptor.headers is not None
    assert receptor.headers['Content-Type'] == 'application/json'

    payload = receptor.payload
    assert payload is not None
    assert payload['version'] == 'v1'
    assert payload['model'] == 'kumo-rfm'
    assert payload['task']['kind'] == 'regression'
    assert payload['schema']['relationships']
    assert payload['context']['instance_table']['format'] == 'arrays'
    assert payload['predict']['instance_table']['format'] == 'arrays'
    assert 'embeddings' in payload['output']['fields']
    assert payload['inference']['run_mode'] == 'fast'
    assert payload['inference']['inference_config']['kind'] == 'regression'
    assert payload['inference']['inference_config']['output_type'] == (
        'quantiles')
    assert 'operation' not in payload['metadata']

    payload_text = str(payload)
    assert 'application/x-protobuf' not in payload_text
    assert "'batch'" not in payload_text
    assert "'row'" not in payload_text
    assert "'col'" not in payload_text


def test_explain_requests_explanation_output_field(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
    mock_api: Any,
) -> None:
    receptor = JsonPayloadReceptor()
    mock_api.post(
        f'{MOCK_URL}/v1/predictions',
        additional_matcher=receptor,
        json={
            'id': 'pred-test',
            'model': 'kumo-rfm',
            'predictions': [{
                'id': '0',
                'prediction': 0.5,
                'explanation': {
                    'format': 'natural_language_summary',
                    'summary': 'Order frequency dropped.',
                    'warning': 'Cross-region fallback used.',
                },
            }],
            'metadata': {
                'version': 'v1',
                'task_kind': 'regression',
            },
        },
    )

    model = KumoRFM(user_store_graph, verbose=False)
    model._client = RFMAPI(KumoClient(MOCK_URL, api_key='DISABLED'))  # type: ignore

    result = model.predict(ltv, indices=[0], explain=True, verbose=False)

    assert isinstance(result, Explanation)
    assert result.prediction.to_dict('records') == [{
        'ENTITY': 0,
        'prediction': 0.5,
    }]
    assert result.summary == 'Order frequency dropped.'
    assert result.details == {
        'format': 'natural_language_summary',
        'summary': 'Order frequency dropped.',
        'warning': 'Cross-region fallback used.',
    }
    assert result.warning == 'Cross-region fallback used.'

    assert receptor.headers is not None
    assert receptor.headers['Content-Type'] == 'application/json'
    payload = receptor.payload
    assert payload is not None
    assert 'explanation' in payload['output']['fields']
    assert 'operation' not in payload['metadata']

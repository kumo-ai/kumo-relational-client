from typing import Any

import requests
from kumoapi.pquery import ValidatedPredictiveQuery

from kumoai.client import KumoClient
from kumoai.client.rfm import RFMAPI
from kumoai.rfm import Graph, KumoRFM

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
            'prediction': {
                'columns': ['ENTITY', 'SCORE'],
                'data': [[0, 0.5]],
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

    assert result.to_dict('records') == [{'ENTITY': 0, 'SCORE': 0.5}]
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
    assert 'embedding' in payload['output']['fields']
    assert payload['inference']['run_mode'] == 'fast'
    assert payload['inference']['inference_config']['kind'] == 'regression'
    assert payload['inference']['inference_config']['output_type'] == (
        'quantiles')
    assert payload['metadata']['operation'] == 'predict'

    payload_text = str(payload)
    assert 'application/x-protobuf' not in payload_text
    assert "'batch'" not in payload_text
    assert "'row'" not in payload_text
    assert "'col'" not in payload_text

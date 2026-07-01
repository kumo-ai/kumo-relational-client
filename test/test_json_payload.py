import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import requests
from kumoapi.pquery import ValidatedPredictiveQuery
from kumoapi.rfm.context import Table
from kumoapi.typing import Stype

from kumoai.client import KumoClient
from kumoai.client.rfm import RFMAPI
from kumoai.rfm import Graph, KumoRFM
from kumoai.rfm.payload import (
    ANCHOR_TIME_PREFIX,
    ENTITY_REFERENCE_PREFIX,
    INSTANCE_ID,
    SYNTHETIC_NODE_ID,
    _occurrence_dataframe,
)
from kumoai.rfm.rfm import Explanation

from conftest import MOCK_URL

CANONICAL_SPEC = Path('../structured-data-api/nim-sd.openapi.yaml')


def test_keyless_table_synthetic_key_is_opaque_on_name_collision() -> None:
    source = pd.DataFrame({
        SYNTHETIC_NODE_ID: [10, 11],
        f'{SYNTHETIC_NODE_ID}_1': [20, 21],
        'VALUE': [1.5, 2.5],
    })
    table = Table(
        df=source,
        row=None,
        batch=np.array([0, 1]),
        num_sampled_nodes=[2],
        stype_dict={'VALUE': Stype.numerical},
        primary_key=None,
    )

    materialized, key_column = _occurrence_dataframe(table)

    assert key_column == f'{SYNTHETIC_NODE_ID}_2'
    assert materialized.columns.tolist() == [
        INSTANCE_ID,
        key_column,
        SYNTHETIC_NODE_ID,
        f'{SYNTHETIC_NODE_ID}_1',
        'VALUE',
    ]
    assert materialized[key_column].tolist() == [0, 1]
    assert materialized[SYNTHETIC_NODE_ID].tolist() == [10, 11]
    assert materialized[f'{SYNTHETIC_NODE_ID}_1'].tolist() == [20, 21]
    assert source.columns.tolist() == [
        SYNTHETIC_NODE_ID,
        f'{SYNTHETIC_NODE_ID}_1',
        'VALUE',
    ]


class JsonPayloadReceptor:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None
        self.headers: requests.structures.CaseInsensitiveDict[str] | None = None

    def __call__(self, request: Any) -> bool:
        self.payload = request.json()
        self.headers = request.headers
        return True


def _correlated_response(item_fields: dict[str, Any]) -> Any:
    def response(request: Any, _context: Any) -> dict[str, Any]:
        payload = request.json()
        table = payload['predict']['instance_table']
        instance_index = table['columns'].index(INSTANCE_ID)
        predictions = [
            {
                'id': str(row[instance_index]),
                'row_index': row_index,
                **item_fields,
            }
            for row_index, row in enumerate(table['rows'])
        ]
        return {
            'id': 'pred-test',
            'model': 'kumo-rfm',
            'predictions': predictions,
            'metadata': {
                'task_kind': 'regression',
            },
        }

    return response


def test_predict_posts_universal_json_payload(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
    mock_api: Any,
) -> None:
    receptor = JsonPayloadReceptor()
    mock_api.post(
        f'{MOCK_URL}/v1/predictions',
        additional_matcher=receptor,
        json=_correlated_response({
            'prediction': 0.5,
            'embeddings': [0.1, 0.2],
        }),
    )

    model = KumoRFM(user_store_graph, verbose=False)
    model._client = RFMAPI(KumoClient(MOCK_URL, api_key='DISABLED'))  # type: ignore

    result = model.predict(
        ltv,
        indices=[3],
        inference_config={'output_type': 'quantiles'},
        return_embeddings=True,
        verbose=False,
    )

    assert result.to_dict('records') == [{
        'ENTITY': 3,
        'prediction': 0.5,
        'embeddings': [0.1, 0.2],
    }]
    assert receptor.headers is not None
    assert receptor.headers['Content-Type'] == 'application/json'

    payload = receptor.payload
    assert payload is not None
    assert 'version' not in payload
    assert payload['model'] == 'kumo-rfm'
    assert payload['task']['kind'] == 'regression'
    assert payload['schema']['relationships']
    assert payload['context']['instance_table']['format'] == 'arrays'
    assert payload['predict']['instance_table']['format'] == 'arrays'
    assert payload['task']['target']['column_name'] in payload['context'][
        'instance_table']['columns']
    assert payload['task']['target']['column_name'] not in payload['predict'][
        'instance_table']['columns']
    assert payload['schema']['instance_table']['primary_key'] == INSTANCE_ID
    predict_instance_table = payload['predict']['instance_table']
    predict_instance_id_index = predict_instance_table['columns'].index(
        INSTANCE_ID)
    assert predict_instance_table['rows'][0][predict_instance_id_index] != 3
    context_instance_table = payload['context']['instance_table']
    instance_id_index = context_instance_table['columns'].index(INSTANCE_ID)
    instance_ids = [row[instance_id_index]
                    for row in context_instance_table['rows']]
    assert len(instance_ids) == len(set(instance_ids))
    assert '__kumo_instance_feature' not in payload['context'][
        'instance_table']['columns']
    anchor_column = payload['task']['anchor_time_column']
    assert anchor_column.startswith(ANCHOR_TIME_PREFIX)
    assert anchor_column in payload['context']['instance_table']['columns']
    assert anchor_column in payload['predict']['instance_table']['columns']
    assert payload['schema']['instance_table']['columns'][anchor_column][
        'stype'] == 'timestamp'
    entity_relationship = payload['schema']['relationships'][0]
    assert 'source_table' not in entity_relationship
    assert entity_relationship['source_columns'][0].startswith(
        ENTITY_REFERENCE_PREFIX)
    assert entity_relationship['target_table'] == 'USERS'
    assert payload['schema']['related_tables']['USERS']['primary_key'] == (
        [INSTANCE_ID, 'USER_ID'])
    for schema in payload['schema']['related_tables'].values():
        assert INSTANCE_ID in schema['primary_key']
        assert schema['columns'][INSTANCE_ID]['nullable'] is False
    assert payload['schema']['related_tables']['ORDERS']['columns']['TIME'][
        'dtype'] == 'timestamp[us]'
    assert payload['context']['related_tables']['ORDERS']['rows'][0][
        payload['context']['related_tables']['ORDERS']['columns'].index(
            'TIME')].endswith('Z')
    assert payload['task']['target']['dtype'] == 'float32'
    assert 'embeddings' in payload['output']['fields']
    assert 'quantiles' in payload['output']['fields']
    assert payload['inference']['run_mode'] == 'fast'
    assert payload['inference']['inference_config']['kind'] == 'regression'
    assert payload['inference']['inference_config']['output_type'] == (
        'quantiles')
    assert 'operation' not in payload['metadata']
    _assert_payload_matches_local_prediction_request_schema(payload)

    payload_text = str(payload)
    assert 'application/x-protobuf' not in payload_text
    assert "'batch'" not in payload_text
    assert "'row'" not in payload_text
    assert "'col'" not in payload_text
    assert "'evaluate'" not in payload_text


def test_entity_identity_survives_batch_local_row_indexes(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
    mock_api: Any,
) -> None:
    mock_api.post(
        f'{MOCK_URL}/v1/predictions',
        json=_correlated_response({'prediction': 0.5}),
    )
    model = KumoRFM(user_store_graph, verbose=False)
    model._client = RFMAPI(
        KumoClient(MOCK_URL, api_key='DISABLED'))  # type: ignore

    with model.batch_mode(batch_size=1):
        result = model.predict(ltv, indices=[3, 1], verbose=False)

    assert result['ENTITY'].tolist() == [3, 1]


def test_explain_requests_explanation_output_field(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
    mock_api: Any,
) -> None:
    receptor = JsonPayloadReceptor()
    mock_api.post(
        f'{MOCK_URL}/v1/predictions',
        additional_matcher=receptor,
        json=_correlated_response({
            'prediction': 0.5,
            'explanation': {
                'format': 'natural_language_summary',
                'summary': 'Order frequency dropped.',
                'warning': 'Cross-region fallback used.',
            },
        }),
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


def _assert_payload_matches_local_prediction_request_schema(
        payload: dict[str, Any]) -> None:
    spec = _load_local_canonical_spec_at_generated_revision()
    if spec is None:
        return

    schema = spec['components']['schemas']['PredictionRequest']
    assert set(payload) == set(schema['properties'])
    assert set(schema['required']) <= set(payload)
    assert 'operation' not in payload.get('metadata', {})
    assert 'evaluate' not in payload


def _load_local_canonical_spec_at_generated_revision() -> dict | None:
    if not CANONICAL_SPEC.exists():
        return None
    output = Path('kumoai/client/generated/tfm_api.py')
    if _generated_source_sha(output) != _file_sha256(CANONICAL_SPEC):
        return None
    yaml = pytest.importorskip('yaml')
    return yaml.safe_load(CANONICAL_SPEC.read_text())


def _generated_source_sha(path: Path) -> str:
    match = re.search(r'^# Source SHA256: ([0-9a-f]+)$',
                      path.read_text(),
                      flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

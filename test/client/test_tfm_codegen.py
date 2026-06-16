import json
import subprocess
import sys
from pathlib import Path

from kumoai.client.endpoints import HTTPMethod
from kumoai.client.generated.tfm_api import (
    PredictionResponse,
    TFM_API_VERSION,
    TFM_ENDPOINTS_BY_OPERATION_ID,
    TFM_MODEL_KUMO_RFM,
    TFM_OUTPUT_FIELD_EMBEDDINGS,
    TFM_OUTPUT_FIELD_EXPLANATION,
    TFMOperations,
)


def test_generated_tfm_api_runtime_metadata() -> None:
    operation = TFMOperations.create_prediction

    assert TFM_API_VERSION == 'v1'
    assert TFM_MODEL_KUMO_RFM == 'kumo-rfm'
    assert TFM_OUTPUT_FIELD_EMBEDDINGS == 'embeddings'
    assert TFM_OUTPUT_FIELD_EXPLANATION == 'explanation'
    assert operation.operation_id == 'createPrediction'
    assert operation.request_schema == 'PredictionRequest'
    assert operation.response_schema == 'PredictionResponse'
    assert operation.endpoint.method == HTTPMethod.POST
    assert operation.endpoint.get_path() == '/predictions'
    assert TFM_ENDPOINTS_BY_OPERATION_ID['createPrediction'] == (
        operation.endpoint)


def test_generated_prediction_response_parser() -> None:
    response = PredictionResponse.from_dict({
        'id': 'pred-1',
        'model': 'kumo-rfm',
        'predictions': [{
            'id': 7,
            'prediction': True,
            'probabilities': {
                'false': 0.25,
                'true': 0.75,
            },
            'embeddings': [0.1, 0.2],
            'explanation': {
                'format': 'natural_language_summary',
                'summary': 'Order frequency dropped.',
            },
        }],
        'metadata': {
            'version': 'v1',
            'task_kind': 'classification',
        },
    })

    assert response.id == 'pred-1'
    assert response.model == 'kumo-rfm'
    assert response.metadata['version'] == 'v1'
    assert len(response.predictions) == 1
    item = response.predictions[0]
    assert item.id == '7'
    assert item.prediction is True
    assert item.probabilities == {'false': 0.25, 'true': 0.75}
    assert item.embeddings == (0.1, 0.2)
    assert item.explanation == {
        'format': 'natural_language_summary',
        'summary': 'Order frequency dropped.',
    }


def test_generator_creates_minimal_bindings(tmp_path: Path) -> None:
    spec = tmp_path / 'api_spec.json'
    output = tmp_path / 'generated.py'
    spec.write_text(json.dumps(_minimal_openapi_spec()))

    subprocess.run(
        [
            sys.executable,
            'scripts/generate_tfm_api.py',
            '--spec',
            str(spec),
            '--output',
            str(output),
        ],
        check=True,
    )

    generated = output.read_text()
    assert "Source: " in generated
    assert "class TFMOperations" in generated
    assert "class PredictionResponse" in generated
    assert "create_prediction: Final[TFMOperation]" in generated
    assert "path='/predictions'" in generated
    assert "TFM_MODEL_KUMO_RFM: Final[str] = 'kumo-rfm'" in generated
    assert "TFM_OUTPUT_FIELD_EMBEDDINGS: Final[str] = 'embeddings'" in generated
    assert "TFM_OUTPUT_FIELD_EXPLANATION: Final[str] = 'explanation'" in generated

    subprocess.run(
        [
            sys.executable,
            'scripts/generate_tfm_api.py',
            '--spec',
            str(spec),
            '--output',
            str(output),
            '--check',
        ],
        check=True,
    )

    output.write_text(generated + '\n# stale edit\n')
    result = subprocess.run(
        [
            sys.executable,
            'scripts/generate_tfm_api.py',
            '--spec',
            str(spec),
            '--output',
            str(output),
            '--check',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert 'not up to date' in result.stderr


def _minimal_openapi_spec() -> dict:
    return {
        'openapi': '3.0.0',
        'paths': {
            '/v1/predictions': {
                'post': {
                    'operationId': 'createPrediction',
                    'summary': 'Run a one-shot TFM prediction',
                    'requestBody': {
                        'content': {
                            'application/json': {
                                'schema': {
                                    '$ref':
                                    '#/components/schemas/PredictionRequest',
                                },
                            },
                        },
                    },
                    'responses': {
                        '200': {
                            'content': {
                                'application/json': {
                                    'schema': {
                                        '$ref':
                                        '#/components/schemas/'
                                        'PredictionResponse',
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
        'components': {
            'schemas': {
                'PredictionRequest': {
                    'properties': {
                        'version': {
                            'enum': ['v1'],
                        },
                        'model': {
                            'enum': ['tabicl', 'kumo-rfm'],
                        },
                    },
                },
                'PredictionItem': {},
                'PredictionResponse': {},
                'OutputSpec': {
                    'properties': {
                        'fields': {
                            'items': {
                                'enum': [
                                    'prediction',
                                    'probabilities',
                                    'embeddings',
                                    'explanation',
                                ],
                            },
                        },
                    },
                },
                'TaskSpec': {
                    'properties': {
                        'kind': {
                            'enum': ['classification', 'regression'],
                        },
                    },
                },
            },
        },
    }

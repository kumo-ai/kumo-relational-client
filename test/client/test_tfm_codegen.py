from dataclasses import fields
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from kumoai.client.endpoints import HTTPMethod
from kumoai.client.rfm import _prediction_item_to_row
from kumoai.client.generated.tfm_api import (
    PredictionItem,
    PredictionResponse,
    TFM_API_VERSION,
    TFM_ENDPOINTS_BY_OPERATION_ID,
    TFM_MODEL_KUMO_RFM,
    TFM_OUTPUT_FIELD_EMBEDDINGS,
    TFM_OUTPUT_FIELD_EXPLANATION,
    TFM_OUTPUT_FIELD_VALUES,
    TFMOperations,
)


CANONICAL_SPEC = Path('../structured-data-api/api_spec.yaml')


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
            'scores': [0.4, '0.6'],
            'rankings': [{
                'id': 11,
                'score': '0.7',
            }],
            'embeddings': [0.1, 0.2],
            'quantiles': {
                '0.5': '1.25',
            },
            'explanation': {
                'format': 'natural_language_summary',
                'summary': 'Order frequency dropped.',
            },
            'metadata': {
                'adapter_status': 'stubbed',
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
    assert item.scores == (0.4, 0.6)
    assert item.rankings == ({'id': 11, 'score': '0.7'}, )
    assert item.embeddings == (0.1, 0.2)
    assert item.quantiles == {'0.5': 1.25}
    assert item.explanation == {
        'format': 'natural_language_summary',
        'summary': 'Order frequency dropped.',
    }
    assert item.metadata == {'adapter_status': 'stubbed'}


def test_prediction_item_adapter_maps_known_fields() -> None:
    item = PredictionItem(
        id='7',
        prediction='yes',
        probabilities={
            'no': 0.2,
            'yes': 0.8,
        },
        scores=(0.3, 0.7),
        rankings=({
            'id': 'merchant-1',
            'score': 0.9,
        }, ),
        embeddings=(0.1, 0.2),
        quantiles={
            'p50': 12.5,
        },
        explanation={
            'format': 'natural_language_summary',
            'summary': 'Recent purchases increased.',
        },
        metadata={
            'adapter_status': 'stubbed',
        },
    )

    row = _prediction_item_to_row(item)

    assert row == {
        'ENTITY': 7,
        'prediction': 'yes',
        'no_PROB': 0.2,
        'yes_PROB': 0.8,
        'scores': [0.3, 0.7],
        'rankings': [{
            'id': 'merchant-1',
            'score': 0.9,
        }],
        'embeddings': [0.1, 0.2],
        'q_p50': 12.5,
        'explanation': {
            'format': 'natural_language_summary',
            'summary': 'Recent purchases increased.',
        },
    }

    mapped_fields = {
        'id',
        'prediction',
        'probabilities',
        'scores',
        'rankings',
        'embeddings',
        'quantiles',
        'explanation',
    }
    parsed_fields = {field.name for field in fields(PredictionItem)}
    intentionally_unmapped_fields = {'metadata'}
    assert parsed_fields == mapped_fields | intentionally_unmapped_fields


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

    subprocess.run(
        [
            sys.executable,
            'scripts/generate_tfm_api.py',
            '--spec',
            str(spec),
            '--output',
            str(output),
            '--validate-contract',
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


def test_generator_validation_reports_spec_drift(tmp_path: Path) -> None:
    from scripts.generate_tfm_api import (
        generate_code,
        validate_generated_code,
    )

    spec = _minimal_openapi_spec()
    spec_text = json.dumps(spec)
    code = generate_code(
        spec,
        spec_text=spec_text,
        source='inline',
        output_path=tmp_path / 'generated.py',
    )
    assert validate_generated_code(spec, code) == []

    del spec['components']['schemas']['PredictionItem']['properties'][
        'metadata']
    errors = validate_generated_code(spec, code)
    assert any('PredictionItem fields differ' in error for error in errors)


def test_generator_validation_reports_output_enum_drift(
        tmp_path: Path) -> None:
    from scripts.generate_tfm_api import (
        generate_code,
        validate_generated_code,
    )

    spec = _minimal_openapi_spec()
    spec_text = json.dumps(spec)
    code = generate_code(
        spec,
        spec_text=spec_text,
        source='inline',
        output_path=tmp_path / 'generated.py',
    )
    assert validate_generated_code(spec, code) == []

    spec['components']['schemas']['OutputSpec']['properties']['fields'][
        'items']['enum'].append('attributions')
    errors = validate_generated_code(spec, code)
    assert any('TFM_OUTPUT_FIELD_VALUES differs' in error for error in errors)


@pytest.mark.skipif(
    not CANONICAL_SPEC.exists(),
    reason='canonical structured-data-api checkout is not available beside '
    'this repo',
)
def test_generated_tfm_api_matches_local_canonical_spec() -> None:
    spec = _load_local_canonical_spec_at_generated_revision()
    from scripts.generate_tfm_api import generate_code_from_source

    output = Path('kumoai/client/generated/tfm_api.py')
    expected = generate_code_from_source(
        str(CANONICAL_SPEC),
        output_path=output,
    )
    assert output.read_text() == expected

    # Keep the loaded spec live so the skip guard above cannot be accidentally
    # removed without also updating this test.
    assert spec['paths']['/v1/predictions']['post']['operationId'] == (
        'createPrediction')


@pytest.mark.skipif(
    not CANONICAL_SPEC.exists(),
    reason='canonical structured-data-api checkout is not available beside '
    'this repo',
)
def test_generated_tfm_api_contract_matches_local_canonical_spec() -> None:
    spec = _load_local_canonical_spec_at_generated_revision()
    from scripts.generate_tfm_api import validate_generated_code

    output = Path('kumoai/client/generated/tfm_api.py')
    assert validate_generated_code(spec, output.read_text()) == []

    schemas = spec['components']['schemas']
    assert tuple(field.name for field in fields(PredictionItem)) == tuple(
        schemas['PredictionItem']['properties'])
    assert tuple(field.name for field in fields(PredictionResponse)) == tuple(
        schemas['PredictionResponse']['properties'])
    assert set(schemas['PredictionItem']['required']) <= {
        field.name
        for field in fields(PredictionItem)
    }
    assert set(schemas['PredictionResponse']['required']) <= {
        field.name
        for field in fields(PredictionResponse)
    }
    assert TFM_OUTPUT_FIELD_VALUES == tuple(
        schemas['OutputSpec']['properties']['fields']['items']['enum'])
    assert TFMOperations.create_prediction.response_schema == (
        'PredictionResponse')


@pytest.mark.skipif(
    not CANONICAL_SPEC.exists(),
    reason='canonical structured-data-api checkout is not available beside '
    'this repo',
)
def test_documented_prediction_response_examples_parse() -> None:
    spec = _load_local_canonical_spec_at_generated_revision()
    examples = spec['paths']['/v1/predictions']['post']['responses']['200'][
        'content']['application/json']['examples']
    required = set(
        spec['components']['schemas']['PredictionResponse']['required'])

    for name, example in examples.items():
        response = PredictionResponse.from_dict(example['value'])
        assert required <= set(example['value'])
        assert response.id
        assert response.model
        assert response.metadata['version'] == 'v1'
        assert response.predictions, name
        for item in response.predictions:
            assert isinstance(item.id, str)
            if item.probabilities is not None:
                assert all(
                    isinstance(value, float)
                    for value in item.probabilities.values())
            if item.embeddings is not None:
                assert isinstance(item.embeddings, tuple)
            if item.scores is not None:
                assert isinstance(item.scores, tuple)
            if item.explanation is not None:
                assert isinstance(item.explanation, dict)
            if item.metadata is not None:
                assert isinstance(item.metadata, dict)


@pytest.mark.skipif(
    not CANONICAL_SPEC.exists(),
    reason='canonical structured-data-api checkout is not available beside '
    'this repo',
)
def test_documented_prediction_request_examples_match_envelope_shape() -> None:
    spec = _load_local_canonical_spec_at_generated_revision()
    request_schema = spec['components']['schemas']['PredictionRequest']
    examples = spec['paths']['/v1/predictions']['post']['requestBody'][
        'content']['application/json']['examples']
    property_names = set(request_schema['properties'])
    required = set(request_schema['required'])

    assert {'tabicl_arrays', 'kumo_rfm_arrays'} <= set(examples)
    for example in examples.values():
        value = example['value']
        assert set(value) <= property_names
        assert required <= set(value)
        _assert_request_envelope_shape(value)


def _generated_source_sha(path: Path) -> str:
    match = re.search(r'^# Source SHA256: ([0-9a-f]+)$',
                      path.read_text(),
                      flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_local_canonical_spec_at_generated_revision() -> dict:
    yaml = pytest.importorskip('yaml')
    output = Path('kumoai/client/generated/tfm_api.py')
    generated_source_sha = _generated_source_sha(output)
    if generated_source_sha != _file_sha256(CANONICAL_SPEC):
        pytest.skip(
            'local structured-data-api checkout is not the generated spec '
            'revision')
    return yaml.safe_load(CANONICAL_SPEC.read_text())


def _assert_request_envelope_shape(value: dict) -> None:
    assert value['version'] == 'v1'
    assert value['model'] in {'tabicl', 'kumo-rfm'}
    assert 'kind' in value['task']
    assert 'instance_table' in value['schema']
    assert 'instance_table' in value['context']
    assert 'instance_table' in value['predict']
    assert isinstance(value['output']['fields'], list)
    assert value['output']['fields']
    assert 'operation' not in value.get('metadata', {})
    assert 'evaluate' not in value


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
                    'required': [
                        'version',
                        'model',
                        'task',
                        'schema',
                        'context',
                        'predict',
                        'output',
                    ],
                    'properties': {
                        'version': {
                            'enum': ['v1'],
                        },
                        'model': {
                            'enum': ['tabicl', 'kumo-rfm'],
                        },
                        'task': {},
                        'schema': {},
                        'context': {},
                        'predict': {},
                        'output': {},
                        'inference': {},
                        'metadata': {},
                    },
                },
                'PredictionItem': {
                    'required': ['id'],
                    'properties': {
                        'id': {},
                        'prediction': {},
                        'probabilities': {},
                        'scores': {},
                        'rankings': {},
                        'embeddings': {},
                        'quantiles': {},
                        'explanation': {},
                        'metadata': {},
                    },
                },
                'PredictionResponse': {
                    'required': [
                        'id',
                        'model',
                        'predictions',
                        'metadata',
                    ],
                    'properties': {
                        'id': {},
                        'model': {},
                        'predictions': {},
                        'metadata': {},
                    },
                },
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

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import pytest
import requests

from rfm_nim_payloads import (
    SDK_CONFIG_PATH,
    NIM_V1_PREDICTION_PATH,
    NIM_V1_SESSIONS_PATH,
    SDK_V1_CONNECTORS_PATH,
    SDK_V1_RFM_PARSE_QUERY_PATH,
    SDK_V1_RFM_VALIDATE_QUERY_PATH,
    nim_v1_empty_predict_rows_payload,
    nim_v1_explicit_utc_offset_timestamp_payload,
    nim_v1_fast_run_mode_payload,
    nim_v1_multiclass_payload,
    nim_v1_prediction_only_output_payload,
    nim_v1_regression_payload,
    nim_v1_reordered_predict_rows_payload,
    nim_v1_session_create_payload,
    nim_v1_session_predict_minimal_payload,
    nim_v1_smoke_payload,
    nim_v1_two_predict_rows_payload,
    nim_v1_without_inference_payload,
)

_ENV_VAR = 'RFM_NIM_BASE_URL'
_TIMEOUT_SECONDS = 30

pytestmark = [
    pytest.mark.live_nim,
    pytest.mark.skipif(
        not os.environ.get(_ENV_VAR),
        reason=f'set {_ENV_VAR} to run live Kumo RFM NIM tests',
    ),
]


@pytest.fixture(scope='module')
def nim_base_url() -> str:
    return _normalize_base_url(os.environ[_ENV_VAR])


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip('/')


def _url(base_url: str, path: str) -> str:
    return f'{base_url}{path}'


def _request(
    method: str,
    base_url: str,
    path: str,
    **kwargs: Any,
) -> requests.Response:
    return requests.request(
        method,
        _url(base_url, path),
        timeout=_TIMEOUT_SECONDS,
        **kwargs,
    )


def _assert_json_response(response: requests.Response) -> dict[str, Any]:
    assert response.headers.get('content-type', '').startswith(
        'application/json')
    body = response.json()
    assert isinstance(body, dict)
    return body


def _assert_problem_details(
    response: requests.Response,
    expected_status: int,
    expected_code: str | None = None,
) -> dict[str, Any]:
    assert response.status_code == expected_status
    assert response.headers.get('content-type', '').startswith(
        'application/problem+json')
    body = response.json()
    assert isinstance(body, dict)
    assert body['status'] == expected_status
    assert isinstance(body.get('title'), str)
    assert isinstance(body.get('detail'), str)
    if expected_code is not None:
        assert body['code'] == expected_code
    return body


def _assert_prediction_response_invariants(
    response: requests.Response,
    *,
    expected_count: int = 1,
    expected_task_kind: str = 'classification',
    expected_ids: list[str] | None = None,
    expect_probabilities: bool = True,
    expected_probability_labels: set[str] | None = None,
) -> dict[str, Any]:
    assert response.status_code == 200
    body = _assert_json_response(response)
    assert isinstance(body.get('id'), str)
    assert body['id']
    assert body['model'] == 'kumo-rfm'

    predictions = body['predictions']
    assert isinstance(predictions, list)
    assert len(predictions) == expected_count
    if expected_ids is not None:
        assert [prediction.get('id') for prediction in predictions] == (
            expected_ids)

    for index, prediction in enumerate(predictions):
        assert isinstance(prediction, dict)
        if 'id' in prediction:
            assert isinstance(prediction['id'], str)
        else:
            assert prediction['row_index'] == index
        assert 'prediction' in prediction
        assert isinstance(prediction.get('metadata'), dict)
        assert prediction['metadata']['backend_status'] == 'driver'

        if not expect_probabilities:
            assert 'probabilities' not in prediction
            continue

        probabilities = prediction['probabilities']
        assert isinstance(probabilities, dict)
        assert probabilities
        if expected_probability_labels is not None:
            assert set(probabilities) == expected_probability_labels
        for value in probabilities.values():
            assert isinstance(value, (int, float))
            assert not isinstance(value, bool)
            assert 0 <= value <= 1
        assert sum(probabilities.values()) == pytest.approx(1.0)

    if expected_count == 0:
        assert predictions == []

    metadata = body['metadata']
    assert isinstance(metadata, dict)
    assert metadata['task_kind'] == expected_task_kind
    assert metadata['adapter'] == 'kumo_rfm'
    assert metadata['backend_status'] == 'driver'
    return body


@pytest.mark.parametrize(
    ('raw_base_url', 'expected_base_url'),
    [
        ('http://localhost:8000', 'http://localhost:8000'),
        ('http://localhost:8000/', 'http://localhost:8000'),
        ('http://localhost:8000///', 'http://localhost:8000'),
        ('https://nim.example.test/rfm/', 'https://nim.example.test/rfm'),
    ],
)
def test_live_nim_base_url_trailing_slash_normalization(
    raw_base_url: str,
    expected_base_url: str,
) -> None:
    normalized = _normalize_base_url(raw_base_url)

    assert normalized == expected_base_url
    assert _url(normalized, '/v1/health/live') == (
        f'{expected_base_url}/v1/health/live')


@pytest.mark.parametrize(
    'path',
    [
        '/v1/health/live',
        '/v1/health/ready',
        '/v1/metadata',
        '/v1/version',
        '/v1/models',
    ],
)
def test_live_nim_available_get_endpoints(
    nim_base_url: str,
    path: str,
) -> None:
    response = _request('GET', nim_base_url, path)

    assert response.status_code == 200


@pytest.mark.parametrize(
    ('method', 'path', 'payload_factory'),
    [
        ('GET', '/v1/capabilities', None),
        ('GET', '/v0/capabilities', None),
        ('GET', '/v1/models/kumo-rfm/capabilities', None),
        ('GET', '/v0/models/kumo-rfm/capabilities', None),
        ('GET', SDK_V1_CONNECTORS_PATH, None),
        ('GET', SDK_CONFIG_PATH, None),
        ('GET', '/api/config', None),
        ('POST', SDK_V1_RFM_PARSE_QUERY_PATH, None),
        ('POST', SDK_V1_RFM_VALIDATE_QUERY_PATH, None),
    ],
)
def test_live_nim_currently_absent_endpoints_return_404(
    nim_base_url: str,
    method: str,
    path: str,
    payload_factory: Callable[[], dict[str, Any]] | None,
) -> None:
    kwargs = {'json': payload_factory()} if payload_factory is not None else {}
    response = _request(method, nim_base_url, path, **kwargs)

    assert response.status_code == 404


@pytest.mark.parametrize(
    ('path', 'allowed_method'),
    [
        (NIM_V1_PREDICTION_PATH, 'POST'),
        (NIM_V1_SESSIONS_PATH, 'POST'),
    ],
)
def test_live_nim_v1_write_endpoints_reject_safe_method_mismatches(
    nim_base_url: str,
    path: str,
    allowed_method: str,
) -> None:
    response = _request('GET', nim_base_url, path)

    assert response.status_code == 405
    assert allowed_method in response.headers.get('allow', '')


def test_live_nim_v1_prediction_accepts_container_smoke_payload(
    nim_base_url: str,
) -> None:
    response = _request(
        'POST',
        nim_base_url,
        NIM_V1_PREDICTION_PATH,
        json=nim_v1_smoke_payload(),
    )

    _assert_prediction_response_invariants(
        response,
        expected_probability_labels={'False', 'True'},
    )


@pytest.mark.parametrize(
    ('payload_factory', 'assertion_kwargs'),
    [
        pytest.param(
            nim_v1_prediction_only_output_payload,
            {'expect_probabilities': False},
            id='prediction-only-output',
        ),
        pytest.param(
            nim_v1_fast_run_mode_payload,
            {'expected_probability_labels': {'False', 'True'}},
            id='fast-run-mode',
        ),
        pytest.param(
            nim_v1_without_inference_payload,
            {'expected_probability_labels': {'False', 'True'}},
            id='default-inference',
        ),
        pytest.param(
            nim_v1_two_predict_rows_payload,
            {
                'expected_count': 2,
                'expected_ids': ['601', '602'],
                'expected_probability_labels': {'False', 'True'},
            },
            id='two-predict-rows',
        ),
        pytest.param(
            nim_v1_reordered_predict_rows_payload,
            {
                'expected_count': 2,
                'expected_ids': ['602', '601'],
                'expected_probability_labels': {'False', 'True'},
            },
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    'Prediction response should preserve predict row order, but '
                    'the current NIM returns predictions sorted by entity id.'),
            ),
            id='reordered-predict-rows',
        ),
        pytest.param(
            nim_v1_empty_predict_rows_payload,
            {
                'expected_count': 0,
                'expect_probabilities': False,
            },
            id='empty-predict-rows',
        ),
        pytest.param(
            nim_v1_explicit_utc_offset_timestamp_payload,
            {'expected_probability_labels': {'False', 'True'}},
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    'ISO timestamps with explicit UTC offsets should be accepted '
                    'or rejected cleanly, but the current NIM returns 500.'),
            ),
            id='timestamp-offsets',
        ),
        pytest.param(
            nim_v1_regression_payload,
            {
                'expected_task_kind': 'regression',
                'expect_probabilities': False,
            },
            id='regression',
        ),
        pytest.param(
            nim_v1_multiclass_payload,
            {'expected_task_kind': 'multiclass_classification'},
            id='multiclass-categorical-target',
        ),
    ],
)
def test_live_nim_v1_prediction_accepts_deterministic_variants(
    nim_base_url: str,
    payload_factory: Callable[[], dict[str, Any]],
    assertion_kwargs: dict[str, Any],
) -> None:
    response = _request(
        'POST',
        nim_base_url,
        NIM_V1_PREDICTION_PATH,
        json=payload_factory(),
    )

    _assert_prediction_response_invariants(response, **assertion_kwargs)


def _invalid_run_mode_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['inference']['run_mode'] = 'turbo'
    return payload


def _unsupported_output_field_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['output']['fields'].append('feature_importances')
    return payload


def _unsupported_task_kind_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['task']['kind'] = 'forecasting'
    return payload


def _missing_target_column_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['task']['target']['column_name'] = 'missing_status'
    return payload


def _unknown_predict_column_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['predict']['instance_table']['columns'].append('leakage')
    payload['predict']['instance_table']['rows'][0].append('not-declared')
    return payload


def _invalid_timestamp_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['predict']['instance_table']['rows'][0][1] = '2025-02-01 00:00:00'
    return payload


def _related_row_unknown_instance_key_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['context']['related_tables']['accounts']['rows'].append(
        [999, 'orphan'])
    return payload


def _relationship_source_key_mismatch_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['schema']['relationships'][0]['source_columns'] = [
        'missing_account_id',
    ]
    return payload


def _duplicate_predict_primary_key_payload() -> dict[str, Any]:
    payload = nim_v1_two_predict_rows_payload()
    payload['predict']['instance_table']['rows'][1][0] = 601
    return payload


def _missing_predict_entity_related_table_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['predict']['related_tables'].pop('accounts')
    return payload


def _incomplete_predict_entity_related_rows_payload() -> dict[str, Any]:
    payload = nim_v1_two_predict_rows_payload()
    payload['predict']['related_tables']['accounts']['rows'] = [[601, 'enterprise']]
    return payload


def _missing_context_entity_related_table_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['context']['related_tables'].pop('accounts')
    return payload


def _incomplete_context_entity_related_rows_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['context']['related_tables']['accounts']['rows'] = [[501, 'enterprise']]
    return payload


def _empty_context_rows_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['context']['instance_table']['rows'] = []
    payload['context']['related_tables']['accounts']['rows'] = []
    return payload


def _missing_entity_names_and_instance_primary_key_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['task'].pop('entity_table_names')
    payload['schema']['instance_table'].pop('primary_key')
    return payload


def _regression_null_context_target_payload() -> dict[str, Any]:
    payload = nim_v1_regression_payload()
    payload['context']['instance_table']['rows'][0][1] = None
    return payload


def _multiclass_null_context_target_payload() -> dict[str, Any]:
    payload = nim_v1_multiclass_payload()
    payload['context']['instance_table']['rows'][0][1] = None
    return payload


@pytest.mark.parametrize(
    ('payload_factory', 'expected_code', 'detail_fragment'),
    [
        pytest.param(
            _invalid_run_mode_payload,
            'INVALID_SCHEMA',
            'run_mode',
            id='invalid-run-mode',
        ),
        pytest.param(
            _unsupported_output_field_payload,
            'UNSUPPORTED_OUTPUT',
            'feature_importances',
            id='unsupported-output-field',
        ),
        pytest.param(
            _unsupported_task_kind_payload,
            'UNSUPPORTED_TASK',
            'forecasting',
            id='unsupported-task-kind',
        ),
        pytest.param(
            _missing_target_column_payload,
            'TASK_SCHEMA_MISMATCH',
            'missing_status',
            id='missing-target-column',
        ),
        pytest.param(
            _unknown_predict_column_payload,
            'VALIDATION_FAILED',
            'leakage',
            id='unknown-predict-column',
        ),
        pytest.param(
            _invalid_timestamp_payload,
            'VALIDATION_FAILED',
            'timestamp',
            id='invalid-timestamp',
        ),
        pytest.param(
            _related_row_unknown_instance_key_payload,
            'INVALID_SCHEMA',
            'unknown instance key',
            id='related-row-unknown-instance-key',
        ),
        pytest.param(
            _relationship_source_key_mismatch_payload,
            'INVALID_SCHEMA',
            'missing_account_id',
            id='relationship-source-key-mismatch',
        ),
        pytest.param(
            _duplicate_predict_primary_key_payload,
            'INVALID_SCHEMA',
            'duplicate',
            id='duplicate-predict-primary-key',
        ),
        pytest.param(
            _missing_predict_entity_related_table_payload,
            'INVALID_SCHEMA',
            'accounts',
            id='missing-predict-entity-related-table',
        ),
        pytest.param(
            _incomplete_predict_entity_related_rows_payload,
            'INVALID_SCHEMA',
            'accounts',
            id='incomplete-predict-entity-related-rows',
        ),
        pytest.param(
            _missing_context_entity_related_table_payload,
            'INVALID_SCHEMA',
            'accounts',
            id='missing-context-entity-related-table',
        ),
        pytest.param(
            _incomplete_context_entity_related_rows_payload,
            'INVALID_SCHEMA',
            'accounts',
            id='incomplete-context-entity-related-rows',
        ),
        pytest.param(
            _empty_context_rows_payload,
            'VALIDATION_FAILED',
            'context',
            id='empty-context-rows',
        ),
        pytest.param(
            _missing_entity_names_and_instance_primary_key_payload,
            'INVALID_SCHEMA',
            'primary_key',
            id='missing-entity-names-and-primary-key',
        ),
    ],
)
def test_live_nim_v1_prediction_rejects_deterministic_variants(
    nim_base_url: str,
    payload_factory: Callable[[], dict[str, Any]],
    expected_code: str,
    detail_fragment: str,
) -> None:
    response = _request(
        'POST',
        nim_base_url,
        NIM_V1_PREDICTION_PATH,
        json=payload_factory(),
    )

    body = _assert_problem_details(
        response,
        expected_status=422,
        expected_code=expected_code,
    )
    assert detail_fragment in str(body)


def test_live_nim_v1_prediction_accepts_metadata_field(
    nim_base_url: str,
) -> None:
    payload = nim_v1_smoke_payload()
    payload['metadata'] = {'source_query': 'rfm-sdk-nim-contract-smoke'}

    response = _request(
        'POST',
        nim_base_url,
        NIM_V1_PREDICTION_PATH,
        json=payload,
    )

    _assert_prediction_response_invariants(response)


def test_live_nim_v1_prediction_rejects_empty_body_with_problem_details(
    nim_base_url: str,
) -> None:
    response = _request(
        'POST',
        nim_base_url,
        NIM_V1_PREDICTION_PATH,
        json={},
    )

    body = _assert_problem_details(
        response,
        expected_status=422,
        expected_code='VALIDATION_FAILED',
    )
    assert body['errors']


def test_live_nim_v1_prediction_rejects_missing_required_field(
    nim_base_url: str,
) -> None:
    payload = nim_v1_smoke_payload()
    payload.pop('schema')

    response = _request(
        'POST',
        nim_base_url,
        NIM_V1_PREDICTION_PATH,
        json=payload,
    )

    body = _assert_problem_details(
        response,
        expected_status=422,
        expected_code='VALIDATION_FAILED',
    )
    assert body['errors']
    assert any('schema' in str(error) for error in body['errors'])


def test_live_nim_v1_session_prediction_rejects_missing_predict_entity_related_table(
    nim_base_url: str,
) -> None:
    created = _request(
        'POST',
        nim_base_url,
        NIM_V1_SESSIONS_PATH,
        json=nim_v1_session_create_payload(),
    )
    assert created.status_code == 201
    session_id = str(created.json()['session_id'])
    session_url = f'{NIM_V1_SESSIONS_PATH}/{quote(session_id, safe="")}'
    session_predictions_url = f'{session_url}/predictions'

    try:
        payload = nim_v1_session_predict_minimal_payload()
        payload['predict']['related_tables'].pop('accounts')
        response = _request(
            'POST',
            nim_base_url,
            session_predictions_url,
            json=payload,
        )
        _assert_problem_details(
            response,
            expected_status=422,
            expected_code='SESSION_PREDICT_SCHEMA_MISMATCH',
        )
    finally:
        delete_response = _request('DELETE', nim_base_url, session_url)
        assert delete_response.status_code in (204, 404)


def test_live_nim_v1_session_rejects_incomplete_context_entity_related_rows(
    nim_base_url: str,
) -> None:
    payload = nim_v1_session_create_payload()
    payload['context']['related_tables']['accounts']['rows'] = [[501, 'enterprise']]
    created = _request(
        'POST',
        nim_base_url,
        NIM_V1_SESSIONS_PATH,
        json=payload,
    )
    if created.status_code != 201:
        _assert_problem_details(
            created,
            expected_status=422,
            expected_code='SESSION_CREATE_VALIDATION_FAILED',
        )
        return

    session_id = str(created.json()['session_id'])
    session_url = f'{NIM_V1_SESSIONS_PATH}/{quote(session_id, safe="")}'
    session_predictions_url = f'{session_url}/predictions'
    try:
        response = _request(
            'POST',
            nim_base_url,
            session_predictions_url,
            json=nim_v1_session_predict_minimal_payload(),
        )
        _assert_problem_details(
            response,
            expected_status=422,
            expected_code='INVALID_SCHEMA',
        )
    finally:
        delete_response = _request('DELETE', nim_base_url, session_url)
        assert delete_response.status_code in (204, 404)


def test_live_nim_v1_session_create_predict_delete(
    nim_base_url: str,
) -> None:
    created = _request(
        'POST',
        nim_base_url,
        NIM_V1_SESSIONS_PATH,
        json=nim_v1_session_create_payload(),
    )
    assert created.status_code == 201
    body = created.json()
    session_id = str(body['session_id'])
    assert session_id
    assert body['ttl_seconds'] > 0
    assert body['expires_at']

    deleted_session = False
    session_url = f'{NIM_V1_SESSIONS_PATH}/{quote(session_id, safe="")}'
    session_predictions_url = f'{session_url}/predictions'

    try:
        predict = _request(
            'POST',
            nim_base_url,
            session_predictions_url,
            json=nim_v1_session_predict_minimal_payload(),
        )
        _assert_prediction_response_invariants(predict)

        delete_response = _request(
            'DELETE',
            nim_base_url,
            session_url,
        )
        assert delete_response.status_code == 204
        deleted_session = True

        after_delete = _request(
            'POST',
            nim_base_url,
            session_predictions_url,
            json=nim_v1_session_predict_minimal_payload(),
        )
        _assert_problem_details(after_delete, expected_status=404)
    finally:
        if not deleted_session:
            _request('DELETE', nim_base_url, session_url)


@pytest.mark.parametrize(
    ('payload_factory', 'expected_code', 'detail_fragment'),
    [
        pytest.param(
            _regression_null_context_target_payload,
            'VALIDATION_FAILED',
            'score',
            id='regression-null-context-target',
        ),
        pytest.param(
            _multiclass_null_context_target_payload,
            'VALIDATION_FAILED',
            'tier',
            id='multiclass-null-context-target',
        ),
    ],
)
def test_live_nim_v1_prediction_rejects_null_non_binary_context_targets_last(
    nim_base_url: str,
    payload_factory: Callable[[], dict[str, Any]],
    expected_code: str,
    detail_fragment: str,
) -> None:
    response = _request(
        'POST',
        nim_base_url,
        NIM_V1_PREDICTION_PATH,
        json=payload_factory(),
    )

    body = _assert_problem_details(
        response,
        expected_status=422,
        expected_code=expected_code,
    )
    assert detail_fragment in str(body)

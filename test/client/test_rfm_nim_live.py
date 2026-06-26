from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import pytest
import requests

from rfm_nim_payloads import (
    SDK_CONFIG_PATH,
    NIM_V0_PREDICTION_PATH,
    NIM_V0_SESSIONS_PATH,
    SDK_V1_CONNECTORS_PATH,
    SDK_V1_PREDICTION_PATH,
    SDK_V1_RFM_PARSE_QUERY_PATH,
    SDK_V1_RFM_VALIDATE_QUERY_PATH,
    SDK_V1_SESSIONS_PATH,
    nim_v0_session_create_payload,
    nim_v0_session_predict_minimal_payload,
    nim_v0_smoke_payload,
    sdk_v1_smoke_payload,
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
) -> dict[str, Any]:
    assert response.status_code == 200
    body = _assert_json_response(response)
    assert body['model'] == 'kumo-rfm'

    predictions = body['predictions']
    assert isinstance(predictions, list)
    assert len(predictions) == 1

    prediction = predictions[0]
    assert isinstance(prediction, dict)
    assert prediction['prediction'] is True

    probabilities = prediction['probabilities']
    assert isinstance(probabilities, dict)
    assert set(probabilities) == {'False', 'True'}
    for value in probabilities.values():
        assert isinstance(value, (int, float))
        assert not isinstance(value, bool)
        assert 0 <= value <= 1
    assert sum(probabilities.values()) == pytest.approx(1.0)

    metadata = body['metadata']
    assert isinstance(metadata, dict)
    assert metadata['task_kind'] == 'classification'
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
    assert _url(normalized, '/health/live') == (
        f'{expected_base_url}/health/live')


@pytest.mark.parametrize(
    'path',
    [
        '/health/live',
        '/health/ready',
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
        ('POST', SDK_V1_SESSIONS_PATH, nim_v0_session_create_payload),
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


def test_live_nim_sdk_v1_prediction_endpoint_remains_absent(
    nim_base_url: str,
) -> None:
    assert SDK_V1_PREDICTION_PATH == '/v1/predictions'
    assert NIM_V0_PREDICTION_PATH == '/v0/predictions'
    assert SDK_V1_PREDICTION_PATH != NIM_V0_PREDICTION_PATH

    response = _request(
        'POST',
        nim_base_url,
        SDK_V1_PREDICTION_PATH,
        json=sdk_v1_smoke_payload(),
    )

    assert response.status_code == 404


@pytest.mark.xfail(
    strict=True,
    reason=(
        'Known SDK/container compatibility bug: SDK-style /v1/predictions '
        'requests should work against the Kumo RFM NIM, but currently 404.'),
)
def test_live_nim_sdk_v1_prediction_endpoint_should_work(
    nim_base_url: str,
) -> None:
    response = _request(
        'POST',
        nim_base_url,
        SDK_V1_PREDICTION_PATH,
        json=sdk_v1_smoke_payload(),
    )

    _assert_prediction_response_invariants(response)


@pytest.mark.parametrize(
    ('sdk_health_path', 'current_nim_path'),
    [
        ('/v1/health/live', '/health/live'),
        ('/v1/health/ready', '/health/ready'),
    ],
)
def test_live_nim_sdk_health_routes_are_currently_absent(
    nim_base_url: str,
    sdk_health_path: str,
    current_nim_path: str,
) -> None:
    current_response = _request('GET', nim_base_url, current_nim_path)
    sdk_response = _request('GET', nim_base_url, sdk_health_path)

    assert current_response.status_code == 200
    assert sdk_response.status_code == 404


@pytest.mark.parametrize(
    'sdk_health_path',
    [
        '/v1/health/live',
        '/v1/health/ready',
    ],
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        'Known SDK/container compatibility bug: generated TFM health routes are '
        'version-prefixed by the SDK but absent from this NIM container.'),
)
def test_live_nim_sdk_health_routes_should_work(
    nim_base_url: str,
    sdk_health_path: str,
) -> None:
    response = _request('GET', nim_base_url, sdk_health_path)

    assert response.status_code == 200


@pytest.mark.parametrize(
    ('path', 'allowed_method'),
    [
        (NIM_V0_PREDICTION_PATH, 'POST'),
        (NIM_V0_SESSIONS_PATH, 'POST'),
    ],
)
def test_live_nim_v0_write_endpoints_reject_safe_method_mismatches(
    nim_base_url: str,
    path: str,
    allowed_method: str,
) -> None:
    response = _request('GET', nim_base_url, path)

    assert response.status_code == 405
    assert allowed_method in response.headers.get('allow', '')


def test_live_nim_v0_prediction_accepts_container_smoke_payload(
    nim_base_url: str,
) -> None:
    response = _request(
        'POST',
        nim_base_url,
        NIM_V0_PREDICTION_PATH,
        json=nim_v0_smoke_payload(),
    )

    _assert_prediction_response_invariants(response)


@pytest.mark.xfail(
    strict=True,
    reason=(
        'Known SDK/container compatibility bug: SDK-shaped prediction payloads '
        'include v1-only top-level fields that the current /v0/predictions '
        'contract rejects.'),
)
def test_live_nim_v0_prediction_should_accept_sdk_generated_payload_shape(
    nim_base_url: str,
) -> None:
    response = _request(
        'POST',
        nim_base_url,
        NIM_V0_PREDICTION_PATH,
        json=sdk_v1_smoke_payload(),
    )

    _assert_prediction_response_invariants(response)


def test_live_nim_v0_prediction_rejects_empty_body_with_problem_details(
    nim_base_url: str,
) -> None:
    response = _request(
        'POST',
        nim_base_url,
        NIM_V0_PREDICTION_PATH,
        json={},
    )

    body = _assert_problem_details(
        response,
        expected_status=422,
        expected_code='VALIDATION_FAILED',
    )
    assert body['errors']


def test_live_nim_v0_prediction_rejects_missing_required_field(
    nim_base_url: str,
) -> None:
    payload = nim_v0_smoke_payload()
    payload.pop('schema')

    response = _request(
        'POST',
        nim_base_url,
        NIM_V0_PREDICTION_PATH,
        json=payload,
    )

    body = _assert_problem_details(
        response,
        expected_status=422,
        expected_code='VALIDATION_FAILED',
    )
    assert body['errors']
    assert any('schema' in str(error) for error in body['errors'])


def test_live_nim_v0_session_create_predict_delete(
    nim_base_url: str,
) -> None:
    created = _request(
        'POST',
        nim_base_url,
        NIM_V0_SESSIONS_PATH,
        json=nim_v0_session_create_payload(),
    )
    assert created.status_code == 201
    body = created.json()
    session_id = str(body['session_id'])
    assert session_id
    assert body['ttl_seconds'] > 0
    assert body['expires_at']

    deleted_session = False
    session_url = f'{NIM_V0_SESSIONS_PATH}/{quote(session_id, safe="")}'
    session_predictions_url = f'{session_url}/predictions'

    try:
        predict = _request(
            'POST',
            nim_base_url,
            session_predictions_url,
            json=nim_v0_session_predict_minimal_payload(),
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
            json=nim_v0_session_predict_minimal_payload(),
        )
        _assert_problem_details(after_delete, expected_status=404)
    finally:
        if not deleted_session:
            _request('DELETE', nim_base_url, session_url)

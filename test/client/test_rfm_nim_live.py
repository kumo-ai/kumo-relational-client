from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import pytest
import requests

from rfm_nim_payloads import (
    NIM_V0_PREDICTION_PATH,
    NIM_V0_SESSIONS_PATH,
    SDK_V1_PREDICTION_PATH,
    nim_v0_session_create_payload,
    nim_v0_session_predict_minimal_payload,
    nim_v0_smoke_payload,
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
    return os.environ[_ENV_VAR].rstrip('/')


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
        ('POST', SDK_V1_PREDICTION_PATH, nim_v0_smoke_payload),
        ('POST', '/v1/sessions', nim_v0_session_create_payload),
        ('GET', '/v1/capabilities', None),
        ('GET', '/v0/capabilities', None),
        ('GET', '/v1/models/kumo-rfm/capabilities', None),
        ('GET', '/v0/models/kumo-rfm/capabilities', None),
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


def test_live_nim_v0_prediction_accepts_container_smoke_payload(
    nim_base_url: str,
) -> None:
    response = _request(
        'POST',
        nim_base_url,
        NIM_V0_PREDICTION_PATH,
        json=nim_v0_smoke_payload(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body['model'] == 'kumo-rfm'
    assert isinstance(body['predictions'], list)
    assert body['predictions']
    prediction = body['predictions'][0]
    assert prediction['prediction'] is True
    assert set(prediction['probabilities']) == {'False', 'True'}
    assert body['metadata']['task_kind'] == 'classification'
    assert body['metadata']['adapter'] == 'kumo_rfm'
    assert body['metadata']['backend_status'] == 'driver'


def test_live_nim_v0_prediction_rejects_empty_body_with_problem_details(
    nim_base_url: str,
) -> None:
    response = _request(
        'POST',
        nim_base_url,
        NIM_V0_PREDICTION_PATH,
        json={},
    )

    assert response.status_code == 422
    assert response.headers['content-type'].startswith(
        'application/problem+json')
    body = response.json()
    assert body['code'] == 'VALIDATION_FAILED'
    assert body['status'] == 422
    assert body['errors']


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

    try:
        predict = _request(
            'POST',
            nim_base_url,
            f'{NIM_V0_SESSIONS_PATH}/{quote(session_id, safe="")}/predictions',
            json=nim_v0_session_predict_minimal_payload(),
        )
        assert predict.status_code == 200
        prediction_body = predict.json()
        assert prediction_body['model'] == 'kumo-rfm'
        assert isinstance(prediction_body['predictions'], list)
        assert prediction_body['predictions']
        assert prediction_body['predictions'][0]['prediction'] is True
    finally:
        deleted = _request(
            'DELETE',
            nim_base_url,
            f'{NIM_V0_SESSIONS_PATH}/{quote(session_id, safe="")}',
        )
        assert deleted.status_code == 204

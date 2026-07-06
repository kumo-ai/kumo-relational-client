from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import requests

from rfm_nim_live_harness import (
    LiveNimClient,
    assert_prediction_response,
    assert_problem_details,
    assert_ready_and_model_available,
    normalize_base_url,
)
from rfm_nim_payloads import nim_v1_two_predict_rows_payload


def _response(
    status: int,
    media_type: str,
    body: dict,
) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.headers['Content-Type'] = media_type
    response._content = json.dumps(body).encode()
    return response


@pytest.mark.parametrize(
    ('raw_url', 'expected'),
    [
        ('http://localhost:8000', 'http://localhost:8000'),
        ('http://localhost:8000///', 'http://localhost:8000'),
        ('https://example.test/rfm/', 'https://example.test/rfm'),
    ],
)
def test_normalize_base_url(raw_url: str, expected: str) -> None:
    assert normalize_base_url(raw_url) == expected


@pytest.mark.parametrize(
    'raw_url',
    [
        'localhost:8000',
        'ftp://example.test/rfm',
        'https://user:password@example.test/rfm',
        'https://example.test/rfm?token=secret',
        'https://example.test/rfm#fragment',
    ],
)
def test_normalize_base_url_rejects_invalid_urls(raw_url: str) -> None:
    with pytest.raises(ValueError):
        normalize_base_url(raw_url)


@pytest.mark.parametrize(
    ('base_url', 'expected_error'),
    [
        (
            'https://user:runner-url-secret@example.test/rfm',
            'cannot contain credentials',
        ),
        (
            'https://example.test/rfm?token=runner-url-secret',
            'cannot contain a query or fragment',
        ),
    ],
)
def test_live_runner_rejects_sensitive_urls_without_logging_them(
    base_url: str,
    expected_error: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env['RFM_NIM_PYTHON'] = sys.executable

    result = subprocess.run(
        [repo_root / 'scripts/run_rfm_nim_live_tests.sh', '--url', base_url],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 2
    assert expected_error in output
    assert 'runner-url-secret' not in output


def test_live_client_applies_transport_configuration(mock_api) -> None:
    mock_api.get('https://example.test/rfm/v1/health/ready', json={})
    client = LiveNimClient(
        'https://example.test/rfm/',
        timeout_seconds=7,
        verify_ssl=False,
        api_key='test-key',
    )

    response = client.request('GET', '/v1/health/ready')

    assert response.status_code == 200
    request = mock_api.request_history[-1]
    assert request.headers['X-API-Key'] == 'test-key'


def test_assert_problem_details_checks_contract_fields() -> None:
    response = _response(
        422,
        'application/problem+json',
        {
            'type': '/problems/validation-failed',
            'status': 422,
            'title': 'Validation failed',
            'detail': 'Request data is invalid.',
            'implementation_extension': {'allowed': True},
        },
    )

    body = assert_problem_details(response)

    assert body['implementation_extension'] == {'allowed': True}


def test_assert_prediction_response_correlates_rows_semantically() -> None:
    payload = nim_v1_two_predict_rows_payload()
    response = _response(
        200,
        'application/json',
        {
            'id': 'pred-test',
            'model': 'kumo-rfm',
            'predictions': [
                {
                    'id': '602',
                    'row_index': 1,
                    'prediction': False,
                    'probabilities': {'False': 0.8, 'True': 0.2},
                },
                {
                    'id': '601',
                    'row_index': 0,
                    'prediction': True,
                    'probabilities': {'False': 0.1, 'True': 0.9},
                },
            ],
            'metadata': {'implementation_detail': 'not-pinned'},
        },
    )

    body = assert_prediction_response(response, payload)

    assert len(body['predictions']) == 2


def test_assert_prediction_response_rejects_wrong_row_identity() -> None:
    payload = nim_v1_two_predict_rows_payload()
    response = _response(
        200,
        'application/json',
        {
            'id': 'pred-test',
            'model': 'kumo-rfm',
            'predictions': [
                {
                    'id': '602',
                    'row_index': 0,
                    'prediction': False,
                    'probabilities': {'False': 1.0},
                },
                {
                    'id': '601',
                    'row_index': 1,
                    'prediction': True,
                    'probabilities': {'True': 1.0},
                },
            ],
            'metadata': {},
        },
    )

    with pytest.raises(AssertionError):
        assert_prediction_response(response, payload)


@pytest.mark.parametrize(
    'ready_body',
    [
        {'status': 'ready'},
        {'status': 'healthy', 'check': 'ready'},
    ],
)
def test_preflight_accepts_supported_readiness_shapes(
    mock_api,
    ready_body: dict,
) -> None:
    mock_api.get(
        'http://nim.test/v1/health/ready',
        json=ready_body,
        headers={'Content-Type': 'application/json'},
    )
    mock_api.get(
        'http://nim.test/v1/models',
        json={'data': [{'id': 'kumo-rfm'}]},
        headers={'Content-Type': 'application/json'},
    )

    assert_ready_and_model_available(LiveNimClient('http://nim.test'))

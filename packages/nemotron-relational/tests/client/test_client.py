# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemotron_relational.client import RelationalClient


def test_authenticate_accepts_universal_tfm_nim(requests_mock):
    base_url = 'https://nim.test'
    requests_mock.get(
        f'{base_url}/v1/health/ready',
        json={
            'status': 'healthy',
            'check': 'ready',
        },
    )
    requests_mock.get(
        f'{base_url}/v1/models',
        json={
            'object': 'list',
            'data': [
                {
                    'id': 'nemotron-relational-v1',
                    'object': 'model',
                }
            ],
        },
    )

    RelationalClient(base_url).authenticate()


@pytest.mark.parametrize(
    'ready_payload',
    [
        {
            'status': 'unhealthy',
            'check': 'ready',
        },
        {
            'status': 'healthy',
            'check': 'live',
        },
        ['ready'],
    ],
)
def test_authenticate_rejects_unready_universal_tfm_nim(
    requests_mock,
    ready_payload,
):
    base_url = 'https://nim.test'
    requests_mock.get(
        f'{base_url}/v1/health/ready',
        json=ready_payload,
    )

    with pytest.raises(ValueError):
        RelationalClient(base_url).authenticate()

    assert requests_mock.call_count == 1


@pytest.mark.parametrize('status_code', [401, 403])
def test_authenticate_surfaces_gateway_auth_error(requests_mock, status_code):
    base_url = 'https://nim.test'
    requests_mock.get(f'{base_url}/v1/health/ready', json={'status': 'ready'})
    requests_mock.get(f'{base_url}/v1/models', status_code=status_code)

    with pytest.raises(ValueError, match='authentication failed'):
        RelationalClient(base_url, api_key='wrong-key').authenticate()

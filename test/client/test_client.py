import pytest

from kumoai.client import KumoClient


def test_authenticate_accepts_universal_tfm_nim(requests_mock):
    base_url = 'http://nim.test'
    requests_mock.get(f'{base_url}/v1/connectors', status_code=404)
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
            'data': [{
                'id': 'kumo-rfm',
                'object': 'model',
            }],
        },
    )

    KumoClient(base_url, api_key='test:DISABLED').authenticate()


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
    base_url = 'http://nim.test'
    requests_mock.get(f'{base_url}/v1/connectors', status_code=404)
    requests_mock.get(
        f'{base_url}/v1/health/ready',
        json=ready_payload,
    )

    with pytest.raises(ValueError):
        KumoClient(base_url, api_key='test:DISABLED').authenticate()

    assert requests_mock.call_count == 2

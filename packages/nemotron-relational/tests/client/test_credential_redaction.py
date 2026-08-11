# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import nemotron_relational
import pytest
from nemotron_relational.client.client import (
    RelationalClient,
    redact_url,
    scrub_userinfo,
)

_SECRET = 'sk-not-a-real-key-0123456789'


@pytest.fixture(autouse=True)
def _restore_global_state():
    before = (
        nemotron_relational.global_state._url,
        nemotron_relational.global_state._api_key,
    )
    yield
    (
        nemotron_relational.global_state._url,
        nemotron_relational.global_state._api_key,
    ) = before


def test_global_state_repr_does_not_render_the_api_key() -> None:
    nemotron_relational.global_state._api_key = _SECRET

    assert _SECRET not in repr(nemotron_relational.global_state)


def test_global_state_repr_still_names_the_deployment() -> None:
    nemotron_relational.global_state._url = 'https://nim.example'

    assert 'https://nim.example' in repr(nemotron_relational.global_state)


def test_api_key_is_still_readable_through_the_attribute() -> None:
    nemotron_relational.global_state._api_key = _SECRET

    assert nemotron_relational.global_state._api_key == _SECRET


@pytest.mark.parametrize(
    ('url', 'expected'),
    [
        ('https://user:tok@host/v1', 'https://host/v1'),
        ('https://tok@host/v1', 'https://host/v1'),
        ('https://user:tok@host:8443/v1', 'https://host:8443/v1'),
        ('http://user:tok@[::1]:8000/v1', 'http://[::1]:8000/v1'),
        ('https://host/v1', 'https://host/v1'),
        ('http://localhost:8000', 'http://localhost:8000'),
        (None, None),
        ('', ''),
    ],
)
def test_redact_url_strips_userinfo_and_leaves_everything_else(
    url: str | None,
    expected: str | None,
) -> None:
    assert redact_url(url) == expected


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        (
            "HTTPSConnectionPool(host='https://u:s@h/x')",
            "HTTPSConnectionPool(host='https://h/x')",
        ),
        ('no url here', 'no url here'),
        ('https://h/x', 'https://h/x'),
    ],
)
def test_scrub_userinfo_strips_credentials_from_free_text(
    text: str,
    expected: str,
) -> None:
    assert scrub_userinfo(text) == expected


def test_connection_error_message_does_not_leak_url_userinfo() -> None:
    client = RelationalClient(
        f'https://svc:{_SECRET}@127.0.0.1:1', timeout=0.2, max_retries=0
    )

    with pytest.raises(ValueError) as caught:
        client.authenticate()

    assert _SECRET not in str(caught.value)


def test_invalid_url_message_does_not_leak_url_userinfo() -> None:
    with pytest.raises(ValueError) as caught:
        RelationalClient(f'ftp://svc:{_SECRET}@nim.example')

    assert _SECRET not in str(caught.value)


def test_not_ready_message_does_not_leak_url_userinfo(requests_mock) -> None:
    url = f'https://svc:{_SECRET}@nim.example'
    requests_mock.get(
        f'{url}/v1/health/ready', status_code=200, json={'status': 'nope'}
    )
    client = RelationalClient(url)

    with pytest.raises(ValueError) as caught:
        client.authenticate()

    assert _SECRET not in str(caught.value)


def test_init_log_line_does_not_leak_url_userinfo(
    caplog: pytest.LogCaptureFixture,
    requests_mock,
) -> None:
    url = f'https://svc:{_SECRET}@nim.example'
    requests_mock.get(
        f'{url}/v1/health/ready', status_code=200, json={'status': 'ready'}
    )
    requests_mock.get(
        f'{url}/v1/models',
        status_code=200,
        json={'data': [{'id': 'nemotron-relational-v1'}]},
    )

    with caplog.at_level(logging.INFO, logger='nemotron_relational'):
        nemotron_relational.init(url=url, api_key=_SECRET)

    logged = '\n'.join(record.getMessage() for record in caplog.records)
    assert _SECRET not in logged
    assert 'nim.example' in logged

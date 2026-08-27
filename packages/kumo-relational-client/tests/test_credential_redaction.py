# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import requests
import requests_mock

from kumo_relational_client import RelationalClient, RelationalError
from kumo_relational_client.core.transport import (
    Transport,
    redact_url,
    scrub_userinfo,
)

_SECRET = 'sk-not-a-real-key-0123456789'
_URL = f'https://user:{_SECRET}@nim.example'


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
def test_redact_url_strips_userinfo_and_leaves_everything_else(url, expected):
    assert redact_url(url) == expected


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('failed for https://user:tok@host/v1', 'failed for https://host/v1'),
        ('nothing to strip here', 'nothing to strip here'),
        (
            'two https://a:b@h1/ and https://c:d@h2/',
            'two https://h1/ and https://h2/',
        ),
    ],
)
def test_scrub_userinfo_removes_the_credential_from_free_text(text, expected):
    assert scrub_userinfo(text) == expected


def test_the_url_stays_readable_through_the_property():
    # Redaction covers what is rendered, not what is stored: the KumoRelational
    # adapter hands this URL to the driver and the credential has to survive.
    assert Transport(_URL).url == _URL


def test_client_repr_does_not_render_a_credential_carried_in_the_url():
    with RelationalClient(_URL) as client:
        assert _SECRET not in repr(client)
        assert 'nim.example' in repr(client)


def test_transport_errors_do_not_render_a_credential_carried_in_the_url():
    transport = Transport(_URL, timeout=1.0, max_retries=0)
    with requests_mock.Mocker() as mock:
        mock.get(
            f'{_URL}/v1/health/ready',
            # `requests` quotes the requested URL back in its own message, so
            # the credential reaches the caller through the driver exception
            # as well as through the URL this layer names.
            exc=requests.ConnectTimeout(
                f'timed out for {_URL}/v1/health/ready'
            ),
        )
        with pytest.raises(RelationalError) as caught:
            transport.health_ready()
    assert _SECRET not in str(caught.value)
    assert 'nim.example' in str(caught.value)


def test_an_unusable_url_is_reported_without_the_credential():
    with pytest.raises(RelationalError) as caught:
        Transport(f'ftp://user:{_SECRET}@host')
    assert _SECRET not in str(caught.value)

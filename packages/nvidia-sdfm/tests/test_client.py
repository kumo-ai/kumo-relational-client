# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from nvidia_sdfm import SDFMClient, ModelCapabilities


def test_client_lists_builtin_models():
    assert set(SDFMClient(url='http://nim.test').models()) == {'kumo-rfm', 'tabicl'}


def test_client_reports_capabilities():
    client = SDFMClient(url='http://nim.test')
    caps = client.capabilities('tabicl')
    assert isinstance(caps, ModelCapabilities)
    assert caps.model == 'tabicl'
    assert 'classification' in caps.tasks
    assert 'prediction' in caps.outputs
    assert caps.request_type == 'TabICLRequest'


def test_two_clients_have_independent_transports():
    a = SDFMClient(url='http://a.test')
    b = SDFMClient(url='http://b.test')
    assert a.url == 'http://a.test'
    assert b.url == 'http://b.test'
    assert a._transport is not b._transport
    assert a._registry is not b._registry


def test_context_manager_closes_transport():
    with SDFMClient(url='http://nim.test') as client:
        session = client._transport._session
    assert session is not None


def test_capabilities_unknown_model_raises():
    from nvidia_sdfm.errors import UnknownModelError

    with pytest.raises(UnknownModelError):
        SDFMClient(url='http://nim.test').capabilities('nope')

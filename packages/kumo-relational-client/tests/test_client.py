# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from kumo_relational_client import ModelCapabilities, RelationalClient


def test_client_lists_builtin_models():
    assert set(RelationalClient(url='http://nim.test').models()) == {
        'kumo-relational',
    }


def test_client_reports_capabilities():
    client = RelationalClient(url='http://nim.test')
    caps = client.capabilities('kumo-relational')
    assert isinstance(caps, ModelCapabilities)
    assert caps.model == 'kumo-relational'
    assert 'regression' in caps.tasks
    assert 'prediction' in caps.outputs
    assert caps.request_type == (
        'KumoRelationalRequest | KumoRelationalTaskRequest'
    )


def test_two_clients_have_independent_transports():
    a = RelationalClient(url='http://a.test')
    b = RelationalClient(url='http://b.test')
    assert a.url == 'http://a.test'
    assert b.url == 'http://b.test'
    assert a._transport is not b._transport
    assert a._registry is not b._registry


def test_context_manager_closes_transport():
    with RelationalClient(url='http://nim.test') as client:
        session = client._transport._session
    assert session is not None


def test_capabilities_unknown_model_raises():
    from kumo_relational_client.errors import UnknownModelError

    with pytest.raises(UnknownModelError):
        RelationalClient(url='http://nim.test').capabilities('nope')

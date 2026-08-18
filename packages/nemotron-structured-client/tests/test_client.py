# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from nemotron_structured import ModelCapabilities, StructuredClient


def test_client_lists_builtin_models():
    assert set(StructuredClient(url='http://nim.test').models()) == {
        'nemotron-relational',
        'nemotron-tabular',
    }


def test_client_reports_capabilities():
    client = StructuredClient(url='http://nim.test')
    caps = client.capabilities('nemotron-tabular')
    assert isinstance(caps, ModelCapabilities)
    assert caps.model == 'nemotron-tabular'
    assert 'classification' in caps.tasks
    assert 'prediction' in caps.outputs
    assert caps.request_type == 'NemotronTabularRequest'


def test_two_clients_have_independent_transports():
    a = StructuredClient(url='http://a.test')
    b = StructuredClient(url='http://b.test')
    assert a.url == 'http://a.test'
    assert b.url == 'http://b.test'
    assert a._transport is not b._transport
    assert a._registry is not b._registry


def test_context_manager_closes_transport():
    with StructuredClient(url='http://nim.test') as client:
        session = client._transport._session
    assert session is not None


def test_capabilities_unknown_model_raises():
    from nemotron_structured.errors import UnknownModelError

    with pytest.raises(UnknownModelError):
        StructuredClient(url='http://nim.test').capabilities('nope')

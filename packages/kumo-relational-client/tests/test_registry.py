# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from kumo_relational_client.base import (
    AdapterRegistry,
    ModelAdapter,
    ModelCapabilities,
)
from kumo_relational_client.errors import UnknownModelError
from kumo_relational_client.requests import ModelRequest


@dataclass
class _StubRequest(ModelRequest):
    model: ClassVar[str] = 'stub'


class _StubAdapter(ModelAdapter):
    name = 'stub'
    request_type = _StubRequest

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(model='stub', request_type='_StubRequest')

    def predict(self, transport, request):
        return 'ok'


def test_register_and_get_round_trips():
    registry = AdapterRegistry()
    adapter = _StubAdapter()
    registry.register(adapter)
    assert registry.get('stub') is adapter


def test_names_sorted():
    registry = AdapterRegistry()
    registry.register(_StubAdapter())
    assert registry.names() == ['stub']


def test_get_unknown_model_raises():
    registry = AdapterRegistry()
    registry.register(_StubAdapter())
    with pytest.raises(UnknownModelError) as excinfo:
        registry.get('does-not-exist')
    assert excinfo.value.model == 'does-not-exist'
    assert excinfo.value.details['known_models'] == ['stub']


def test_default_client_has_builtin_models():
    from kumo_relational_client import RelationalClient

    assert set(RelationalClient(url='http://nim.test').models()) == {
        'kumo-relational',
    }

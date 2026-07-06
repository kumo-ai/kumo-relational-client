from __future__ import annotations

import pytest

from nvidia_sdfm.base import AdapterRegistry, ModelAdapter
from nvidia_sdfm.errors import UnknownModelError


class _StubAdapter(ModelAdapter):
    name = 'stub'

    def predict(self, client, **kwargs):
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


def test_default_registry_has_tabicl_and_rfm():
    import nvidia_sdfm

    assert set(nvidia_sdfm.registry.names()) == {'kumo-rfm', 'tabicl'}

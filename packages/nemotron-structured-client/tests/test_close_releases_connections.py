# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What ``StructuredClient.close()`` has to release.

The client owns one pool, its transport. A model backed by a driver does not
send through that pool: the Nemotron Relational driver opens its own, so closing the
transport alone left it open for the life of the process. These cover the
release path that closes both, and the constraint that makes it safe to
release a pool another client may still be using.
"""

from __future__ import annotations

import pytest

from nemotron_structured import StructuredClient
from nemotron_structured.base import (
    AdapterRegistry,
    ModelAdapter,
    ModelCapabilities,
)


class _Recording(ModelAdapter):
    r"""An adapter that records its close, and optionally fails it."""

    request_type = ()

    def __init__(self, name: str, error: Exception | None = None) -> None:
        self.name = name
        self.closed = 0
        self._error = error

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model=self.name, request_type='none', tasks=(), outputs=()
        )

    def predict(self, transport, request):
        raise NotImplementedError

    def close(self) -> None:
        self.closed += 1
        if self._error is not None:
            raise self._error


def test_default_adapter_close_does_nothing() -> None:
    r"""An adapter with no pool of its own must not have to implement this."""

    class Bare(_Recording):
        pass

    adapter = Bare('bare')
    ModelAdapter.close(adapter)


def test_closing_a_client_closes_its_adapters() -> None:
    client = StructuredClient(url='http://nim.test')
    adapter = _Recording('recording')
    client._register(adapter)

    client.close()

    assert adapter.closed == 1


def test_closing_a_client_still_closes_the_transport_if_an_adapter_fails() -> (
    None
):
    r"""The transport is the pool that cannot be rebuilt, so it wins."""
    client = StructuredClient(url='http://nim.test')
    client._register(_Recording('broken', error=RuntimeError('no')))

    with pytest.raises(RuntimeError, match='no'):
        client.close()

    assert client._transport._closed


def test_one_failing_adapter_does_not_strand_the_others() -> None:
    registry = AdapterRegistry()
    first = _Recording('a', error=RuntimeError('first'))
    second = _Recording('b')
    registry.register(first)
    registry.register(second)

    with pytest.raises(RuntimeError, match='first'):
        registry.close()

    assert second.closed == 1


def test_two_clients_close_independently() -> None:
    a, b = (
        StructuredClient(url='http://a.test'),
        StructuredClient(url='http://b.test'),
    )
    a_adapter, b_adapter = _Recording('x'), _Recording('x')
    a._register(a_adapter)
    b._register(b_adapter)

    a.close()

    assert (a_adapter.closed, b_adapter.closed) == (1, 0)
    assert not b._transport._closed


def test_nemotron_relational_adapter_close_is_inert_before_any_prediction() -> (
    None
):
    r"""Closing a client that only used Nemotron Tabular must not import the driver,
    which is an optional dependency and may not be installed.
    """
    from nemotron_structured.adapters.relational import (
        NemotronRelationalAdapter,
    )

    adapter = NemotronRelationalAdapter()
    assert not adapter._opened_engine_client
    adapter.close()


def test_nemotron_relational_adapter_releases_the_engine_it_opened(
    monkeypatch,
) -> None:
    from nemotron_structured.adapters import relational as adapter_module

    released: list[object] = []
    token = object()

    class _Engine:
        _CLIENT_TOKEN = token

        @staticmethod
        def close_client(_token: object | None = None) -> None:
            released.append(_token)

    monkeypatch.setattr(adapter_module, '_load_engine', lambda: _Engine)

    adapter = adapter_module.NemotronRelationalAdapter()
    adapter._opened_engine_client = True
    adapter.close()

    assert released == [token]


def test_nemotron_relational_adapter_does_not_release_twice(
    monkeypatch,
) -> None:
    r"""A second close has nothing left to release; releasing again would
    close a pool another client rebuilt in between.
    """
    from nemotron_structured.adapters import relational as adapter_module

    released: list[object] = []

    class _Engine:
        _CLIENT_TOKEN = object()

        @staticmethod
        def close_client(_token: object | None = None) -> None:
            released.append(_token)

    monkeypatch.setattr(adapter_module, '_load_engine', lambda: _Engine)

    adapter = adapter_module.NemotronRelationalAdapter()
    adapter._opened_engine_client = True
    adapter.close()
    adapter.close()

    assert len(released) == 1

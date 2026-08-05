# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initializing against something other than a base URL.

The RFM path reads its client from ``GlobalState.client``. That used to be
hardwired to build a ``KumoClient`` from ``_url``, so a deployment addressed by
name rather than by URL was unreachable. These tests cover the seam and, more
importantly, that the raw-NIM path is unchanged by it.
"""
from __future__ import annotations

import threading
from typing import Any

import pytest

import kumorfm
from kumorfm.client.client import KumoClient
from kumorfm.client.databricks_serving import DatabricksServingClient


class _Endpoints:
    @staticmethod
    def query(**kwargs: Any) -> Any:
        return {"predictions": [{"response_json": "{}"}]}


class _Workspace:
    serving_endpoints = _Endpoints()


@pytest.fixture(autouse=True)
def _clean_state() -> Any:
    kumorfm.global_state.clear()
    yield
    kumorfm.global_state.clear()


# -- the new mode ----------------------------------------------------------

def test_initializes_without_a_url() -> None:
    kumorfm.init_databricks_serving("kumo-rfm", workspace_client=_Workspace())
    assert kumorfm.global_state.initialized
    assert kumorfm.global_state._url is None
    assert isinstance(kumorfm.global_state.client, DatabricksServingClient)


def test_a_bad_endpoint_leaves_state_untouched() -> None:
    """Validate before mutating: a rejected endpoint name must not leave the
    process half-initialized.
    """
    with pytest.raises(ValueError):
        kumorfm.init_databricks_serving(
            "https://workspace/serving-endpoints/kumo-rfm",
            workspace_client=_Workspace(),
        )
    assert not kumorfm.global_state.initialized


def test_repeated_init_does_not_rebuild_the_client() -> None:
    """Building the transport resolves Databricks credentials, and the SDFM
    adapter re-initializes on every predict. An unchanged re-init must be a
    no-op rather than a fresh authentication per scored partition.
    """
    workspace = _Workspace()
    kumorfm.init_databricks_serving("kumo-rfm", workspace_client=workspace)
    first_client = kumorfm.global_state.client

    kumorfm.init_databricks_serving("kumo-rfm", workspace_client=workspace)

    assert kumorfm.global_state.client is first_client


def test_changed_arguments_still_rebuild_the_client() -> None:
    """The fast path must key on the arguments, not merely on being
    initialized, or a caller could never switch endpoint or workspace.
    """
    kumorfm.init_databricks_serving("kumo-rfm", workspace_client=_Workspace())
    first_client = kumorfm.global_state.client

    kumorfm.init_databricks_serving("kumo-rfm-2",
                                    workspace_client=_Workspace())
    assert kumorfm.global_state.client is not first_client
    assert kumorfm.global_state.client.endpoint == "kumo-rfm-2"


def test_transport_overrides_reach_the_client() -> None:
    """The size cap and timeout are constructor arguments on the transport;
    without plumbing them through init no caller can reach them.
    """
    kumorfm.init_databricks_serving("kumo-rfm", workspace_client=_Workspace(),
                                    max_request_bytes=1234, timeout=12.5)
    client = kumorfm.global_state.client
    assert client._max_request_bytes == 1234
    assert client._timeout == 12.5


def test_clear_resets_the_factory() -> None:
    kumorfm.init_databricks_serving("kumo-rfm", workspace_client=_Workspace())
    kumorfm.global_state.clear()
    assert not kumorfm.global_state.initialized
    assert kumorfm.global_state._client_factory is None


def test_each_thread_gets_its_own_client() -> None:
    """The cache is per thread, so an existing KumoRFM keeps the client it was
    built with.
    """
    kumorfm.init_databricks_serving("kumo-rfm", workspace_client=_Workspace())
    main = kumorfm.global_state.client
    seen: list[Any] = []

    thread = threading.Thread(target=lambda: seen.append(kumorfm.global_state.client))
    thread.start()
    thread.join()

    assert isinstance(seen[0], DatabricksServingClient)
    assert seen[0] is not main
    assert kumorfm.global_state.client is main


# -- the raw-NIM path is unchanged ----------------------------------------

def test_url_mode_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A URL still builds an authenticated KumoClient, and is still required.

    ``authenticate()`` probes the NIM; serving mode has no equivalent, but the
    URL path must not have quietly lost it.
    """
    called: list[bool] = []
    monkeypatch.setattr(KumoClient, "authenticate",
                        lambda self: called.append(True))

    kumorfm.init(url="http://nim.test")

    assert isinstance(kumorfm.global_state.client, KumoClient)
    assert kumorfm.global_state._url == "http://nim.test"
    assert kumorfm.global_state._client_factory is None
    assert called == [True]

    kumorfm.global_state.clear()
    monkeypatch.delenv("KUMO_API_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="no endpoint"):
        kumorfm.init()


def test_switching_modes_replaces_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-initializing must not leave a stale client of the previous kind."""
    monkeypatch.setattr(KumoClient, "authenticate", lambda self: None)
    kumorfm.init(url="http://nim.test")
    assert isinstance(kumorfm.global_state.client, KumoClient)

    kumorfm.init_databricks_serving("kumo-rfm", workspace_client=_Workspace())
    assert isinstance(kumorfm.global_state.client, DatabricksServingClient)
    assert kumorfm.global_state._url is None


def test_uninitialized_state_still_raises() -> None:
    with pytest.raises(ValueError, match="Client creation"):
        _ = kumorfm.global_state.client

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The adapter's serving-mode branch, without the native relational driver.

``test_kumo_relational_engine_adapter.py`` skips everything when ``kumo_relational_engine.rfm`` is unusable,
which on a machine without the compiled ``relationallib`` is everything. That is the
repo's existing convention and it left the branch added for Databricks Model
Serving with no coverage at all.

``KumoRelationalAdapter.predict`` imports the engine lazily, inside the method, so a
stub module registered in ``sys.modules`` is enough to exercise the branch --
no compiled extension, no Databricks SDK, no network.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any

import pandas as pd
import pytest

from kumo_relational_client.adapters.relational import KumoRelationalAdapter
from kumo_relational_client.core.serving import DatabricksServingTarget
from kumo_relational_client.core.transport import Transport
from kumo_relational_client.errors import RelationalError
from kumo_relational_client.requests import KumoRelationalRequest

_CLIENT_TOKEN = object()

# Everything here runs against a stub. The one exception is the cross-layer
# agreement test, which has to import kumo_relational_engine's real validator -- a stub cannot
# stand in for the thing being compared. It runs in integration_tests, which
# installs the engine; client_tests does not.
requires_kumo_relational_engine = pytest.mark.skipif(
    importlib.util.find_spec('kumo_relational_engine') is None,
    reason='kumo_relational_engine is not installed; the cross-layer check needs the real one',
)


@pytest.fixture()
def engine(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """A stand-in for kumo_relational_engine.rfm that records how it was initialized."""
    calls: dict[str, Any] = {}

    class _KumoRelational:
        def __init__(self, graph: Any, **kwargs: Any) -> None:
            calls['graph'] = graph

        def predict(self, query: str, **kwargs: Any) -> pd.DataFrame:
            calls['predict'] = {'query': query, **kwargs}
            return pd.DataFrame({'prediction': [1]})

        def batch_mode(self, *args: Any, **kwargs: Any) -> Any:
            import contextlib

            return contextlib.nullcontext()

        def retry(self, *args: Any, **kwargs: Any) -> Any:
            # The adapter calls this when num_retries is set without a batch
            # size. A double that omits it passes only until the adapter starts
            # using it, which is how this file broke.
            import contextlib

            calls['retry'] = (args, kwargs)
            return contextlib.nullcontext()

    # The adapter resolves HTTPException out of sys.modules at call time, so a
    # stub carrying the real signature is enough. Importing the real
    # kumo_relational_engine.exceptions instead would cost this file the property its
    # docstring claims -- and the client_tests job, which installs no engine,
    # is the only place the branch gets covered.
    exceptions = types.ModuleType('kumo_relational_engine.exceptions')

    class _HTTPError(Exception):
        def __init__(
            self,
            status_code: int,
            detail: str | None = None,
            headers: dict[str, str] | None = None,
        ) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail
            self.headers = headers

    exceptions.HTTPException = _HTTPError

    module = types.ModuleType('kumo_relational_engine.rfm')
    module.init = lambda **kw: calls.setdefault('init', kw)
    module.init_client = lambda **kw: calls.setdefault('init', kw)
    module.init_databricks_serving = lambda endpoint, **kw: calls.setdefault(
        'init_databricks_serving', {'endpoint': endpoint, **kw}
    )
    module.KumoRelational = _KumoRelational
    module._CLIENT_TOKEN = _CLIENT_TOKEN

    parent = types.ModuleType('kumo_relational_engine')
    parent.rfm = module
    parent.exceptions = exceptions

    monkeypatch.setitem(sys.modules, 'kumo_relational_engine', parent)
    monkeypatch.setitem(sys.modules, 'kumo_relational_engine.rfm', module)
    monkeypatch.setitem(
        sys.modules, 'kumo_relational_engine.exceptions', exceptions
    )
    module.calls = calls
    return module


def _request() -> KumoRelationalRequest:
    return KumoRelationalRequest(
        graph=object(), query='PREDICT x FOR y', indices=[1]
    )


def test_serving_target_initializes_by_endpoint_name(
    engine: types.SimpleNamespace,
) -> None:
    """The branch under test: a ServingTarget must not be sent through the
    URL-based init, which would read attributes that raise by design."""
    target = DatabricksServingTarget('kumo-relational', platform_client='WS')
    KumoRelationalAdapter().predict(target, _request())

    assert 'init_databricks_serving' in engine.calls
    assert (
        engine.calls['init_databricks_serving']['endpoint'] == 'kumo-relational'
    )
    assert engine.calls['init_databricks_serving']['workspace_client'] == 'WS'
    assert 'init' not in engine.calls, (
        'took the raw-NIM path for a serving target'
    )


def test_serving_target_never_reads_url_or_api_key(
    engine: types.SimpleNamespace,
) -> None:
    """ServingTarget.url raises. If the adapter duck-typed instead of checking
    the type, this is where it would blow up."""
    KumoRelationalAdapter().predict(
        DatabricksServingTarget('kumo-relational'), _request()
    )
    sent = engine.calls['init_databricks_serving']
    assert (
        'url' not in sent and 'api_key' not in sent and 'verify_ssl' not in sent
    )


def test_transport_still_takes_the_url_path(
    engine: types.SimpleNamespace,
) -> None:
    """The raw-NIM path must be unchanged by the branch."""
    transport = Transport('https://nim.example.com:8000', api_key='secret')
    KumoRelationalAdapter().predict(transport, _request())

    assert 'init' in engine.calls
    assert engine.calls['init']['url'] == 'https://nim.example.com:8000'
    assert engine.calls['init']['api_key'] == 'secret'
    assert 'init_databricks_serving' not in engine.calls


def test_both_paths_reach_the_same_predict(
    engine: types.SimpleNamespace,
) -> None:
    """Only initialization differs; inference must be identical."""
    KumoRelationalAdapter().predict(
        DatabricksServingTarget('kumo-relational'), _request()
    )
    serving = engine.calls['predict']

    engine.calls.clear()
    KumoRelationalAdapter().predict(
        Transport('https://nim.example.com:8000'), _request()
    )
    assert engine.calls['predict'] == serving


# -- the documented happy path --------------------------------------------


def test_a_serving_client_closes() -> None:
    """The class docstring offers close() as the alternative to the context
    manager. It did not work: ServingTarget had no close()."""
    from kumo_relational_client import RelationalClient

    client = RelationalClient.for_databricks_serving('kumo-relational')
    client.close()


def test_a_serving_client_works_as_a_context_manager() -> None:
    from kumo_relational_client import RelationalClient

    with RelationalClient.for_databricks_serving('kumo-relational') as client:
        assert 'kumo-relational' in client.models()


def test_a_serving_client_reprs() -> None:
    """__repr__ went through .url, which raises for a serving client -- so
    printing one, or letting a debugger or a nested traceback render it, blew
    up in exactly the places you most want it to work."""
    from kumo_relational_client import RelationalClient

    text = repr(RelationalClient.for_databricks_serving('kumo-relational'))
    assert 'kumo-relational' in text
    assert str(RelationalClient.for_databricks_serving('kumo-relational'))


def test_a_url_client_still_reprs_with_its_url() -> None:
    from kumo_relational_client import RelationalClient

    assert 'https://nim.example.com:8000' in repr(
        RelationalClient('https://nim.example.com:8000')
    )


def test_the_repr_does_not_render_the_workspace_client() -> None:
    """A real WorkspaceClient renders its workspace host; an injected one
    renders whatever it likes, including credentials."""
    from kumo_relational_client import RelationalClient
    from kumo_relational_client.core.serving import DatabricksServingTarget

    class _Leaky:
        def __repr__(self) -> str:
            return "WorkspaceClient(token='dapi-SECRET', host='acme.databricks.com')"

    target = DatabricksServingTarget('kumo-relational', _Leaky())
    assert 'dapi-SECRET' not in repr(target)
    assert 'acme.databricks.com' not in repr(target)
    assert 'dapi-SECRET' not in repr(
        RelationalClient.for_databricks_serving(
            'kumo-relational', workspace_client=_Leaky()
        )
    )


def test_both_construction_paths_populate_the_same_fields() -> None:
    """for_databricks_serving does not run __init__, so its client is only as
    complete as whoever last remembered to update both paths. Adding a field to
    one and not the other is an AttributeError at first use, not here."""
    from kumo_relational_client import RelationalClient

    by_url = RelationalClient('https://nim.example.com:8000')
    by_endpoint = RelationalClient.for_databricks_serving('kumo-relational')
    assert vars(by_url).keys() == vars(by_endpoint).keys()


# -- what the target refuses to do ----------------------------------------


@pytest.mark.parametrize(
    ('refuse', 'code'),
    [
        pytest.param(lambda c: c.url, 'INVALID_CONFIGURATION', id='url'),
        pytest.param(
            lambda c: c.health_ready(), 'UNSUPPORTED_FEATURE', id='health_ready'
        ),
    ],
)
def test_a_serving_target_refuses_what_does_not_apply(refuse, code) -> None:
    """A serving endpoint has no URL and no readiness route. Each member
    exists only to say so, naming the endpoint, rather than returning a
    plausible-looking value."""
    from kumo_relational_client import RelationalClient

    with pytest.raises(RelationalError) as caught:
        refuse(RelationalClient.for_databricks_serving('kumo-relational'))
    assert caught.value.code == code
    assert 'kumo-relational' in caught.value.message


# -- the two endpoint-validation layers -----------------------------------

_BAD_ENDPOINTS = [
    pytest.param('', id='empty'),
    pytest.param('   ', id='whitespace-only'),
    pytest.param(
        'https://user:PWSECRET@acme.cloud.databricks.com/serving',
        id='userinfo-credentials',
    ),
    pytest.param('acme.databricks.com/serving?token=dapiTOK', id='token-query'),
    pytest.param('relational-endpoint\n', id='trailing-newline'),
    pytest.param('kumo#rfm', id='fragment'),
]


@requires_kumo_relational_engine
@pytest.mark.parametrize('endpoint', _BAD_ENDPOINTS)
def test_the_two_endpoint_validation_layers_agree(endpoint: str) -> None:
    """ServingTarget and kumo_relational_engine's DatabricksServingClient validate the same
    endpoint independently, with the same predicates in the same order. Only
    the raised type differs, so tightening one and not the other is a silent
    divergence -- this pins the messages themselves as equal.

    The rejected value is never echoed by either: a pasted workspace URL is
    the one input guaranteed to be able to carry credentials.
    """
    from kumo_relational_engine.client.databricks_serving import (
        DatabricksServingClient,
    )

    from kumo_relational_client import RelationalClient

    with pytest.raises(RelationalError) as ours:
        RelationalClient.for_databricks_serving(endpoint)
    with pytest.raises(ValueError) as theirs:
        DatabricksServingClient(endpoint)

    assert ours.value.code == 'INVALID_CONFIGURATION'
    assert ours.value.message == str(theirs.value)
    for leak in ('PWSECRET', 'dapiTOK', 'acme'):
        assert leak not in ours.value.message
        assert leak not in str(theirs.value)


# -- failures coming back out of the engine -------------------------------


def _missing_sdk() -> Exception:
    """Verbatim what kumo_relational_engine raises when databricks-sdk is absent."""
    return ImportError(
        "Databricks Model Serving support requires the 'databricks-serving' "
        "extra: pip install 'kumo_relational_engine[databricks-serving]'"
    )


def _ambient_auth_failed() -> Exception:
    from kumo_relational_engine.exceptions import HTTPException

    return HTTPException(
        503,
        'could not authenticate to Databricks from the ambient configuration',
    )


@pytest.mark.parametrize(
    ('build', 'code', 'says'),
    [
        pytest.param(
            _missing_sdk,
            'MISSING_EXTRA',
            'pip install kumo-relational-client[databricks-serving]',
            id='missing-databricks-sdk',
        ),
        pytest.param(
            _ambient_auth_failed,
            'SERVING_INIT_FAILED',
            'could not authenticate',
            id='ambient-auth-failed',
        ),
    ],
)
def test_engine_init_failures_are_translated_at_the_boundary(
    engine: types.SimpleNamespace,
    build,
    code,
    says,
) -> None:
    """kumo_relational_engine names its own extra, so an unwrapped ImportError tells someone
    who installed kumo-relational-client[databricks-serving] to install a package they
    never named. Nothing from the engine reaches the caller untranslated."""

    def _raise(endpoint: str, **kwargs: Any) -> None:
        raise build()

    engine.init_databricks_serving = _raise

    with pytest.raises(RelationalError) as caught:
        KumoRelationalAdapter().predict(
            DatabricksServingTarget('kumo-relational'), _request()
        )
    assert caught.value.code == code
    assert says in caught.value.message
    assert 'kumo_relational_engine[' not in caught.value.message


def test_engine_failure_on_the_serving_path_keeps_its_own_message(
    engine: types.SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An engine failure must survive translation on the serving path.

    The translation step used to be handed ``transport.url`` for the message.
    ServingTarget raises from that property by design, and this runs inside an
    ``except`` block, so the raise replaced the very failure being reported:
    every engine error arrived as 'the serving endpoint has no URL' and the
    real cause was discarded. Found while debugging a NIM BAD_REQUEST that the
    client reported as a configuration problem.
    """

    class _Failing:
        def __init__(self, graph: Any, **kwargs: Any) -> None:
            pass

        def retry(self, *args: Any, **kwargs: Any) -> Any:
            import contextlib

            return contextlib.nullcontext()

        def predict(self, query: str, **kwargs: Any) -> pd.DataFrame:
            raise RuntimeError('NIM said: payload too large')

    monkeypatch.setattr(engine, 'KumoRelational', _Failing)

    with pytest.raises(RelationalError) as excinfo:
        KumoRelationalAdapter().predict(
            DatabricksServingTarget('kumo-relational'), _request()
        )

    message = str(excinfo.value)
    assert 'payload too large' in message
    assert 'has no URL' not in message
    assert 'kumo-relational' in message

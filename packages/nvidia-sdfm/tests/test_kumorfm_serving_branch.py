"""The adapter's serving-mode branch, without the native RFM engine.

``test_kumorfm_adapter.py`` skips everything when ``kumorfm.rfm`` is unusable,
which on a machine without the compiled ``kumolib`` is everything. That is the
repo's existing convention and it left the branch added for Databricks Model
Serving with no coverage at all.

``KumoRFMAdapter.predict`` imports the engine lazily, inside the method, so a
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
from nvidia_sdfm.adapters.kumorfm import KumoRFMAdapter
from nvidia_sdfm.core.serving import ServingTarget
from nvidia_sdfm.core.transport import Transport
from nvidia_sdfm.errors import SdfmError
from nvidia_sdfm.requests import KumoRFMRequest

_SDFM_CLIENT_TOKEN = object()

# Everything here runs against a stub. The one exception is the cross-layer
# agreement test, which has to import kumorfm's real validator -- a stub cannot
# stand in for the thing being compared. It runs in integration_tests, which
# installs the engine; client_tests does not.
requires_kumorfm = pytest.mark.skipif(
    importlib.util.find_spec('kumorfm') is None,
    reason='kumorfm is not installed; the cross-layer check needs the real one',
)


@pytest.fixture()
def engine(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """A stand-in for kumorfm.rfm that records how it was initialized."""
    calls: dict[str, Any] = {}

    class _KumoRFM:
        def __init__(self, graph: Any, **kwargs: Any) -> None:
            calls["graph"] = graph

        def predict(self, query: str, **kwargs: Any) -> pd.DataFrame:
            calls["predict"] = {"query": query, **kwargs}
            return pd.DataFrame({"prediction": [1]})

        def batch_mode(self, *args: Any, **kwargs: Any) -> Any:
            import contextlib

            return contextlib.nullcontext()

        def retry(self, *args: Any, **kwargs: Any) -> Any:
            # The adapter calls this when num_retries is set without a batch
            # size. A double that omits it passes only until the adapter starts
            # using it, which is how this file broke.
            import contextlib

            calls["retry"] = (args, kwargs)
            return contextlib.nullcontext()

    # The adapter resolves HTTPException out of sys.modules at call time, so a
    # stub carrying the real signature is enough. Importing the real
    # kumorfm.exceptions instead would cost this file the property its
    # docstring claims -- and the client_tests job, which installs no engine,
    # is the only place the branch gets covered.
    exceptions = types.ModuleType("kumorfm.exceptions")

    class _HTTPException(Exception):
        def __init__(self, status_code: int, detail: str | None = None,
                     headers: dict[str, str] | None = None) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail
            self.headers = headers

    exceptions.HTTPException = _HTTPException

    module = types.ModuleType("kumorfm.rfm")
    module.init = lambda **kw: calls.setdefault("init", kw)
    module.init_client = lambda **kw: calls.setdefault("init", kw)
    module.init_databricks_serving = lambda endpoint, **kw: calls.setdefault(
        "init_databricks_serving", {"endpoint": endpoint, **kw}
    )
    module.KumoRFM = _KumoRFM
    module._SDFM_CLIENT_TOKEN = _SDFM_CLIENT_TOKEN

    parent = types.ModuleType("kumorfm")
    parent.rfm = module
    parent.exceptions = exceptions

    monkeypatch.setitem(sys.modules, "kumorfm", parent)
    monkeypatch.setitem(sys.modules, "kumorfm.rfm", module)
    monkeypatch.setitem(sys.modules, "kumorfm.exceptions", exceptions)
    module.calls = calls
    return module


def _request() -> KumoRFMRequest:
    return KumoRFMRequest(
        graph=object(), query="PREDICT x FOR y", indices=[1]
    )


def test_serving_target_initializes_by_endpoint_name(
    engine: types.SimpleNamespace,
) -> None:
    """The branch under test: a ServingTarget must not be sent through the
    URL-based init, which would read attributes that raise by design."""
    target = ServingTarget("kumo-rfm", workspace_client="WS")
    KumoRFMAdapter().predict(target, _request())

    assert "init_databricks_serving" in engine.calls
    assert engine.calls["init_databricks_serving"]["endpoint"] == "kumo-rfm"
    assert engine.calls["init_databricks_serving"]["workspace_client"] == "WS"
    assert "init" not in engine.calls, "took the raw-NIM path for a serving target"


def test_serving_target_never_reads_url_or_api_key(
    engine: types.SimpleNamespace,
) -> None:
    """ServingTarget.url raises. If the adapter duck-typed instead of checking
    the type, this is where it would blow up."""
    KumoRFMAdapter().predict(ServingTarget("kumo-rfm"), _request())
    sent = engine.calls["init_databricks_serving"]
    assert "url" not in sent and "api_key" not in sent and "verify_ssl" not in sent


def test_transport_still_takes_the_url_path(
    engine: types.SimpleNamespace,
) -> None:
    """The raw-NIM path must be unchanged by the branch."""
    transport = Transport("https://nim.example.com:8000", api_key="secret")
    KumoRFMAdapter().predict(transport, _request())

    assert "init" in engine.calls
    assert engine.calls["init"]["url"] == "https://nim.example.com:8000"
    assert engine.calls["init"]["api_key"] == "secret"
    assert "init_databricks_serving" not in engine.calls


def test_both_paths_reach_the_same_predict(
    engine: types.SimpleNamespace,
) -> None:
    """Only initialization differs; inference must be identical."""
    KumoRFMAdapter().predict(ServingTarget("kumo-rfm"), _request())
    serving = engine.calls["predict"]

    engine.calls.clear()
    KumoRFMAdapter().predict(
        Transport("https://nim.example.com:8000"), _request()
    )
    assert engine.calls["predict"] == serving


# -- the documented happy path --------------------------------------------

def test_a_serving_client_closes() -> None:
    """The class docstring offers close() as the alternative to the context
    manager. It did not work: ServingTarget had no close()."""
    from nvidia_sdfm import SDFMClient

    client = SDFMClient.for_databricks_serving("kumo-rfm")
    client.close()


def test_a_serving_client_works_as_a_context_manager() -> None:
    from nvidia_sdfm import SDFMClient

    with SDFMClient.for_databricks_serving("kumo-rfm") as client:
        assert "kumo-rfm" in client.models()


def test_a_serving_client_reprs() -> None:
    """__repr__ went through .url, which raises for a serving client -- so
    printing one, or letting a debugger or a nested traceback render it, blew
    up in exactly the places you most want it to work."""
    from nvidia_sdfm import SDFMClient

    text = repr(SDFMClient.for_databricks_serving("kumo-rfm"))
    assert "kumo-rfm" in text
    assert str(SDFMClient.for_databricks_serving("kumo-rfm"))


def test_a_url_client_still_reprs_with_its_url() -> None:
    from nvidia_sdfm import SDFMClient

    assert "https://nim.example.com:8000" in repr(
        SDFMClient("https://nim.example.com:8000")
    )


def test_the_repr_does_not_render_the_workspace_client() -> None:
    """A real WorkspaceClient renders its workspace host; an injected one
    renders whatever it likes, including credentials."""
    from nvidia_sdfm import SDFMClient
    from nvidia_sdfm.core.serving import ServingTarget

    class _Leaky:
        def __repr__(self) -> str:
            return "WorkspaceClient(token='dapi-SECRET', host='acme.databricks.com')"

    target = ServingTarget("kumo-rfm", _Leaky())
    assert "dapi-SECRET" not in repr(target)
    assert "acme.databricks.com" not in repr(target)
    assert "dapi-SECRET" not in repr(
        SDFMClient.for_databricks_serving("kumo-rfm", workspace_client=_Leaky())
    )


def test_both_construction_paths_populate_the_same_fields() -> None:
    """for_databricks_serving does not run __init__, so its client is only as
    complete as whoever last remembered to update both paths. Adding a field to
    one and not the other is an AttributeError at first use, not here."""
    from nvidia_sdfm import SDFMClient

    by_url = SDFMClient("https://nim.example.com:8000")
    by_endpoint = SDFMClient.for_databricks_serving("kumo-rfm")
    assert vars(by_url).keys() == vars(by_endpoint).keys()


# -- what the target refuses to do ----------------------------------------

@pytest.mark.parametrize(("refuse", "code"), [
    pytest.param(lambda c: c.url, "INVALID_CONFIGURATION", id="url"),
    pytest.param(lambda c: c.health_ready(), "UNSUPPORTED_FEATURE",
                 id="health_ready"),
    pytest.param(lambda c: ServingTarget("kumo-rfm").predict({}),
                 "UNSUPPORTED_FEATURE", id="predict"),
])
def test_a_serving_target_refuses_what_does_not_apply(refuse, code) -> None:
    """A serving endpoint has no URL, no readiness route, and no generic
    prediction path. Each member exists only to say so, naming the endpoint,
    rather than returning a plausible-looking value."""
    from nvidia_sdfm import SDFMClient

    with pytest.raises(SdfmError) as caught:
        refuse(SDFMClient.for_databricks_serving("kumo-rfm"))
    assert caught.value.code == code
    assert "kumo-rfm" in caught.value.message


# -- the two endpoint-validation layers -----------------------------------

_BAD_ENDPOINTS = [
    pytest.param("", id="empty"),
    pytest.param("   ", id="whitespace-only"),
    pytest.param(
        "https://user:PWSECRET@acme.cloud.databricks.com/serving",
        id="userinfo-credentials",
    ),
    pytest.param("acme.databricks.com/serving?token=dapiTOK", id="token-query"),
    pytest.param("kumo-rfm\n", id="trailing-newline"),
    pytest.param("kumo#rfm", id="fragment"),
]


@requires_kumorfm
@pytest.mark.parametrize("endpoint", _BAD_ENDPOINTS)
def test_the_two_endpoint_validation_layers_agree(endpoint: str) -> None:
    """ServingTarget and kumorfm's DatabricksServingClient validate the same
    endpoint independently, with the same predicates in the same order. Only
    the raised type differs, so tightening one and not the other is a silent
    divergence -- this pins the messages themselves as equal.

    The rejected value is never echoed by either: a pasted workspace URL is
    the one input guaranteed to be able to carry credentials.
    """
    from kumorfm.client.databricks_serving import DatabricksServingClient
    from nvidia_sdfm import SDFMClient

    with pytest.raises(SdfmError) as ours:
        SDFMClient.for_databricks_serving(endpoint)
    with pytest.raises(ValueError) as theirs:
        DatabricksServingClient(endpoint)

    assert ours.value.code == "INVALID_CONFIGURATION"
    assert ours.value.message == str(theirs.value)
    for leak in ("PWSECRET", "dapiTOK", "acme"):
        assert leak not in ours.value.message
        assert leak not in str(theirs.value)


# -- failures coming back out of the engine -------------------------------

def _missing_sdk() -> Exception:
    """Verbatim what kumorfm raises when databricks-sdk is absent."""
    return ImportError(
        "Databricks Model Serving support requires the 'databricks-serving' "
        "extra: pip install 'kumorfm[databricks-serving]'"
    )


def _ambient_auth_failed() -> Exception:
    from kumorfm.exceptions import HTTPException

    return HTTPException(
        503,
        "could not authenticate to Databricks from the ambient configuration",
    )


@pytest.mark.parametrize(("build", "code", "says"), [
    pytest.param(_missing_sdk, "MISSING_EXTRA",
                 "pip install nvidia-sdfm[databricks-serving]",
                 id="missing-databricks-sdk"),
    pytest.param(_ambient_auth_failed, "SERVING_INIT_FAILED",
                 "could not authenticate", id="ambient-auth-failed"),
])
def test_engine_init_failures_are_translated_at_the_boundary(
    engine: types.SimpleNamespace, build, code, says,
) -> None:
    """kumorfm names its own extra, so an unwrapped ImportError tells someone
    who installed nvidia-sdfm[databricks-serving] to install a package they
    never named. Nothing from the engine reaches the caller untranslated."""
    def _raise(endpoint: str, **kwargs: Any) -> None:
        raise build()

    engine.init_databricks_serving = _raise

    with pytest.raises(SdfmError) as caught:
        KumoRFMAdapter().predict(ServingTarget("kumo-rfm"), _request())
    assert caught.value.code == code
    assert says in caught.value.message
    assert "kumorfm[" not in caught.value.message


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
    SDK reported as a configuration problem.
    """
    class _Failing:
        def __init__(self, graph: Any, **kwargs: Any) -> None:
            pass

        def retry(self, *args: Any, **kwargs: Any) -> Any:
            import contextlib

            return contextlib.nullcontext()

        def predict(self, query: str, **kwargs: Any) -> pd.DataFrame:
            raise RuntimeError('NIM said: payload too large')

    monkeypatch.setattr(engine, 'KumoRFM', _Failing)

    with pytest.raises(SdfmError) as excinfo:
        KumoRFMAdapter().predict(ServingTarget('kumo-rfm'), _request())

    message = str(excinfo.value)
    assert 'payload too large' in message
    assert 'has no URL' not in message
    assert 'kumo-rfm' in message

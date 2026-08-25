# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib

import pandas as pd
import pytest

from kumo_relational_client.adapters.relational import NemotronRelationalAdapter
from kumo_relational_client.core.transport import Transport
from kumo_relational_client.errors import (
    MissingExtraError,
    NimRequestError,
    RelationalError,
)
from kumo_relational_client.requests import (
    KumoRelationalRequest,
    KumoRelationalTaskRequest,
)

try:
    import nemotron_relational.rfm as rfm_engine
except (ImportError, RuntimeError) as error:
    rfm_engine = None
    _ENGINE_UNUSABLE = str(error)
else:
    _ENGINE_UNUSABLE = ''

requires_engine = pytest.mark.skipif(
    rfm_engine is None,
    reason=f'nemotron_relational.rfm is not usable in this environment: {_ENGINE_UNUSABLE}',
)


@pytest.fixture
def client() -> Transport:
    return Transport('https://nim.example.com:8000', api_key='secret')


class _FakeGraph:
    r"""Minimal stand-in exposing the ``tables`` mapping the adapter reads."""

    def __init__(self, *table_names: str) -> None:
        self.tables = dict.fromkeys(table_names)


class _FakeEngineModel:
    r"""Base for the engine stand-ins, supplying the retry context manager the
    adapter now enters whenever ``num_retries`` is non-zero."""

    def retry(self, num_retries=1):
        return contextlib.nullcontext()


@requires_engine
def test_predict_forwards_url_and_api_key_to_engine_init(monkeypatch, client):
    captured = {}
    monkeypatch.setattr(
        rfm_engine,
        'init_client',
        lambda **kwargs: captured.update(init=kwargs),
    )

    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            captured['graph'] = graph

        def predict(self, query, **kwargs):
            captured['predict'] = {'query': query, **kwargs}
            return pd.DataFrame({'entity': kwargs.get('indices')})

    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )

    result = NemotronRelationalAdapter().predict(
        client,
        KumoRelationalRequest(
            graph='fake-graph',
            query='PREDICT target FOR entity=1',
            indices=[1, 2, 3],
        ),
    )

    assert captured['init'] == {
        'url': client.url,
        'api_key': client.api_key,
        'verify_ssl': client.verify_ssl,
        'timeout': client.timeout,
        'max_retries': client.max_retries,
        '_token': rfm_engine._CLIENT_TOKEN,
    }
    assert captured['graph'] == 'fake-graph'
    assert captured['predict']['query'] == 'PREDICT target FOR entity=1'
    assert captured['predict']['run_mode'] == 'fast'
    assert list(result['entity']) == [1, 2, 3]


@requires_engine
def test_predict_forwards_the_client_timeout_to_engine_init(monkeypatch):
    captured = {}

    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            pass

        def predict(self, query, **kwargs):
            return pd.DataFrame({'entity': [1]})

    monkeypatch.setattr(
        rfm_engine, 'init_client', lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )

    transport = Transport('https://nim.example.com:8000', timeout=3.5)
    NemotronRelationalAdapter().predict(
        transport, KumoRelationalRequest(graph='g', query='PREDICT x')
    )

    assert captured['timeout'] == 3.5


@requires_engine
def test_predict_forwards_custom_run_mode_and_options(monkeypatch, client):
    captured = {}

    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            pass

        def predict(self, query, **kwargs):
            captured.update(kwargs)
            return pd.DataFrame({'entity': [1]})

    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)
    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )

    NemotronRelationalAdapter().predict(
        client,
        KumoRelationalRequest(
            graph='g',
            query='PREDICT x',
            indices=[1],
            run_mode='best',
            options={'num_hops': 3},
        ),
    )

    assert captured['run_mode'] == 'best'
    assert captured['num_hops'] == 3


@requires_engine
def test_predict_rejects_reserved_option_keys(monkeypatch, client):
    called = {}
    monkeypatch.setattr(
        rfm_engine,
        'init_client',
        lambda **kwargs: called.setdefault('init', True),
    )
    with pytest.raises(RelationalError) as excinfo:
        NemotronRelationalAdapter().predict(
            client,
            KumoRelationalRequest(
                graph='g', query='PREDICT x', options={'run_mode': 'best'}
            ),
        )
    assert excinfo.value.code == 'INVALID_REQUEST'
    assert 'init' not in called


@requires_engine
def test_predict_raises_type_error_on_non_dataframe_result(monkeypatch, client):
    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            pass

        def predict(self, query, **kwargs):
            return object()

    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)
    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )

    with pytest.raises(TypeError):
        NemotronRelationalAdapter().predict(
            client, KumoRelationalRequest(graph='g', query='PREDICT x')
        )


@requires_engine
def test_adapter_authorizes_engine_init(monkeypatch, client):
    captured = {}
    monkeypatch.setattr(
        rfm_engine, 'init_client', lambda **kwargs: captured.update(kwargs)
    )

    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            pass

        def predict(self, query, **kwargs):
            return pd.DataFrame({'entity': [1]})

    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )
    NemotronRelationalAdapter().predict(
        client, KumoRelationalRequest(graph='g', query='PREDICT x')
    )

    assert captured['_token'] is rfm_engine._CLIENT_TOKEN


def test_nemotron_relational_shim_does_not_expose_driver():
    from kumo_relational_client import relational as nemotron_relational_shim

    assert 'NemotronRelational' not in nemotron_relational_shim.__all__
    assert 'Graph' in nemotron_relational_shim.__all__
    with pytest.raises(AttributeError):
        nemotron_relational_shim.NemotronRelational


def test_nemotron_relational_shim_namespace_is_pinned():
    from kumo_relational_client import relational as nemotron_relational_shim

    # This namespace is the documented Nemotron Relational surface (README, quickstart
    # notebook). Pinned so a change here has to be a change to the docs too.
    assert nemotron_relational_shim.__all__ == [
        'Dtype',
        'ExplainConfig',
        'Explanation',
        'Graph',
        'LocalTable',
        'Stype',
        'Table',
        'ViewConversionWarning',
    ]
    assert nemotron_relational_shim.__dir__() == sorted(
        nemotron_relational_shim.__all__
    )


def test_nemotron_relational_shim_withholds_names_the_supported_api_cannot_reach():
    r"""`MaterializedPredictionRequest` and `TaskTable` are off the shim.

    Neither is reachable through `RelationalClient`: the first is returned only by
    `NemotronRelational.materialize_*` and the second is built by the adapter itself, and
    `NemotronRelational` is deliberately absent, so exporting them promised a surface that
    does not exist. `NemotronRelational` and `init` stay withheld for the same reason.
    """
    from kumo_relational_client import relational as nemotron_relational_shim

    for name in (
        'MaterializedPredictionRequest',
        'TaskTable',
        'NemotronRelational',
        'init',
        'LocalGraph',
    ):
        assert not hasattr(nemotron_relational_shim, name), name


@requires_engine
def test_nemotron_relational_shim_every_exported_name_resolves():
    from kumo_relational_client import relational as nemotron_relational_shim

    for name in nemotron_relational_shim.__all__:
        assert getattr(nemotron_relational_shim, name) is not None, name


@requires_engine
def test_nemotron_relational_shim_exposes_stype_documented_by_the_quickstart():
    # The quickstart documents `graph[t][c].stype = nemotron_relational.Stype.<type>`.
    from kumo_relational_client import relational as nemotron_relational_shim

    assert nemotron_relational_shim.Stype.categorical is not None
    assert nemotron_relational_shim.Dtype.bool is not None


def _patch_engine_import(monkeypatch, error: BaseException) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'nemotron_relational.rfm':
            raise error
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)


def test_predict_raises_missing_extra_error_when_engine_not_installed(
    monkeypatch,
    client,
):
    _patch_engine_import(
        monkeypatch,
        ModuleNotFoundError(
            "No module named 'nemotron_relational'", name='nemotron_relational'
        ),
    )
    with pytest.raises(MissingExtraError):
        NemotronRelationalAdapter().predict(
            client, KumoRelationalRequest(graph='g', query='PREDICT x')
        )


def test_predict_propagates_broken_driver_error(monkeypatch, client):
    _patch_engine_import(
        monkeypatch,
        RuntimeError('relationallib native extension failed to load'),
    )
    with pytest.raises(RuntimeError, match='native extension'):
        NemotronRelationalAdapter().predict(
            client, KumoRelationalRequest(graph='g', query='PREDICT x')
        )


def test_predict_propagates_transitive_import_error(monkeypatch, client):
    _patch_engine_import(
        monkeypatch,
        ModuleNotFoundError("No module named 'some_dep'", name='some_dep'),
    )
    with pytest.raises(ModuleNotFoundError, match='some_dep'):
        NemotronRelationalAdapter().predict(
            client, KumoRelationalRequest(graph='g', query='PREDICT x')
        )


@requires_engine
def test_predict_explain_field_returns_explanation(monkeypatch, client):
    from nemotron_relational.rfm.rfm import Explanation

    captured = {}
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)

    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            pass

        def predict(self, query, **kwargs):
            captured.update(kwargs)
            if kwargs.get('explain'):
                return Explanation(
                    prediction=pd.DataFrame({'ENTITY': [1]}),
                    summary='Price drove the prediction.',
                    details={'format': 'kumo_rfm_v2_1'},
                )
            return pd.DataFrame({'ENTITY': [1]})

    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )

    result = NemotronRelationalAdapter().predict(
        client,
        KumoRelationalRequest(
            graph='fake-graph', query='PREDICT t FOR e=1', explain=True
        ),
    )

    assert isinstance(result, Explanation)
    assert result.summary == 'Price drove the prediction.'
    assert captured['explain'] is True


@requires_engine
def test_predict_explain_via_options_returns_explanation(monkeypatch, client):
    from nemotron_relational.rfm.rfm import Explanation

    captured = {}
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)

    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            pass

        def predict(self, query, **kwargs):
            captured.update(kwargs)
            return Explanation(
                prediction=pd.DataFrame({'ENTITY': [1]}),
                summary='',
                details={'skip_summary': True},
            )

    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )

    result = NemotronRelationalAdapter().predict(
        client,
        KumoRelationalRequest(
            graph='fake-graph',
            query='PREDICT t FOR e=1',
            options={'explain': {'skip_summary': True}},
        ),
    )

    assert isinstance(result, Explanation)
    assert captured['explain'] == {'skip_summary': True}


@requires_engine
def test_predict_without_explain_still_requires_dataframe(monkeypatch, client):
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)

    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            pass

        def predict(self, query, **kwargs):
            return object()

    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )

    with pytest.raises(TypeError, match='expected a DataFrame result'):
        NemotronRelationalAdapter().predict(
            client,
            KumoRelationalRequest(
                graph='fake-graph', query='PREDICT t FOR e=1'
            ),
        )


@requires_engine
def test_predict_explain_rejects_non_explanation_result(monkeypatch, client):
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)

    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            pass

        def predict(self, query, **kwargs):
            return pd.DataFrame({'ENTITY': [1]})

    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )

    with pytest.raises(TypeError, match='expected an Explanation result'):
        NemotronRelationalAdapter().predict(
            client,
            KumoRelationalRequest(
                graph='fake-graph', query='PREDICT t FOR e=1', explain=True
            ),
        )


def test_capabilities_advertise_explanation():
    assert 'explanation' in NemotronRelationalAdapter().capabilities().outputs


@requires_engine
@pytest.mark.parametrize('field_value', [True, {}, {'skip_summary': True}])
def test_predict_rejects_explain_specified_twice(
    monkeypatch, client, field_value
):
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)
    monkeypatch.setattr(rfm_engine, 'NemotronRelational', lambda graph: None)

    with pytest.raises(RelationalError) as err:
        NemotronRelationalAdapter().predict(
            client,
            KumoRelationalRequest(
                graph='fake-graph',
                query='PREDICT t FOR e=1',
                explain=field_value,
                options={'explain': False},
            ),
        )
    assert err.value.code == 'INVALID_REQUEST'


@requires_engine
@pytest.mark.parametrize('bad', [0, '', 'yes', 1.0])
def test_predict_rejects_malformed_explain_values(monkeypatch, client, bad):
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)
    monkeypatch.setattr(rfm_engine, 'NemotronRelational', lambda graph: None)

    with pytest.raises(RelationalError) as err:
        NemotronRelationalAdapter().predict(
            client,
            KumoRelationalRequest(
                graph='fake-graph',
                query='PREDICT t FOR e=1',
                options={'explain': bad},
            ),
        )
    assert err.value.code == 'INVALID_REQUEST'


@requires_engine
def test_client_predict_explain_returns_explanation(monkeypatch):
    """Exercise issue #19: client.relational(graph).predict(explain=True)."""
    from nemotron_relational.rfm.rfm import Explanation

    from kumo_relational_client import RelationalClient

    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)

    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            pass

        def predict(self, query, **kwargs):
            assert kwargs['explain'] is True
            return Explanation(
                prediction=pd.DataFrame({'ENTITY': [1]}),
                summary='',
                details={'format': 'kumo_rfm_v2_1', 'details': {'cohorts': []}},
            )

    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )

    monkeypatch.setattr(
        'kumo_relational_client.core.transport.Transport.health_ready',
        lambda self: True,
    )
    with RelationalClient(url='http://127.0.0.1:18001') as client:
        result = client.relational('fake-graph').predict(
            'PREDICT t FOR e=1', explain=True
        )

    assert isinstance(result, Explanation)
    assert result.details['format'] == 'kumo_rfm_v2_1'


@requires_engine
def test_predict_accepts_explain_config_object(monkeypatch, client):
    """ExplainConfig instances are a valid driver explain input, not rejected
    as INVALID_REQUEST (the driver's predict accepts bool|ExplainConfig|dict)."""
    from nemotron_relational.rfm.rfm import ExplainConfig, Explanation

    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)
    captured = {}

    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            pass

        def predict(self, query, **kwargs):
            captured['explain'] = kwargs['explain']
            return Explanation(
                prediction=pd.DataFrame({'ENTITY': [1]}),
                summary='',
                details={},
                warning=None,
            )

    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )

    cfg = ExplainConfig(skip_summary=True)
    result = NemotronRelationalAdapter().predict(
        client,
        KumoRelationalRequest(
            graph='fake-graph', query='PREDICT t FOR e=1', explain=cfg
        ),
    )

    assert isinstance(result, Explanation)
    assert captured['explain'] is cfg


@requires_engine
def test_adapter_enters_batch_mode_when_batch_size_set(monkeypatch, client):
    calls = {}
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)

    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            pass

        def batch_mode(self, batch_size, num_retries=1):
            calls['batch_mode'] = (batch_size, num_retries)
            return contextlib.nullcontext()

        def predict(self, query, **kwargs):
            calls['predicted'] = True
            return pd.DataFrame({'ENTITY': [1]})

    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )
    NemotronRelationalAdapter().predict(
        client,
        KumoRelationalRequest(
            graph='g',
            query='PREDICT x FOR EACH t.id',
            indices=list(range(1500)),
            batch_size='max',
            num_retries=3,
        ),
    )

    assert calls['batch_mode'] == ('max', 3)
    assert calls.get('predicted')


@requires_engine
def test_adapter_skips_batch_mode_when_unset(monkeypatch, client):
    calls = {}
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)

    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            pass

        def batch_mode(self, *args, **kwargs):
            calls['batch_mode'] = True
            return contextlib.nullcontext()

        def predict(self, query, **kwargs):
            return pd.DataFrame({'ENTITY': [1]})

    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )
    NemotronRelationalAdapter().predict(
        client,
        KumoRelationalRequest(graph='g', query='PREDICT x FOR t.id=1'),
    )

    assert 'batch_mode' not in calls


@requires_engine
def test_adapter_rejects_invalid_batch_size(monkeypatch, client):
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)
    monkeypatch.setattr(rfm_engine, 'NemotronRelational', lambda graph: None)

    with pytest.raises(RelationalError) as err:
        NemotronRelationalAdapter().predict(
            client,
            KumoRelationalRequest(
                graph='g', query='PREDICT x FOR t.id=1', batch_size='auto'
            ),
        )
    assert err.value.code == 'INVALID_REQUEST'


def _failing_engine(monkeypatch, error: BaseException) -> None:
    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            pass

        def predict(self, query, **kwargs):
            raise error

    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)
    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )


def _predict(client):
    return NemotronRelationalAdapter().predict(
        client,
        KumoRelationalRequest(graph='g', query='PREDICT x FOR t.id=1'),
    )


@requires_engine
def test_nim_failure_becomes_a_nim_request_error(monkeypatch, client):
    from nemotron_relational.exceptions import NimFailureError

    params = [
        {
            'name': 'context.related_tables.users.rows[0][big]',
            'reason': 'exceeds the JSON safe integer range',
        }
    ]
    _failing_engine(
        monkeypatch,
        NimFailureError(
            'The NemotronRelational NIM rejected this prediction (HTTP 422): bad data.',
            status_code=422,
            detail='bad data',
            invalid_params=params,
        ),
    )

    with pytest.raises(NimRequestError) as excinfo:
        _predict(client)
    assert isinstance(excinfo.value, RelationalError)
    assert excinfo.value.status_code == 422
    assert excinfo.value.details['invalid_params'] == params
    assert 'create an issue' not in str(excinfo.value)


@requires_engine
def test_nim_failure_without_a_status_becomes_a_transport_error(
    monkeypatch, client
):
    from nemotron_relational.exceptions import NimFailureError

    _failing_engine(
        monkeypatch,
        NimFailureError(
            'did not answer within the configured timeout', transient=True
        ),
    )

    with pytest.raises(RelationalError) as excinfo:
        _predict(client)
    assert excinfo.value.code == 'TRANSPORT_ERROR'


@requires_engine
def test_unexpected_engine_failure_becomes_internal_error(monkeypatch, client):
    """No bare exception escapes."""
    _failing_engine(monkeypatch, RuntimeError('the wheels came off'))

    with pytest.raises(RelationalError) as excinfo:
        _predict(client)
    assert excinfo.value.code == 'INTERNAL_ERROR'
    assert client.url in str(excinfo.value)
    assert 'RuntimeError' in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


@requires_engine
def test_engine_lookup_failure_becomes_invalid_request(monkeypatch, client):
    """A name the caller supplied that the engine looked up and did not find is a
    caller mistake, not a client failure, however deep the lookup happened. The
    key is reported unwrapped rather than as ``KeyError``'s ``repr``.
    """
    _failing_engine(monkeypatch, KeyError('nope'))

    with pytest.raises(RelationalError) as excinfo:
        _predict(client)
    assert excinfo.value.code == 'INVALID_REQUEST'
    assert str(excinfo.value) == '[INVALID_REQUEST] nope'
    assert isinstance(excinfo.value.__cause__, KeyError)


@requires_engine
def test_malformed_response_becomes_invalid_response(monkeypatch, client):
    """A malformed server response is the server's fault, so it must not be
    reported as a bad request.
    """
    from nemotron_relational.exceptions import InvalidResponseError

    _failing_engine(
        monkeypatch,
        InvalidResponseError(
            'The NemotronRelational NIM returned a prediction response that does not match '
            'the contract (KeyError: id)'
        ),
    )

    with pytest.raises(RelationalError) as excinfo:
        _predict(client)
    assert excinfo.value.code == 'INVALID_RESPONSE'
    assert 'does not match the contract' in str(excinfo.value)


@requires_engine
def test_engine_validation_error_keeps_its_message(monkeypatch, client):
    """Client-side validation the engine performs is already actionable, so it
    must not be relabelled as an internal client failure.
    """
    _failing_engine(
        monkeypatch,
        ValueError('Context anchor timestamp is too early for the given graph'),
    )

    with pytest.raises(RelationalError) as excinfo:
        _predict(client)
    assert excinfo.value.code == 'INVALID_REQUEST'
    assert 'Context anchor timestamp is too early' in str(excinfo.value)


@requires_engine
def test_adapter_predict_error_is_not_rewrapped(monkeypatch, client):
    original = RelationalError('already ours', code='INVALID_REQUEST')
    _failing_engine(monkeypatch, original)

    with pytest.raises(RelationalError) as excinfo:
        _predict(client)
    assert excinfo.value is original


@requires_engine
def test_num_retries_applies_without_batch_size(monkeypatch, client):
    calls = {}
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)

    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            pass

        def retry(self, num_retries=1):
            calls['retry'] = num_retries
            return contextlib.nullcontext()

        def predict(self, query, **kwargs):
            return pd.DataFrame({'ENTITY': [1]})

    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )
    NemotronRelationalAdapter().predict(
        client,
        KumoRelationalRequest(
            graph='g', query='PREDICT x FOR t.id=1', num_retries=5
        ),
    )

    assert calls['retry'] == 5


@requires_engine
def test_zero_num_retries_enters_no_context(monkeypatch, client):
    calls = {}
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)

    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            pass

        def retry(self, num_retries=1):
            calls['retry'] = num_retries
            return contextlib.nullcontext()

        def predict(self, query, **kwargs):
            return pd.DataFrame({'ENTITY': [1]})

    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )
    NemotronRelationalAdapter().predict(
        client,
        KumoRelationalRequest(
            graph='g', query='PREDICT x FOR t.id=1', num_retries=0
        ),
    )

    assert 'retry' not in calls


@requires_engine
def test_adapter_rejects_negative_num_retries(monkeypatch, client):
    """Rejected on both paths, not just batch."""
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)
    monkeypatch.setattr(rfm_engine, 'NemotronRelational', lambda graph: None)

    with pytest.raises(RelationalError) as excinfo:
        NemotronRelationalAdapter().predict(
            client,
            KumoRelationalRequest(
                graph='g', query='PREDICT x FOR t.id=1', num_retries=-1
            ),
        )
    assert excinfo.value.code == 'INVALID_REQUEST'


@requires_engine
def test_predict_task_builds_task_table_and_calls_engine(monkeypatch, client):
    captured = {}
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)

    class FakeTaskTable:
        ENTITY_TIME = '__entity_time__'

        def __init__(self, **kwargs):
            captured['task_table'] = kwargs

    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            captured['graph'] = graph

        def predict_task(self, task, **kwargs):
            captured['predict_task'] = {'task': task, **kwargs}
            return pd.DataFrame({'ENTITY': [3]})

    monkeypatch.setattr(rfm_engine, 'TaskTable', FakeTaskTable)
    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )

    context = pd.DataFrame(
        {
            'ENTITY': [1, 2],
            'TARGET': ['a', 'b'],
            'ANCHOR_TIMESTAMP': pd.to_datetime(['2025-01-01', '2025-01-02']),
        }
    )
    predict = pd.DataFrame({'ENTITY': [3]})

    out = NemotronRelationalAdapter().predict(
        client,
        KumoRelationalTaskRequest(
            graph=_FakeGraph('users'),
            context=context,
            predict=predict,
            task_type='multiclass_classification',
            entity_table='users',
        ),
    )

    assert isinstance(out, pd.DataFrame)
    assert isinstance(captured['predict_task']['task'], FakeTaskTable)
    assert captured['predict_task']['run_mode'] == 'fast'
    assert captured['predict_task']['explain'] is False
    tt = captured['task_table']
    assert tt['task_type'] == 'multiclass_classification'
    assert tt['entity_table_name'] == 'users'
    assert tt['entity_column'] == 'ENTITY'
    assert tt['target_column'] == 'TARGET'
    assert tt['time_column'] == 'ANCHOR_TIMESTAMP'
    assert tt['context_df'] is context
    assert tt['pred_df'] is predict


@requires_engine
def test_predict_task_uses_anchor_timestamp_from_predict_only(
    monkeypatch, client
):
    captured = {}
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)

    class FakeTaskTable:
        ENTITY_TIME = '__entity_time__'

        def __init__(self, **kwargs):
            captured['task_table'] = kwargs

    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            pass

        def predict_task(self, task, **kwargs):
            return pd.DataFrame({'ENTITY': [2]})

    monkeypatch.setattr(rfm_engine, 'TaskTable', FakeTaskTable)
    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )

    NemotronRelationalAdapter().predict(
        client,
        KumoRelationalTaskRequest(
            graph=_FakeGraph('users'),
            context=pd.DataFrame({'ENTITY': [1], 'TARGET': ['a']}),
            predict=pd.DataFrame(
                {
                    'ENTITY': [2],
                    'ANCHOR_TIMESTAMP': pd.to_datetime(['2025-02-01']),
                }
            ),
            task_type='multiclass_classification',
            entity_table='users',
        ),
    )

    assert captured['task_table']['time_column'] == 'ANCHOR_TIMESTAMP'


@requires_engine
def test_predict_task_defaults_time_column_to_entity_time(monkeypatch, client):
    captured = {}
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)

    class FakeTaskTable:
        ENTITY_TIME = '__entity_time__'

        def __init__(self, **kwargs):
            captured['task_table'] = kwargs

    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            pass

        def predict_task(self, task, **kwargs):
            return pd.DataFrame({'ENTITY': [2]})

    monkeypatch.setattr(rfm_engine, 'TaskTable', FakeTaskTable)
    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )

    NemotronRelationalAdapter().predict(
        client,
        KumoRelationalTaskRequest(
            graph=_FakeGraph('users'),
            context=pd.DataFrame({'ENTITY': [1], 'TARGET': ['a']}),
            predict=pd.DataFrame({'ENTITY': [2]}),
            task_type='multiclass_classification',
            entity_table='users',
        ),
    )

    assert captured['task_table']['time_column'] == FakeTaskTable.ENTITY_TIME


@requires_engine
def test_predict_task_returns_explanation_and_forwards_options(
    monkeypatch, client
):
    from nemotron_relational.rfm.rfm import Explanation

    captured = {}
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)

    class FakeTaskTable:
        ENTITY_TIME = '__entity_time__'

        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(rfm_engine, 'TaskTable', FakeTaskTable)

    explanation = Explanation.__new__(Explanation)

    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            pass

        def predict_task(self, task, **kwargs):
            captured.update(kwargs)
            return explanation

    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )

    out = NemotronRelationalAdapter().predict(
        client,
        KumoRelationalTaskRequest(
            graph=_FakeGraph('users'),
            context=pd.DataFrame({'ENTITY': [1], 'TARGET': ['a']}),
            predict=pd.DataFrame({'ENTITY': [1]}),
            task_type='multiclass_classification',
            entity_table='users',
            explain=True,
            options={'num_neighbors': [4, 4]},
        ),
    )

    assert out is explanation
    assert captured['explain'] is True
    assert captured['num_neighbors'] == [4, 4]


@requires_engine
def test_predict_task_rejects_reserved_option_keys(monkeypatch, client):
    called = {}
    monkeypatch.setattr(
        rfm_engine,
        'init_client',
        lambda **kwargs: called.setdefault('init', True),
    )

    with pytest.raises(RelationalError) as err:
        NemotronRelationalAdapter().predict(
            client,
            KumoRelationalTaskRequest(
                graph=_FakeGraph('users'),
                context=pd.DataFrame({'ENTITY': [1], 'TARGET': ['a']}),
                predict=pd.DataFrame({'ENTITY': [1]}),
                task_type='regression',
                entity_table='users',
                options={'run_mode': 'best'},
            ),
        )
    assert err.value.code == 'INVALID_REQUEST'
    assert 'init' not in called


def test_capabilities_list_both_request_types():
    assert (
        NemotronRelationalAdapter().capabilities().request_type
        == 'KumoRelationalRequest | KumoRelationalTaskRequest'
    )


@requires_engine
def test_predict_task_rejects_unknown_task_type(monkeypatch, client):
    called = {}
    monkeypatch.setattr(
        rfm_engine,
        'init_client',
        lambda **kwargs: called.setdefault('init', True),
    )

    with pytest.raises(RelationalError) as err:
        NemotronRelationalAdapter().predict(
            client,
            KumoRelationalTaskRequest(
                graph=_FakeGraph('users'),
                context=pd.DataFrame({'ENTITY': [1], 'TARGET': ['a']}),
                predict=pd.DataFrame({'ENTITY': [2]}),
                task_type='__unknown_task_type__',
                entity_table='users',
            ),
        )
    assert err.value.code == 'INVALID_REQUEST'
    assert 'task_type' in str(err.value)
    assert 'init' not in called


@requires_engine
def test_predict_task_rejects_entity_table_absent_from_graph(
    monkeypatch, client
):
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)

    with pytest.raises(RelationalError) as err:
        NemotronRelationalAdapter().predict(
            client,
            KumoRelationalTaskRequest(
                graph=_FakeGraph('users', 'orders'),
                context=pd.DataFrame({'ENTITY': [1], 'TARGET': ['a']}),
                predict=pd.DataFrame({'ENTITY': [2]}),
                task_type='regression',
                entity_table='customers',
            ),
        )
    assert err.value.code == 'INVALID_REQUEST'
    assert 'customers' in str(err.value)


@requires_engine
def test_predict_task_rejects_missing_target_column(monkeypatch, client):
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)

    with pytest.raises(RelationalError) as err:
        NemotronRelationalAdapter().predict(
            client,
            KumoRelationalTaskRequest(
                graph=_FakeGraph('users'),
                context=pd.DataFrame({'ENTITY': [1]}),
                predict=pd.DataFrame({'ENTITY': [2]}),
                task_type='regression',
                entity_table='users',
            ),
        )
    assert err.value.code == 'INVALID_REQUEST'
    assert 'context' in str(err.value)
    assert 'TARGET' in str(err.value)


@requires_engine
def test_predict_task_rejects_missing_entity_column_in_predict(
    monkeypatch, client
):
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)

    with pytest.raises(RelationalError) as err:
        NemotronRelationalAdapter().predict(
            client,
            KumoRelationalTaskRequest(
                graph=_FakeGraph('users'),
                context=pd.DataFrame({'ENTITY': [1], 'TARGET': ['a']}),
                predict=pd.DataFrame({'WRONG': [2]}),
                task_type='regression',
                entity_table='users',
            ),
        )
    assert err.value.code == 'INVALID_REQUEST'
    assert 'predict' in str(err.value)


def test_capabilities_advertise_supported_task_types():
    from kumo_relational_client.adapters.relational import RFM_TASK_TYPES

    tasks = NemotronRelationalAdapter().capabilities().tasks
    assert tasks == RFM_TASK_TYPES
    assert 'multiclass_classification' in tasks


def test_task_types_match_engine_task_type_enum():
    task_module = pytest.importorskip('nemotron_relational.api.task')
    from kumo_relational_client.adapters.relational import RFM_TASK_TYPES

    for name in RFM_TASK_TYPES:
        assert task_module.TaskType(name).value == name


def test_stype_and_dtype_belong_to_the_supported_namespace():
    # The documented way to correct an inferred semantic type is
    # graph[table][column].stype = nemotron_relational.Stype.categorical, and the client
    # presents kumo_relational_client.relational as the supported namespace. Membership is
    # asserted separately from resolution because it holds with or without the
    # engine installed -- this runs in the job that has neither.
    from kumo_relational_client import relational

    assert {'Stype', 'Dtype'} <= set(dir(relational))


@requires_engine
def test_stype_and_dtype_resolve_when_the_engine_is_present():
    # These live on the top-level relational package rather than nemotron_relational.rfm, so
    # they need their own resolution path and used to raise AttributeError.
    from kumo_relational_client import relational

    assert relational.Stype.categorical is not None
    assert relational.Dtype.string is not None


def test_unknown_attribute_still_raises():
    from kumo_relational_client import relational

    with pytest.raises(AttributeError, match='no attribute'):
        relational.DefinitelyNotExported


@requires_engine
def test_verbose_reaches_the_engine_constructor_too(monkeypatch, client):
    """The graph-materialization banner is owned by ``NemotronRelational.__init__``, and a
    handle builds a fresh engine model per prediction -- so forwarding
    ``verbose`` only to ``predict`` still left that banner on stdout on every
    single call, which live verification caught.
    """
    captured = {}

    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, verbose=True, **kwargs):
            captured['init_verbose'] = verbose

        def predict(self, query, **kwargs):
            captured['predict_verbose'] = kwargs.get('verbose')
            return pd.DataFrame({'entity': [1]})

    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)
    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )

    NemotronRelationalAdapter().predict(
        client,
        KumoRelationalRequest(
            graph='g', query='PREDICT x', options={'verbose': False}
        ),
    )

    assert captured['init_verbose'] is False
    assert captured['predict_verbose'] is False


@requires_engine
def test_engine_keeps_its_own_verbose_default_when_unset(monkeypatch, client):
    captured = {}

    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, verbose=True, **kwargs):
            captured['init_verbose'] = verbose

        def predict(self, query, **kwargs):
            captured['predict_verbose'] = 'verbose' in kwargs
            return pd.DataFrame({'entity': [1]})

    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)
    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )

    NemotronRelationalAdapter().predict(
        client, KumoRelationalRequest(graph='g', query='PREDICT x')
    )

    assert captured['init_verbose'] is True
    assert captured['predict_verbose'] is False


@requires_engine
def test_concurrent_clients_predict_against_their_own_endpoint(
    monkeypatch,
) -> None:
    r"""Regression test for `quality-clients-share-process-global-engine-state`.

    The engine's endpoint and credential are process-global, and the adapter
    used to configure them and then let ``NemotronRelational`` read them back lazily --
    at the first HTTP call, after graph sampling. A second client configuring
    a different endpoint inside that window silently redirected the first
    client's prediction, uploading one tenant's relational context to another
    tenant's NIM under that tenant's API key.

    The recorder resolves its endpoint exactly the way ``_api_client`` does:
    the client bound at construction if there is one, the process global
    otherwise. The barrier only forces an interleaving that is otherwise
    timing-dependent.
    """
    import threading

    import nemotron_relational
    from nemotron_relational.client.client import (
        RelationalClient as EngineRelationalClient,
    )

    from kumo_relational_client import RelationalClient

    monkeypatch.setattr(
        EngineRelationalClient, 'authenticate', lambda self: None
    )

    barrier = threading.Barrier(2)
    seen: dict[str, tuple[str, str]] = {}

    class RecordingNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            barrier.wait()
            bound = kwargs.get('_client')
            self._resolved = (
                bound
                if bound is not None
                else nemotron_relational.global_state.client
            )

        def predict(self, query, **kwargs):
            seen[threading.current_thread().name] = (
                self._resolved._url,
                self._resolved._api_key,
            )
            return pd.DataFrame({'entity': []})

    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', RecordingNemotronRelational
    )

    clients = {
        'A': RelationalClient(url='https://tenant-a.example', api_key='key-A'),
        'B': RelationalClient(url='https://tenant-b.example', api_key='key-B'),
    }

    def run(name: str) -> None:
        clients[name].relational(_FakeGraph('users')).predict(
            'PREDICT COUNT(o.*, 0, 30) FOR u.id=1', indices=[1]
        )

    try:
        threads = [
            threading.Thread(target=run, args=(name,), name=name)
            for name in ('A', 'B')
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        rfm_engine.global_state.reset()

    assert seen == {
        'A': ('https://tenant-a.example', 'key-A'),
        'B': ('https://tenant-b.example', 'key-B'),
    }


@requires_engine
def test_predict_task_names_a_feature_column_missing_from_predict(client):
    """Anything in ``context`` beyond entity/target/time becomes a task feature,
    and the engine then reads the same column out of ``predict``. Present in
    only one frame it raised a bare ``KeyError`` naming the column but not the
    constraint, which the classifier could only report as an internal failure.
    """
    with pytest.raises(RelationalError) as excinfo:
        NemotronRelationalAdapter().predict(
            client,
            KumoRelationalTaskRequest(
                graph=_FakeGraph('users'),
                context=pd.DataFrame(
                    {'ENTITY': [1], 'TARGET': ['a'], 'EXTRA': [1]}
                ),
                predict=pd.DataFrame({'ENTITY': [2]}),
                task_type='regression',
                entity_table='users',
            ),
        )

    assert excinfo.value.code == 'INVALID_REQUEST'
    assert "['EXTRA']" in str(excinfo.value)
    assert 'supply them in both frames' in str(excinfo.value)


@requires_engine
@pytest.mark.parametrize(
    ('context_columns', 'predict_columns'),
    [
        ({'ENTITY': [1], 'TARGET': ['a']}, {'ENTITY': [2]}),
        (
            {'ENTITY': [1], 'TARGET': ['a'], 'EXTRA': [1]},
            {'ENTITY': [2], 'EXTRA': [3]},
        ),
        ({'ENTITY': [1], 'TARGET': ['a']}, {'ENTITY': [2], 'EXTRA': [3]}),
        (
            {
                'ENTITY': [1],
                'TARGET': ['a'],
                'ANCHOR_TIMESTAMP': pd.to_datetime(['2025-01-01']),
            },
            {'ENTITY': [2]},
        ),
    ],
)
def test_predict_task_still_accepts_the_supported_frame_shapes(
    monkeypatch, client, context_columns, predict_columns
):
    r"""The guard must not reject what already worked: a feature in both
    frames, a feature in ``predict`` alone (ignored), and an anchor timestamp
    in ``context`` alone.
    """

    class FakeTaskTable:
        ENTITY_TIME = '__entity_time__'

        def __init__(self, **kwargs):
            pass

    class FakeNemotronRelational(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            pass

        def predict_task(self, task, **kwargs):
            return pd.DataFrame({'ENTITY': [2]})

    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: None)
    monkeypatch.setattr(rfm_engine, 'TaskTable', FakeTaskTable)
    monkeypatch.setattr(
        rfm_engine, 'NemotronRelational', FakeNemotronRelational
    )

    out = NemotronRelationalAdapter().predict(
        client,
        KumoRelationalTaskRequest(
            graph=_FakeGraph('users'),
            context=pd.DataFrame(context_columns),
            predict=pd.DataFrame(predict_columns),
            task_type='regression',
            entity_table='users',
        ),
    )
    assert isinstance(out, pd.DataFrame)


class _SignedGraph(_FakeGraph):
    r"""A graph that can describe its schema, so the adapter may cache it."""

    def __init__(self, *table_names: str, signature: str = 'v1') -> None:
        super().__init__(*table_names)
        self._signature = signature

    def _to_api_graph_definition(self) -> str:
        return self._signature


def _counting_engine(monkeypatch, builds: list) -> None:
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: 'client-a')

    class CountingModel(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            builds.append(graph)

        def predict(self, query, **kwargs):
            return pd.DataFrame({'entity': kwargs.get('indices') or [1]})

    monkeypatch.setattr(rfm_engine, 'NemotronRelational', CountingModel)


def _request(graph) -> KumoRelationalRequest:
    return KumoRelationalRequest(
        graph=graph, query='PREDICT target FOR entity=1', indices=[1]
    )


@requires_engine
def test_the_graph_is_materialized_once_across_predictions(
    monkeypatch, client
) -> None:
    r"""Materializing is the expensive half of a prediction and does not depend
    on the query, so repeating a question against one graph must not repeat
    it."""
    builds: list = []
    _counting_engine(monkeypatch, builds)
    graph = _SignedGraph('users', 'orders')
    adapter = NemotronRelationalAdapter()

    for _ in range(3):
        adapter.predict(client, _request(graph))

    assert len(builds) == 1


@requires_engine
def test_a_different_graph_is_materialized_again(monkeypatch, client) -> None:
    builds: list = []
    _counting_engine(monkeypatch, builds)
    adapter = NemotronRelationalAdapter()

    adapter.predict(client, _request(_SignedGraph('users')))
    adapter.predict(client, _request(_SignedGraph('users')))

    assert len(builds) == 2


@requires_engine
def test_a_graph_altered_in_place_is_materialized_again(
    monkeypatch, client
) -> None:
    r"""Identity alone would hand back a model built for the old schema."""
    builds: list = []
    _counting_engine(monkeypatch, builds)
    graph = _SignedGraph('users', signature='before')
    adapter = NemotronRelationalAdapter()

    adapter.predict(client, _request(graph))
    graph._signature = 'after'
    adapter.predict(client, _request(graph))

    assert len(builds) == 2


@requires_engine
def test_a_reconfigured_endpoint_is_not_served_the_old_model(
    monkeypatch, client
) -> None:
    r"""The engine client is a process-wide singleton rebuilt whenever the URL
    or credential changes, so a new one means this prediction must not reuse a
    model bound to the previous endpoint."""
    builds: list = []
    _counting_engine(monkeypatch, builds)
    graph = _SignedGraph('users')
    adapter = NemotronRelationalAdapter()

    adapter.predict(client, _request(graph))
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: 'client-b')
    adapter.predict(client, _request(graph))

    assert len(builds) == 2


@requires_engine
def test_a_graph_that_cannot_describe_itself_is_not_cached(
    monkeypatch, client
) -> None:
    r"""Correctness must not depend on the cache, so an undescribable graph is
    rebuilt rather than reused on a guess."""
    builds: list = []
    _counting_engine(monkeypatch, builds)
    graph = _FakeGraph('users')
    adapter = NemotronRelationalAdapter()

    adapter.predict(client, _request(graph))
    adapter.predict(client, _request(graph))

    assert len(builds) == 2


@requires_engine
def test_closing_releases_the_materialized_graph(monkeypatch, client) -> None:
    r"""A materialized graph can be gigabytes; closing the client has to let it
    go rather than hold it for the process's life."""
    builds: list = []
    _counting_engine(monkeypatch, builds)
    graph = _SignedGraph('users')
    adapter = NemotronRelationalAdapter()

    adapter.predict(client, _request(graph))
    adapter.close()
    adapter.predict(client, _request(graph))

    assert len(builds) == 2


@requires_engine
def test_a_managed_serving_target_is_never_cached(monkeypatch, client) -> None:
    r"""Serving resolves no client object, so an entry keyed on it would be
    keyed on ``None`` and two endpoints would share one model."""
    from kumo_relational_client.core.serving import DatabricksServingTarget

    builds: list = []
    _counting_engine(monkeypatch, builds)
    monkeypatch.setattr(
        'kumo_relational_client.adapters.relational._init_serving',
        lambda engine, target: None,
    )
    graph = _SignedGraph('users')
    adapter = NemotronRelationalAdapter()
    target = DatabricksServingTarget('kumo-relational')

    adapter.predict(target, _request(graph))
    adapter.predict(target, _request(graph))

    assert len(builds) == 2


@requires_engine
def test_each_thread_materializes_its_own_model(monkeypatch, client) -> None:
    r"""A model carries the batch size and retry count that `batch_mode` and
    `retry` set and restore, so two threads must not share one."""
    import threading

    builds: list = []
    _counting_engine(monkeypatch, builds)
    graph = _SignedGraph('users')
    adapter = NemotronRelationalAdapter()
    seen: list = []

    def run() -> None:
        adapter.predict(client, _request(graph))
        adapter.predict(client, _request(graph))
        seen.append(True)

    threads = [threading.Thread(target=run) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(seen) == 3
    assert len(builds) == 3


@requires_engine
def test_alternating_between_graphs_materializes_each_time(
    monkeypatch, client
) -> None:
    r"""One entry is kept, so this is the case the cache does not serve; pinned
    so the behaviour the docstring promises cannot drift from the code."""
    builds: list = []
    _counting_engine(monkeypatch, builds)
    first = _SignedGraph('users', signature='first')
    second = _SignedGraph('orders', signature='second')
    adapter = NemotronRelationalAdapter()

    adapter.predict(client, _request(first))
    adapter.predict(client, _request(second))
    adapter.predict(client, _request(first))

    assert len(builds) == 3


@requires_engine
def test_the_old_graph_is_released_before_the_new_one_is_built(
    monkeypatch, client
) -> None:
    r"""Peak memory, not steady state, is what fails a worker: a materialized
    graph can be gigabytes, so holding the outgoing one while the replacement
    is built doubles the peak exactly when a service refreshes its graph."""
    monkeypatch.setattr(rfm_engine, 'init_client', lambda **kwargs: 'client-a')
    adapter = NemotronRelationalAdapter()
    held_during_build: list = []

    class WatchingModel(_FakeEngineModel):
        def __init__(self, graph, **kwargs):
            held_during_build.append(
                getattr(adapter._engine_cache, 'entry', None)
            )

        def predict(self, query, **kwargs):
            return pd.DataFrame({'entity': [1]})

    monkeypatch.setattr(rfm_engine, 'NemotronRelational', WatchingModel)

    adapter.predict(client, _request(_SignedGraph('users', signature='a')))
    adapter.predict(client, _request(_SignedGraph('orders', signature='b')))

    assert held_during_build == [None, None]

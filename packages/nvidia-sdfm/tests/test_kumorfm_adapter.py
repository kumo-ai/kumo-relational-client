# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib

import pandas as pd
import pytest

from nvidia_sdfm.adapters.kumorfm import KumoRFMAdapter
from nvidia_sdfm.core.transport import Transport
from nvidia_sdfm.errors import MissingExtraError, NimRequestError, SdfmError
from nvidia_sdfm.requests import KumoRFMRequest, KumoRFMTaskRequest

try:
    import kumorfm.rfm as rfm_engine
except (ImportError, RuntimeError) as error:
    rfm_engine = None
    _ENGINE_UNUSABLE = str(error)
else:
    _ENGINE_UNUSABLE = ''

requires_engine = pytest.mark.skipif(
    rfm_engine is None,
    reason=f'kumorfm.rfm is not usable in this environment: {_ENGINE_UNUSABLE}',
)


@pytest.fixture
def client() -> Transport:
    return Transport('https://nim.example.com:8000', api_key='secret')


class _FakeGraph:
    r"""Minimal stand-in exposing the ``tables`` mapping the adapter reads."""

    def __init__(self, *table_names: str) -> None:
        self.tables = {name: None for name in table_names}


class _FakeEngineModel:
    r"""Base for the engine stand-ins, supplying the retry context manager the
    adapter now enters whenever ``num_retries`` is non-zero."""

    def retry(self, num_retries=1):
        return contextlib.nullcontext()


@requires_engine
def test_predict_forwards_url_and_api_key_to_engine_init(monkeypatch, client):
    captured = {}
    monkeypatch.setattr(
        rfm_engine, 'init',
        lambda **kwargs: captured.update(init=kwargs),
    )

    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph):
            captured['graph'] = graph

        def predict(self, query, **kwargs):
            captured['predict'] = {'query': query, **kwargs}
            return pd.DataFrame({'entity': kwargs.get('indices')})

    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    result = KumoRFMAdapter().predict(client, KumoRFMRequest(
        graph='fake-graph', query='PREDICT target FOR entity=1',
        indices=[1, 2, 3]))

    assert captured['init'] == {
        'url': client.url,
        'api_key': client.api_key,
        'verify_ssl': client.verify_ssl,
        'timeout': client.timeout,
        '_token': rfm_engine._SDFM_CLIENT_TOKEN,
    }
    assert captured['graph'] == 'fake-graph'
    assert captured['predict']['query'] == 'PREDICT target FOR entity=1'
    assert captured['predict']['run_mode'] == 'fast'
    assert list(result['entity']) == [1, 2, 3]


@requires_engine
def test_predict_forwards_the_client_timeout_to_engine_init(monkeypatch):
    captured = {}

    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph):
            pass

        def predict(self, query, **kwargs):
            return pd.DataFrame({'entity': [1]})

    monkeypatch.setattr(rfm_engine, 'init',
                        lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    transport = Transport('https://nim.example.com:8000', timeout=3.5)
    KumoRFMAdapter().predict(
        transport, KumoRFMRequest(graph='g', query='PREDICT x'))

    assert captured['timeout'] == 3.5


@requires_engine
def test_predict_forwards_custom_run_mode_and_options(monkeypatch, client):
    captured = {}

    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph):
            pass

        def predict(self, query, **kwargs):
            captured.update(kwargs)
            return pd.DataFrame({'entity': [1]})

    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)
    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    KumoRFMAdapter().predict(client, KumoRFMRequest(
        graph='g', query='PREDICT x', indices=[1], run_mode='best',
        options={'num_hops': 3}))

    assert captured['run_mode'] == 'best'
    assert captured['num_hops'] == 3


@requires_engine
def test_predict_rejects_reserved_option_keys(monkeypatch, client):
    called = {}
    monkeypatch.setattr(rfm_engine, 'init',
                        lambda **kwargs: called.setdefault('init', True))
    with pytest.raises(SdfmError) as excinfo:
        KumoRFMAdapter().predict(client, KumoRFMRequest(
            graph='g', query='PREDICT x', options={'run_mode': 'best'}))
    assert excinfo.value.code == 'INVALID_REQUEST'
    assert 'init' not in called


@requires_engine
def test_predict_raises_type_error_on_non_dataframe_result(monkeypatch, client):
    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph):
            pass

        def predict(self, query, **kwargs):
            return object()

    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)
    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    with pytest.raises(TypeError):
        KumoRFMAdapter().predict(
            client, KumoRFMRequest(graph='g', query='PREDICT x'))


@requires_engine
def test_adapter_authorizes_engine_init(monkeypatch, client):
    captured = {}
    monkeypatch.setattr(rfm_engine, 'init',
                        lambda **kwargs: captured.update(kwargs))

    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph):
            pass

        def predict(self, query, **kwargs):
            return pd.DataFrame({'entity': [1]})

    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)
    KumoRFMAdapter().predict(
        client, KumoRFMRequest(graph='g', query='PREDICT x'))

    assert captured['_token'] is rfm_engine._SDFM_CLIENT_TOKEN


def test_kumorfm_shim_does_not_expose_driver():
    from nvidia_sdfm import kumorfm as kumorfm_shim

    assert 'KumoRFM' not in kumorfm_shim.__all__
    assert 'Graph' in kumorfm_shim.__all__
    with pytest.raises(AttributeError):
        kumorfm_shim.KumoRFM


def test_kumorfm_shim_namespace_is_pinned():
    from nvidia_sdfm import kumorfm as kumorfm_shim

    # This namespace is the documented KumoRFM surface (README, quickstart
    # notebook). Pinned so a change here has to be a change to the docs too.
    assert kumorfm_shim.__all__ == [
        'Dtype',
        'ExplainConfig',
        'Explanation',
        'Graph',
        'LocalTable',
        'MaterializedPredictionRequest',
        'Stype',
        'Table',
        'TaskTable',
    ]
    assert kumorfm_shim.__dir__() == sorted(kumorfm_shim.__all__)


@requires_engine
def test_kumorfm_shim_every_exported_name_resolves():
    from nvidia_sdfm import kumorfm as kumorfm_shim

    for name in kumorfm_shim.__all__:
        assert getattr(kumorfm_shim, name) is not None, name


@requires_engine
def test_kumorfm_shim_exposes_stype_documented_by_the_quickstart():
    # The quickstart documents `graph[t][c].stype = kumorfm.Stype.<type>`.
    from nvidia_sdfm import kumorfm as kumorfm_shim

    assert kumorfm_shim.Stype.categorical is not None
    assert kumorfm_shim.Dtype.bool is not None


def _patch_engine_import(monkeypatch, error: BaseException) -> None:
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'kumorfm.rfm':
            raise error
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)


def test_predict_raises_missing_extra_error_when_engine_not_installed(
    monkeypatch, client,
):
    _patch_engine_import(
        monkeypatch,
        ModuleNotFoundError("No module named 'kumorfm'", name='kumorfm'),
    )
    with pytest.raises(MissingExtraError):
        KumoRFMAdapter().predict(
            client, KumoRFMRequest(graph='g', query='PREDICT x'))


def test_predict_propagates_broken_driver_error(monkeypatch, client):
    _patch_engine_import(
        monkeypatch, RuntimeError('kumolib native extension failed to load'))
    with pytest.raises(RuntimeError, match='native extension'):
        KumoRFMAdapter().predict(
            client, KumoRFMRequest(graph='g', query='PREDICT x'))


def test_predict_propagates_transitive_import_error(monkeypatch, client):
    _patch_engine_import(
        monkeypatch,
        ModuleNotFoundError("No module named 'some_dep'", name='some_dep'),
    )
    with pytest.raises(ModuleNotFoundError, match='some_dep'):
        KumoRFMAdapter().predict(
            client, KumoRFMRequest(graph='g', query='PREDICT x'))


@requires_engine
def test_predict_explain_field_returns_explanation(monkeypatch, client):
    from kumorfm.rfm.rfm import Explanation

    captured = {}
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph):
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

    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    result = KumoRFMAdapter().predict(client, KumoRFMRequest(
        graph='fake-graph', query='PREDICT t FOR e=1', explain=True))

    assert isinstance(result, Explanation)
    assert result.summary == 'Price drove the prediction.'
    assert captured['explain'] is True


@requires_engine
def test_predict_explain_via_options_returns_explanation(monkeypatch, client):
    from kumorfm.rfm.rfm import Explanation

    captured = {}
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph):
            pass

        def predict(self, query, **kwargs):
            captured.update(kwargs)
            return Explanation(
                prediction=pd.DataFrame({'ENTITY': [1]}),
                summary='',
                details={'skip_summary': True},
            )

    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    result = KumoRFMAdapter().predict(client, KumoRFMRequest(
        graph='fake-graph', query='PREDICT t FOR e=1',
        options={'explain': {'skip_summary': True}}))

    assert isinstance(result, Explanation)
    assert captured['explain'] == {'skip_summary': True}


@requires_engine
def test_predict_without_explain_still_requires_dataframe(monkeypatch, client):
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph):
            pass

        def predict(self, query, **kwargs):
            return object()

    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    with pytest.raises(TypeError, match='expected a DataFrame result'):
        KumoRFMAdapter().predict(client, KumoRFMRequest(
            graph='fake-graph', query='PREDICT t FOR e=1'))


@requires_engine
def test_predict_explain_rejects_non_explanation_result(monkeypatch, client):
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph):
            pass

        def predict(self, query, **kwargs):
            return pd.DataFrame({'ENTITY': [1]})

    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    with pytest.raises(TypeError, match='expected an Explanation result'):
        KumoRFMAdapter().predict(client, KumoRFMRequest(
            graph='fake-graph', query='PREDICT t FOR e=1', explain=True))


def test_capabilities_advertise_explanation():
    assert 'explanation' in KumoRFMAdapter().capabilities().outputs


@requires_engine
@pytest.mark.parametrize('field_value', [True, {}, {'skip_summary': True}])
def test_predict_rejects_explain_specified_twice(monkeypatch, client, field_value):
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)
    monkeypatch.setattr(rfm_engine, 'KumoRFM', lambda graph: None)

    with pytest.raises(SdfmError) as err:
        KumoRFMAdapter().predict(client, KumoRFMRequest(
            graph='fake-graph', query='PREDICT t FOR e=1',
            explain=field_value, options={'explain': False}))
    assert err.value.code == 'INVALID_REQUEST'


@requires_engine
@pytest.mark.parametrize('bad', [0, '', 'yes', 1.0])
def test_predict_rejects_malformed_explain_values(monkeypatch, client, bad):
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)
    monkeypatch.setattr(rfm_engine, 'KumoRFM', lambda graph: None)

    with pytest.raises(SdfmError) as err:
        KumoRFMAdapter().predict(client, KumoRFMRequest(
            graph='fake-graph', query='PREDICT t FOR e=1',
            options={'explain': bad}))
    assert err.value.code == 'INVALID_REQUEST'


@requires_engine
def test_sdfm_client_predict_explain_returns_explanation(monkeypatch):
    """Exercise issue #19: client.kumorfm(graph).predict(explain=True)."""
    from kumorfm.rfm.rfm import Explanation

    from nvidia_sdfm import SDFMClient

    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph):
            pass

        def predict(self, query, **kwargs):
            assert kwargs['explain'] is True
            return Explanation(
                prediction=pd.DataFrame({'ENTITY': [1]}),
                summary='',
                details={'format': 'kumo_rfm_v2_1', 'details': {'cohorts': []}},
            )

    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    monkeypatch.setattr(
        'nvidia_sdfm.core.transport.Transport.health_ready', lambda self: True)
    with SDFMClient(url='http://127.0.0.1:18001') as sdfm:
        result = sdfm.kumorfm('fake-graph').predict(
            'PREDICT t FOR e=1', explain=True)

    assert isinstance(result, Explanation)
    assert result.details['format'] == 'kumo_rfm_v2_1'


@requires_engine
def test_predict_accepts_explain_config_object(monkeypatch, client):
    """ExplainConfig instances are a valid driver explain input, not rejected
    as INVALID_REQUEST (the driver's predict accepts bool|ExplainConfig|dict)."""
    from kumorfm.rfm.rfm import ExplainConfig, Explanation

    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)
    captured = {}

    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph):
            pass

        def predict(self, query, **kwargs):
            captured['explain'] = kwargs['explain']
            return Explanation(
                prediction=pd.DataFrame({'ENTITY': [1]}),
                summary='', details={}, warning=None)

    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    cfg = ExplainConfig(skip_summary=True)
    result = KumoRFMAdapter().predict(client, KumoRFMRequest(
        graph='fake-graph', query='PREDICT t FOR e=1', explain=cfg))

    assert isinstance(result, Explanation)
    assert captured['explain'] is cfg


@requires_engine
def test_adapter_enters_batch_mode_when_batch_size_set(monkeypatch, client):
    calls = {}
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph):
            pass

        def batch_mode(self, batch_size, num_retries=1):
            calls['batch_mode'] = (batch_size, num_retries)
            return contextlib.nullcontext()

        def predict(self, query, **kwargs):
            calls['predicted'] = True
            return pd.DataFrame({'ENTITY': [1]})

    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)
    KumoRFMAdapter().predict(client, KumoRFMRequest(
        graph='g', query='PREDICT x FOR EACH t.id',
        indices=list(range(1500)), batch_size='max', num_retries=3))

    assert calls['batch_mode'] == ('max', 3)
    assert calls.get('predicted')


@requires_engine
def test_adapter_skips_batch_mode_when_unset(monkeypatch, client):
    calls = {}
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph):
            pass

        def batch_mode(self, *args, **kwargs):
            calls['batch_mode'] = True
            return contextlib.nullcontext()

        def predict(self, query, **kwargs):
            return pd.DataFrame({'ENTITY': [1]})

    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)
    KumoRFMAdapter().predict(client, KumoRFMRequest(
        graph='g', query='PREDICT x FOR t.id=1'))

    assert 'batch_mode' not in calls


@requires_engine
def test_adapter_rejects_invalid_batch_size(monkeypatch, client):
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)
    monkeypatch.setattr(rfm_engine, 'KumoRFM', lambda graph: None)

    with pytest.raises(SdfmError) as err:
        KumoRFMAdapter().predict(client, KumoRFMRequest(
            graph='g', query='PREDICT x FOR t.id=1', batch_size='auto'))
    assert err.value.code == 'INVALID_REQUEST'


def _failing_engine(monkeypatch, error: BaseException) -> None:
    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph):
            pass

        def predict(self, query, **kwargs):
            raise error

    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)
    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)


def _predict(client):
    return KumoRFMAdapter().predict(
        client, KumoRFMRequest(graph='g', query='PREDICT x FOR t.id=1'))


@requires_engine
def test_nim_failure_becomes_a_nim_request_error(monkeypatch, client):
    r"""client-rfm-path-never-raises-sdfmerror.md +
    rfm-nim-validation-details-discarded.md"""
    from kumorfm.exceptions import NimFailureError

    params = [{'name': 'context.related_tables.users.rows[0][big]',
               'reason': 'exceeds the JSON safe integer range'}]
    _failing_engine(monkeypatch, NimFailureError(
        'The KumoRFM NIM rejected this prediction (HTTP 422): bad data.',
        status_code=422, detail='bad data', invalid_params=params))

    with pytest.raises(NimRequestError) as excinfo:
        _predict(client)
    assert isinstance(excinfo.value, SdfmError)
    assert excinfo.value.status_code == 422
    assert excinfo.value.details['invalid_params'] == params
    assert 'create an issue' not in str(excinfo.value)


@requires_engine
def test_nim_failure_without_a_status_becomes_a_transport_error(
        monkeypatch, client):
    r"""client-rfm-path-never-raises-sdfmerror.md"""
    from kumorfm.exceptions import NimFailureError

    _failing_engine(monkeypatch, NimFailureError(
        'did not answer within the configured timeout', transient=True))

    with pytest.raises(SdfmError) as excinfo:
        _predict(client)
    assert excinfo.value.code == 'TRANSPORT_ERROR'


@requires_engine
def test_unexpected_engine_failure_becomes_internal_error(monkeypatch, client):
    r"""client-rfm-path-never-raises-sdfmerror.md: no bare KeyError escapes."""
    _failing_engine(monkeypatch, KeyError('id'))

    with pytest.raises(SdfmError) as excinfo:
        _predict(client)
    assert excinfo.value.code == 'INTERNAL_ERROR'
    assert client.url in str(excinfo.value)
    assert 'KeyError' in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, KeyError)


@requires_engine
def test_malformed_response_becomes_invalid_response(monkeypatch, client):
    r"""client-rfm-path-never-raises-sdfmerror.md

    A malformed server response is the server's fault, so it must not be
    reported as a bad request.
    """
    from kumorfm.exceptions import InvalidResponseError

    _failing_engine(monkeypatch, InvalidResponseError(
        'The KumoRFM NIM returned a prediction response that does not match '
        'the contract (KeyError: id)'))

    with pytest.raises(SdfmError) as excinfo:
        _predict(client)
    assert excinfo.value.code == 'INVALID_RESPONSE'
    assert 'does not match the contract' in str(excinfo.value)


@requires_engine
def test_engine_validation_error_keeps_its_message(monkeypatch, client):
    r"""client-rfm-path-never-raises-sdfmerror.md

    Client-side validation the engine performs is already actionable, so it
    must not be relabelled as an internal SDK failure.
    """
    _failing_engine(monkeypatch, ValueError(
        'Context anchor timestamp is too early for the given graph'))

    with pytest.raises(SdfmError) as excinfo:
        _predict(client)
    assert excinfo.value.code == 'INVALID_REQUEST'
    assert 'Context anchor timestamp is too early' in str(excinfo.value)


@requires_engine
def test_adapter_sdfm_error_is_not_rewrapped(monkeypatch, client):
    original = SdfmError('already ours', code='INVALID_REQUEST')
    _failing_engine(monkeypatch, original)

    with pytest.raises(SdfmError) as excinfo:
        _predict(client)
    assert excinfo.value is original


@requires_engine
def test_num_retries_applies_without_batch_size(monkeypatch, client):
    r"""rfm-num-retries-silent-noop.md"""
    calls = {}
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph):
            pass

        def retry(self, num_retries=1):
            calls['retry'] = num_retries
            return contextlib.nullcontext()

        def predict(self, query, **kwargs):
            return pd.DataFrame({'ENTITY': [1]})

    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)
    KumoRFMAdapter().predict(client, KumoRFMRequest(
        graph='g', query='PREDICT x FOR t.id=1', num_retries=5))

    assert calls['retry'] == 5


@requires_engine
def test_zero_num_retries_enters_no_context(monkeypatch, client):
    r"""rfm-num-retries-silent-noop.md"""
    calls = {}
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph):
            pass

        def retry(self, num_retries=1):
            calls['retry'] = num_retries
            return contextlib.nullcontext()

        def predict(self, query, **kwargs):
            return pd.DataFrame({'ENTITY': [1]})

    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)
    KumoRFMAdapter().predict(client, KumoRFMRequest(
        graph='g', query='PREDICT x FOR t.id=1', num_retries=0))

    assert 'retry' not in calls


@requires_engine
def test_adapter_rejects_negative_num_retries(monkeypatch, client):
    r"""rfm-num-retries-silent-noop.md: rejected on both paths, not just batch."""
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)
    monkeypatch.setattr(rfm_engine, 'KumoRFM', lambda graph: None)

    with pytest.raises(SdfmError) as excinfo:
        KumoRFMAdapter().predict(client, KumoRFMRequest(
            graph='g', query='PREDICT x FOR t.id=1', num_retries=-1))
    assert excinfo.value.code == 'INVALID_REQUEST'


@requires_engine
def test_predict_task_builds_task_table_and_calls_engine(monkeypatch, client):
    captured = {}
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    class FakeTaskTable:
        ENTITY_TIME = '__entity_time__'

        def __init__(self, **kwargs):
            captured['task_table'] = kwargs

    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph):
            captured['graph'] = graph

        def predict_task(self, task, **kwargs):
            captured['predict_task'] = {'task': task, **kwargs}
            return pd.DataFrame({'ENTITY': [3]})

    monkeypatch.setattr(rfm_engine, 'TaskTable', FakeTaskTable)
    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    context = pd.DataFrame({
        'ENTITY': [1, 2],
        'TARGET': ['a', 'b'],
        'ANCHOR_TIMESTAMP': pd.to_datetime(['2025-01-01', '2025-01-02']),
    })
    predict = pd.DataFrame({'ENTITY': [3]})

    out = KumoRFMAdapter().predict(client, KumoRFMTaskRequest(
        graph=_FakeGraph('users'), context=context, predict=predict,
        task_type='multiclass_classification', entity_table='users'))

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
        monkeypatch, client):
    captured = {}
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    class FakeTaskTable:
        ENTITY_TIME = '__entity_time__'

        def __init__(self, **kwargs):
            captured['task_table'] = kwargs

    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph):
            pass

        def predict_task(self, task, **kwargs):
            return pd.DataFrame({'ENTITY': [2]})

    monkeypatch.setattr(rfm_engine, 'TaskTable', FakeTaskTable)
    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    KumoRFMAdapter().predict(client, KumoRFMTaskRequest(
        graph=_FakeGraph('users'),
        context=pd.DataFrame({'ENTITY': [1], 'TARGET': ['a']}),
        predict=pd.DataFrame({
            'ENTITY': [2],
            'ANCHOR_TIMESTAMP': pd.to_datetime(['2025-02-01']),
        }),
        task_type='multiclass_classification', entity_table='users'))

    assert captured['task_table']['time_column'] == 'ANCHOR_TIMESTAMP'


@requires_engine
def test_predict_task_defaults_time_column_to_entity_time(monkeypatch, client):
    captured = {}
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    class FakeTaskTable:
        ENTITY_TIME = '__entity_time__'

        def __init__(self, **kwargs):
            captured['task_table'] = kwargs

    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph):
            pass

        def predict_task(self, task, **kwargs):
            return pd.DataFrame({'ENTITY': [2]})

    monkeypatch.setattr(rfm_engine, 'TaskTable', FakeTaskTable)
    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    KumoRFMAdapter().predict(client, KumoRFMTaskRequest(
        graph=_FakeGraph('users'),
        context=pd.DataFrame({'ENTITY': [1], 'TARGET': ['a']}),
        predict=pd.DataFrame({'ENTITY': [2]}),
        task_type='multiclass_classification', entity_table='users'))

    assert captured['task_table']['time_column'] == FakeTaskTable.ENTITY_TIME


@requires_engine
def test_predict_task_returns_explanation_and_forwards_options(
        monkeypatch, client):
    from kumorfm.rfm.rfm import Explanation

    captured = {}
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    class FakeTaskTable:
        ENTITY_TIME = '__entity_time__'

        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(rfm_engine, 'TaskTable', FakeTaskTable)

    explanation = Explanation.__new__(Explanation)

    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph):
            pass

        def predict_task(self, task, **kwargs):
            captured.update(kwargs)
            return explanation

    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    out = KumoRFMAdapter().predict(client, KumoRFMTaskRequest(
        graph=_FakeGraph('users'),
        context=pd.DataFrame({'ENTITY': [1], 'TARGET': ['a']}),
        predict=pd.DataFrame({'ENTITY': [1]}),
        task_type='multiclass_classification', entity_table='users',
        explain=True, options={'num_neighbors': [4, 4]}))

    assert out is explanation
    assert captured['explain'] is True
    assert captured['num_neighbors'] == [4, 4]


@requires_engine
def test_predict_task_rejects_reserved_option_keys(monkeypatch, client):
    called = {}
    monkeypatch.setattr(rfm_engine, 'init',
                        lambda **kwargs: called.setdefault('init', True))

    with pytest.raises(SdfmError) as err:
        KumoRFMAdapter().predict(client, KumoRFMTaskRequest(
            graph=_FakeGraph('users'),
            context=pd.DataFrame({'ENTITY': [1], 'TARGET': ['a']}),
            predict=pd.DataFrame({'ENTITY': [1]}),
            task_type='regression', entity_table='users',
            options={'run_mode': 'best'}))
    assert err.value.code == 'INVALID_REQUEST'
    assert 'init' not in called


def test_capabilities_list_both_request_types():
    assert (KumoRFMAdapter().capabilities().request_type
            == 'KumoRFMRequest | KumoRFMTaskRequest')


@requires_engine
def test_predict_task_rejects_unknown_task_type(monkeypatch, client):
    called = {}
    monkeypatch.setattr(rfm_engine, 'init',
                        lambda **kwargs: called.setdefault('init', True))

    with pytest.raises(SdfmError) as err:
        KumoRFMAdapter().predict(client, KumoRFMTaskRequest(
            graph=_FakeGraph('users'),
            context=pd.DataFrame({'ENTITY': [1], 'TARGET': ['a']}),
            predict=pd.DataFrame({'ENTITY': [2]}),
            task_type='__unknown_task_type__', entity_table='users'))
    assert err.value.code == 'INVALID_REQUEST'
    assert 'task_type' in str(err.value)
    assert 'init' not in called


@requires_engine
def test_predict_task_rejects_entity_table_absent_from_graph(
        monkeypatch, client):
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    with pytest.raises(SdfmError) as err:
        KumoRFMAdapter().predict(client, KumoRFMTaskRequest(
            graph=_FakeGraph('users', 'orders'),
            context=pd.DataFrame({'ENTITY': [1], 'TARGET': ['a']}),
            predict=pd.DataFrame({'ENTITY': [2]}),
            task_type='regression', entity_table='customers'))
    assert err.value.code == 'INVALID_REQUEST'
    assert 'customers' in str(err.value)


@requires_engine
def test_predict_task_rejects_missing_target_column(monkeypatch, client):
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    with pytest.raises(SdfmError) as err:
        KumoRFMAdapter().predict(client, KumoRFMTaskRequest(
            graph=_FakeGraph('users'),
            context=pd.DataFrame({'ENTITY': [1]}),
            predict=pd.DataFrame({'ENTITY': [2]}),
            task_type='regression', entity_table='users'))
    assert err.value.code == 'INVALID_REQUEST'
    assert 'context' in str(err.value)
    assert 'TARGET' in str(err.value)


@requires_engine
def test_predict_task_rejects_missing_entity_column_in_predict(
        monkeypatch, client):
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    with pytest.raises(SdfmError) as err:
        KumoRFMAdapter().predict(client, KumoRFMTaskRequest(
            graph=_FakeGraph('users'),
            context=pd.DataFrame({'ENTITY': [1], 'TARGET': ['a']}),
            predict=pd.DataFrame({'WRONG': [2]}),
            task_type='regression', entity_table='users'))
    assert err.value.code == 'INVALID_REQUEST'
    assert 'predict' in str(err.value)


def test_capabilities_advertise_supported_task_types():
    from nvidia_sdfm.adapters.kumorfm import RFM_TASK_TYPES
    tasks = KumoRFMAdapter().capabilities().tasks
    assert tasks == RFM_TASK_TYPES
    assert 'multiclass_classification' in tasks


def test_task_types_match_engine_task_type_enum():
    task_module = pytest.importorskip('kumorfm.api.task')
    from nvidia_sdfm.adapters.kumorfm import RFM_TASK_TYPES
    for name in RFM_TASK_TYPES:
        assert task_module.TaskType(name).value == name


@requires_engine
def test_verbose_reaches_the_engine_constructor_too(monkeypatch, client):
    r"""rfm-no-way-to-silence-progress-output.md

    The graph-materialization banner is owned by ``KumoRFM.__init__``, and a
    handle builds a fresh engine model per prediction -- so forwarding
    ``verbose`` only to ``predict`` still left that banner on stdout on every
    single call, which live verification caught.
    """
    captured = {}

    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph, verbose=True):
            captured['init_verbose'] = verbose

        def predict(self, query, **kwargs):
            captured['predict_verbose'] = kwargs.get('verbose')
            return pd.DataFrame({'entity': [1]})

    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)
    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    KumoRFMAdapter().predict(client, KumoRFMRequest(
        graph='g', query='PREDICT x', options={'verbose': False}))

    assert captured['init_verbose'] is False
    assert captured['predict_verbose'] is False


@requires_engine
def test_engine_keeps_its_own_verbose_default_when_unset(monkeypatch, client):
    captured = {}

    class FakeKumoRFM(_FakeEngineModel):
        def __init__(self, graph, verbose=True):
            captured['init_verbose'] = verbose

        def predict(self, query, **kwargs):
            captured['predict_verbose'] = 'verbose' in kwargs
            return pd.DataFrame({'entity': [1]})

    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)
    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    KumoRFMAdapter().predict(client, KumoRFMRequest(graph='g',
                                                    query='PREDICT x'))

    assert captured['init_verbose'] is True
    assert captured['predict_verbose'] is False

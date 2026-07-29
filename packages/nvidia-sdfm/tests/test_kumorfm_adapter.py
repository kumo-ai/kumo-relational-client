# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib

import pandas as pd
import pytest

from nvidia_sdfm.adapters.kumorfm import KumoRFMAdapter
from nvidia_sdfm.core.transport import Transport
from nvidia_sdfm.errors import MissingExtraError, SdfmError
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


@requires_engine
def test_predict_forwards_url_and_api_key_to_engine_init(monkeypatch, client):
    captured = {}
    monkeypatch.setattr(
        rfm_engine, 'init',
        lambda **kwargs: captured.update(init=kwargs),
    )

    class FakeKumoRFM:
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
        '_token': rfm_engine._SDFM_CLIENT_TOKEN,
    }
    assert captured['graph'] == 'fake-graph'
    assert captured['predict']['query'] == 'PREDICT target FOR entity=1'
    assert captured['predict']['run_mode'] == 'fast'
    assert list(result['entity']) == [1, 2, 3]


@requires_engine
def test_predict_forwards_custom_run_mode_and_options(monkeypatch, client):
    captured = {}

    class FakeKumoRFM:
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
    class FakeKumoRFM:
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

    class FakeKumoRFM:
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

    class FakeKumoRFM:
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

    class FakeKumoRFM:
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

    class FakeKumoRFM:
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

    class FakeKumoRFM:
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

    class FakeKumoRFM:
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

    class FakeKumoRFM:
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

    class FakeKumoRFM:
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

    class FakeKumoRFM:
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


@requires_engine
def test_predict_task_builds_task_table_and_calls_engine(monkeypatch, client):
    captured = {}
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    class FakeTaskTable:
        ENTITY_TIME = '__entity_time__'

        def __init__(self, **kwargs):
            captured['task_table'] = kwargs

    class FakeKumoRFM:
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

    class FakeKumoRFM:
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

    class FakeKumoRFM:
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

    class FakeKumoRFM:
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

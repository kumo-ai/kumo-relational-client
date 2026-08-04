# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
import pytest

from nvidia_sdfm import (
    RFMModel,
    SDFMClient,
    TabICLModel,
)
from nvidia_sdfm.base import ModelAdapter, ModelCapabilities
from nvidia_sdfm.errors import SdfmError
from nvidia_sdfm.requests import (
    KumoRFMRequest,
    KumoRFMTaskRequest,
    TabICLRequest,
)


class _CapturingAdapter(ModelAdapter):
    """Records the request it receives and returns a fixed DataFrame, so a
    handle can be driven end-to-end through the real SDFMClient
    dispatch without a live NIM."""

    def __init__(self, name, request_type, result):
        self.name = name
        self.request_type = request_type
        self.result = result
        self.captured = None

    def capabilities(self):
        return ModelCapabilities(
            model=self.name, request_type=self.request_type.__name__)

    def predict(self, transport, request):
        self.captured = request
        return self.result


def _client_with(adapter) -> SDFMClient:
    client = SDFMClient(url='http://nim.test')
    client._register(adapter)
    return client


def test_rfm_returns_bound_handle():
    client = SDFMClient(url='http://nim.test')
    handle = client.kumorfm('my-graph')
    assert isinstance(handle, RFMModel)


def test_rfm_handle_end_to_end_through_client():
    result = pd.DataFrame({'ENTITY': [1], 'PREDICTION': [0.5]})
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMRequest, result)
    client = _client_with(adapter)

    out = client.kumorfm('my-graph').predict(
        'PREDICT x FOR EACH t.id', [1, 2, 3],
        run_mode='best', anchor_time='2025-01-01')

    assert out is result
    req = adapter.captured
    assert isinstance(req, KumoRFMRequest)
    assert req.model == 'kumo-rfm'
    assert req.graph == 'my-graph'
    assert req.query == 'PREDICT x FOR EACH t.id'
    assert req.indices == [1, 2, 3]
    assert req.run_mode == 'best'
    assert req.options == {'anchor_time': '2025-01-01'}


def test_rfm_handle_defaults_are_minimal():
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMRequest, pd.DataFrame())
    client = _client_with(adapter)

    client.kumorfm('g').predict('PREDICT x')

    req = adapter.captured
    assert req.indices is None
    assert req.run_mode == 'fast'
    assert req.options == {}


def test_rfm_handle_only_forwards_set_options():
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMRequest, pd.DataFrame())
    client = _client_with(adapter)

    client.kumorfm('g').predict(
        'PREDICT x',
        inference_config={'num_estimators': 2},
        return_embeddings=True)

    assert adapter.captured.options == {
        'inference_config': {'num_estimators': 2},
        'return_embeddings': True,
    }


def test_rfm_handle_can_silence_progress_output():
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMRequest, pd.DataFrame())
    client = _client_with(adapter)

    client.kumorfm('g').predict('PREDICT x', verbose=False)

    assert adapter.captured.options == {'verbose': False}


def test_rfm_handle_forwards_engine_arguments_it_does_not_name():
    r"""The wrapper enumerates the engine's keywords, so anything it has not
    listed -- including whatever the engine gains next -- would otherwise be a
    ``TypeError`` with no way through.
    """
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMRequest, pd.DataFrame())
    client = _client_with(adapter)

    client.kumorfm('g').predict('PREDICT x', some_new_engine_knob=7)

    assert adapter.captured.options == {'some_new_engine_knob': 7}


def test_rfm_task_handle_exposes_link_prediction_and_column_exclusion():
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMTaskRequest, pd.DataFrame())
    client = _client_with(adapter)

    client.kumorfm('g').predict_task(
        pd.DataFrame({'ENTITY': [1], 'TARGET': [2]}),
        pd.DataFrame({'ENTITY': [1]}),
        task_type='temporal_link_prediction',
        entity_table=('users', 'items'),
        top_k=5,
        exclude_cols_dict={'users': ['ssn']},
        verbose=False,
    )

    assert adapter.captured.options == {
        'top_k': 5,
        'exclude_cols_dict': {'users': ['ssn']},
        'verbose': False,
    }


def test_rfm_handle_matches_typed_request():
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMRequest, pd.DataFrame())
    client = _client_with(adapter)

    client.kumorfm('g').predict('PREDICT x', [1], run_mode='normal',
                            anchor_time='2025-06-01')
    via_handle = adapter.captured

    typed = KumoRFMRequest(graph='g', query='PREDICT x', indices=[1],
                           run_mode='normal',
                           options={'anchor_time': '2025-06-01'})

    assert via_handle == typed


def test_tabicl_returns_bound_handle():
    client = SDFMClient(url='http://nim.test')
    handle = client.tabicl(pd.DataFrame({'y': [0, 1]}),
                           target='y', task='classification')
    assert isinstance(handle, TabICLModel)


def test_tabicl_handle_end_to_end_through_client():
    result = pd.DataFrame({'prediction': [1]})
    adapter = _CapturingAdapter('tabicl', TabICLRequest, result)
    client = _client_with(adapter)

    ctx = pd.DataFrame({'x': [1, 2], 'y': [0, 1]})
    rows = pd.DataFrame({'x': [3]})
    out = client.tabicl(ctx, target='y', task='classification').predict(
        rows, positive_class='1')

    assert out is result
    req = adapter.captured
    assert isinstance(req, TabICLRequest)
    assert req.model == 'tabicl'
    assert req.context.equals(ctx)
    assert req.predict.equals(rows)
    assert req.task == 'classification'
    assert req.target == 'y'
    assert req.positive_class == '1'


def test_tabicl_handle_defaults_are_minimal():
    adapter = _CapturingAdapter('tabicl', TabICLRequest, pd.DataFrame())
    client = _client_with(adapter)

    ctx = pd.DataFrame({'x': [1], 'y': [0]})
    client.tabicl(ctx, target='y', task='regression').predict(
        pd.DataFrame({'x': [2]}))

    req = adapter.captured
    assert req.outputs == ['prediction']
    assert req.positive_class is None
    assert req.max_results is None


def test_handle_still_dispatches_by_request_type():
    adapter = _CapturingAdapter('kumo-rfm', TabICLRequest, pd.DataFrame())
    client = _client_with(adapter)

    from nvidia_sdfm import SdfmError
    with pytest.raises(SdfmError):
        client.kumorfm('g').predict('PREDICT x')


def test_public_predict_is_not_exposed():
    client = SDFMClient(url='http://nim.test')
    assert not hasattr(client, 'predict')
    assert hasattr(client, 'kumorfm')
    assert hasattr(client, 'tabicl')


def test_rfm_handle_forwards_explain():
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMRequest, pd.DataFrame())
    client = _client_with(adapter)

    client.kumorfm('g').predict('PREDICT x FOR t.id=1', explain=True)
    assert adapter.captured.explain is True

    client.kumorfm('g').predict('PREDICT x FOR t.id=1')
    assert adapter.captured.explain is False


def test_rfm_handle_forwards_num_neighbors_and_num_hops():
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMRequest, pd.DataFrame())
    client = _client_with(adapter)

    client.kumorfm('g').predict('PREDICT x', num_neighbors=[8, 8], num_hops=3)

    assert adapter.captured.options == {
        'num_neighbors': [8, 8],
        'num_hops': 3,
    }


def test_rfm_handle_forwards_explain_config():
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMRequest, pd.DataFrame())
    client = _client_with(adapter)

    cfg = {'skip_summary': True}
    client.kumorfm('g').predict('PREDICT x FOR t.id=1', explain=cfg)
    assert adapter.captured.explain == cfg


def test_tabicl_handle_forwards_request_id():
    adapter = _CapturingAdapter('tabicl', TabICLRequest, pd.DataFrame())
    client = _client_with(adapter)

    ctx = pd.DataFrame({'x': [1], 'y': [0]})
    client.tabicl(ctx, target='y', task='classification').predict(
        pd.DataFrame({'x': [2]}), request_id='trace-123')

    assert adapter.captured.request_id == 'trace-123'


def test_rfm_handle_forwards_batch_size():
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMRequest, pd.DataFrame())
    client = _client_with(adapter)

    client.kumorfm('g').predict('PREDICT x FOR EACH t.id',
                                list(range(1500)),
                                batch_size='max', num_retries=2)

    req = adapter.captured
    assert req.batch_size == 'max'
    assert req.num_retries == 2


def test_rfm_handle_batch_defaults_off():
    """batch_size defaults to None, not 'max': an unqualified predict must not
    enter batch mode. The adapter branches on ``batch_size is not None``, so
    flipping this default silently changes how every existing call executes --
    this makes such a flip a visible, deliberate edit."""
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMRequest, pd.DataFrame())
    client = _client_with(adapter)

    client.kumorfm('g').predict('PREDICT x')

    assert adapter.captured.batch_size is None
    assert adapter.captured.num_retries == 1


def test_rfm_handle_predict_task_end_to_end_through_client():
    result = pd.DataFrame({'ENTITY': [3], 'PREDICTION': ['pro']})
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMTaskRequest, result)
    client = _client_with(adapter)

    context = pd.DataFrame({'ENTITY': [1, 2], 'TARGET': ['free', 'pro']})
    predict = pd.DataFrame({'ENTITY': [3]})
    out = client.kumorfm('my-graph').predict_task(
        context=context, predict=predict,
        task_type='multiclass_classification', entity_table='users',
        run_mode='best', num_neighbors=[8, 8])

    assert out is result
    req = adapter.captured
    assert isinstance(req, KumoRFMTaskRequest)
    assert req.model == 'kumo-rfm'
    assert req.graph == 'my-graph'
    assert req.context is context
    assert req.predict is predict
    assert req.task_type == 'multiclass_classification'
    assert req.entity_table == 'users'
    assert req.run_mode == 'best'
    assert req.options == {'num_neighbors': [8, 8]}


def test_rfm_handle_predict_task_defaults_are_minimal():
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMTaskRequest, pd.DataFrame())
    client = _client_with(adapter)

    client.kumorfm('g').predict_task(
        context=pd.DataFrame({'ENTITY': [1], 'TARGET': ['a']}),
        predict=pd.DataFrame({'ENTITY': [2]}),
        task_type='regression', entity_table='users')

    req = adapter.captured
    assert req.entity_column == 'ENTITY'
    assert req.target_column == 'TARGET'
    assert req.time_column is None
    assert req.num_forecasts == 1
    assert req.step_size is None
    assert req.run_mode == 'fast'
    assert req.options == {}


def test_rfm_handle_predict_task_only_forwards_set_options():
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMTaskRequest, pd.DataFrame())
    client = _client_with(adapter)

    client.kumorfm('g').predict_task(
        context=pd.DataFrame({'ENTITY': [1], 'TARGET': ['a']}),
        predict=pd.DataFrame({'ENTITY': [2]}),
        task_type='regression', entity_table='users',
        inference_config={'num_estimators': 2})

    assert adapter.captured.options == {
        'inference_config': {'num_estimators': 2},
    }


_NOT_A_FRAME = [
    {'a': [1, 2], 'y': [0, 1]},
    [{'a': 1, 'y': 0}, {'a': 2, 'y': 1}],
    pd.Series([1, 2], name='a'),
    None,
    'context.csv',
]


@pytest.mark.parametrize('value', _NOT_A_FRAME)
def test_tabicl_handle_rejects_a_non_frame_context(value):
    r"""client-non-dataframe-tables-raise-a-bare-attributeerror.md

    These are the shapes a pandas user reaches for before building a frame.
    Each used to reach ``frame.columns`` inside the adapter and raise
    ``AttributeError``, which ``except SdfmError`` does not catch and which
    names neither the argument nor the type it should have been.
    """
    client = SDFMClient(url='http://nim.test')
    with pytest.raises(SdfmError) as excinfo:
        client.tabicl(value, target='y', task='classification')
    assert excinfo.value.code == 'INVALID_REQUEST'
    assert 'context must be a pandas DataFrame' in str(excinfo.value)
    assert type(value).__name__ in str(excinfo.value)


@pytest.mark.parametrize('value', _NOT_A_FRAME)
def test_tabicl_handle_rejects_a_non_frame_predict(value):
    adapter = _CapturingAdapter('tabicl', TabICLRequest, pd.DataFrame())
    client = _client_with(adapter)
    handle = client.tabicl(pd.DataFrame({'a': [1.0], 'y': [0]}), target='y',
                           task='classification')

    with pytest.raises(SdfmError) as excinfo:
        handle.predict(value)
    assert excinfo.value.code == 'INVALID_REQUEST'
    assert 'predict must be a pandas DataFrame' in str(excinfo.value)


@pytest.mark.parametrize('argument', ['context', 'predict'])
def test_rfm_handle_predict_task_rejects_a_non_frame(argument):
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMTaskRequest, pd.DataFrame())
    client = _client_with(adapter)
    frames = {
        'context': pd.DataFrame({'ENTITY': [1], 'TARGET': ['a']}),
        'predict': pd.DataFrame({'ENTITY': [2]}),
    }
    frames[argument] = {'ENTITY': [1]}

    with pytest.raises(SdfmError) as excinfo:
        client.kumorfm('g').predict_task(
            **frames, task_type='regression', entity_table='users')
    assert excinfo.value.code == 'INVALID_REQUEST'
    assert f'{argument} must be a pandas DataFrame' in str(excinfo.value)


def test_a_dict_or_series_is_told_how_to_become_a_row():
    client = SDFMClient(url='http://nim.test')
    for value in ({'a': 1, 'y': 0}, pd.Series({'a': 1, 'y': 0})):
        with pytest.raises(SdfmError) as excinfo:
            client.tabicl(value, target='y', task='classification')
        assert 'pd.DataFrame([row])' in str(excinfo.value)


def test_frames_and_subclasses_are_still_accepted():
    r"""The guard must not reject what already worked. A ``DataFrame``
    subclass is a legitimate frame and passes ``isinstance``.
    """

    class _MyFrame(pd.DataFrame):
        pass

    adapter = _CapturingAdapter('tabicl', TabICLRequest, pd.DataFrame())
    client = _client_with(adapter)
    context = _MyFrame({'a': [1.0, 2.0], 'y': [0, 1]})

    client.tabicl(context, target='y', task='classification').predict(
        _MyFrame({'a': [3.0]}))
    assert adapter.captured is not None

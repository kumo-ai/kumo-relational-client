# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nemotron_structured import (
    RelationalModel,
    StructuredClient,
    TabularModel,
)
from nemotron_structured.base import ModelAdapter, ModelCapabilities
from nemotron_structured.errors import StructuredError
from nemotron_structured.requests import (
    NemotronRelationalRequest,
    NemotronRelationalTaskRequest,
    NemotronTabularRequest,
)


class _CapturingAdapter(ModelAdapter):
    """Records the request it receives and returns a fixed DataFrame, so a
    handle can be driven end-to-end through the real StructuredClient
    dispatch without a live NIM."""

    def __init__(self, name, request_type, result):
        self.name = name
        self.request_type = request_type
        self.result = result
        self.captured = None

    def capabilities(self):
        return ModelCapabilities(
            model=self.name, request_type=self.request_type.__name__
        )

    def predict(self, transport, request):
        self.captured = request
        return self.result


def _client_with(adapter) -> StructuredClient:
    client = StructuredClient(url='http://nim.test')
    client._register(adapter)
    return client


def test_rfm_returns_bound_handle():
    client = StructuredClient(url='http://nim.test')
    handle = client.relational('my-graph')
    assert isinstance(handle, RelationalModel)


def test_rfm_handle_end_to_end_through_client():
    result = pd.DataFrame({'ENTITY': [1], 'PREDICTION': [0.5]})
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, result
    )
    client = _client_with(adapter)

    out = client.relational('my-graph').predict(
        'PREDICT x FOR EACH t.id',
        [1, 2, 3],
        run_mode='best',
        anchor_time='2025-01-01',
    )

    assert out is result
    req = adapter.captured
    assert isinstance(req, NemotronRelationalRequest)
    assert req.model == 'nemotron-relational'
    assert req.graph == 'my-graph'
    assert req.query == 'PREDICT x FOR EACH t.id'
    assert req.indices == [1, 2, 3]
    assert req.run_mode == 'best'
    assert req.options == {'anchor_time': '2025-01-01'}


def test_rfm_handle_defaults_are_minimal():
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    client.relational('g').predict('PREDICT x')

    req = adapter.captured
    assert req.indices is None
    assert req.run_mode == 'fast'
    assert req.options == {}


@pytest.mark.parametrize(
    'run_mode',
    ['turbo', 3, None, np.array(['fast']), pd.NA],
)
def test_rfm_handle_names_allowed_run_modes(run_mode):
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    with pytest.raises(
        StructuredError,
        match="run_mode must be one of 'debug', 'fast', 'normal', 'best'",
    ) as exc:
        client.relational('g').predict('PREDICT x', run_mode=run_mode)

    assert exc.value.code == 'INVALID_REQUEST'
    assert adapter.captured is None


def test_rfm_handle_preserves_debug_run_mode():
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    client.relational('g').predict('PREDICT x', run_mode='debug')

    assert adapter.captured.run_mode == 'debug'


def test_rfm_task_handle_names_allowed_run_modes():
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalTaskRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    with pytest.raises(
        StructuredError, match=r"run_mode must be one of.*'best'"
    ):
        client.relational('g').predict_task(
            pd.DataFrame({'ENTITY': [1], 'TARGET': [2]}),
            pd.DataFrame({'ENTITY': [1]}),
            task_type='regression',
            entity_table='users',
            run_mode='turbo',
        )

    assert adapter.captured is None


@pytest.mark.parametrize('indices', ['payment-1', b'payment-1', 42, 4.2])
def test_rfm_handle_rejects_scalar_indices(indices):
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    with pytest.raises(
        StructuredError,
        match=(
            r'indices must be a list-like collection of entity IDs.*'
            r'for one entity ID, pass indices='
        ),
    ) as exc:
        client.relational('g').predict('PREDICT x', indices=indices)

    assert exc.value.code == 'INVALID_REQUEST'
    assert f'{type(indices).__name__} ({indices!r})' in str(exc.value)
    assert f'indices=[{indices!r}]' in str(exc.value)
    assert adapter.captured is None


@pytest.mark.parametrize(
    'indices',
    [
        [1, 2],
        (1, 2),
        np.array([1, 2]),
        pd.Series([1, 2]),
        pd.Index([1, 2]),
    ],
)
def test_rfm_handle_accepts_list_like_indices(indices):
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    client.relational('g').predict('PREDICT x', indices=indices)

    assert adapter.captured.indices is indices


def test_rfm_handle_only_forwards_set_options():
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    client.relational('g').predict(
        'PREDICT x',
        inference_config={'num_estimators': 2},
        return_embeddings=True,
    )

    assert adapter.captured.options == {
        'inference_config': {'num_estimators': 2},
        'return_embeddings': True,
    }


@pytest.mark.parametrize('random_seed', ['forty-two', 4.2, True, -1])
def test_rfm_handle_rejects_malformed_random_seed(random_seed):
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    with pytest.raises(
        StructuredError,
        match='random_seed must be None or a non-negative integer',
    ) as exc:
        client.relational('g').predict('PREDICT x', random_seed=random_seed)

    assert exc.value.code == 'INVALID_REQUEST'
    assert adapter.captured is None


@pytest.mark.parametrize('random_seed', ['forty-two', -1])
def test_rfm_task_handle_rejects_malformed_random_seed(random_seed):
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalTaskRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    with pytest.raises(
        StructuredError,
        match='random_seed must be None or a non-negative integer',
    ):
        client.relational('g').predict_task(
            pd.DataFrame({'ENTITY': [1], 'TARGET': [2]}),
            pd.DataFrame({'ENTITY': [1]}),
            task_type='regression',
            entity_table='users',
            random_seed=random_seed,
        )

    assert adapter.captured is None


@pytest.mark.parametrize(
    'random_seed',
    [np.int64(42), pd.Series([7], dtype='Int64').iloc[0]],
)
def test_rfm_handle_normalizes_integral_random_seed(random_seed):
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    client.relational('g').predict('PREDICT x', random_seed=random_seed)

    assert adapter.captured.options['random_seed'] == int(random_seed)
    assert type(adapter.captured.options['random_seed']) is int


def test_rfm_task_handle_normalizes_integral_random_seed():
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalTaskRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    client.relational('g').predict_task(
        pd.DataFrame({'ENTITY': [1], 'TARGET': [2]}),
        pd.DataFrame({'ENTITY': [1]}),
        task_type='regression',
        entity_table='users',
        random_seed=np.int32(9),
    )

    assert adapter.captured.options['random_seed'] == 9
    assert type(adapter.captured.options['random_seed']) is int


@pytest.mark.parametrize(
    ('option', 'value'),
    [
        ('return_embeddings', 'yes'),
        ('return_embeddings', 1),
        ('use_prediction_time', 'yes'),
        ('use_prediction_time', 0),
    ],
)
def test_rfm_handle_rejects_non_boolean_options(option, value):
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    with pytest.raises(
        StructuredError, match=rf'{option} must be a bool'
    ) as exc:
        client.relational('g').predict('PREDICT x', **{option: value})

    assert exc.value.code == 'INVALID_REQUEST'
    assert adapter.captured is None


@pytest.mark.parametrize('option', ['return_embeddings', 'use_prediction_time'])
def test_rfm_task_handle_rejects_non_boolean_options(option):
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalTaskRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    with pytest.raises(StructuredError, match=rf'{option} must be a bool'):
        client.relational('g').predict_task(
            pd.DataFrame({'ENTITY': [1], 'TARGET': [2]}),
            pd.DataFrame({'ENTITY': [1]}),
            task_type='regression',
            entity_table='users',
            **{option: 'yes'},
        )

    assert adapter.captured is None


@pytest.mark.parametrize('option', ['return_embeddings', 'use_prediction_time'])
@pytest.mark.parametrize('value', [np.bool_(True), pd.Series([False]).iloc[0]])
def test_rfm_handle_normalizes_numpy_boolean_options(option, value):
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    client.relational('g').predict('PREDICT x', **{option: value})

    assert adapter.captured.options[option] is bool(value)


def test_rfm_task_handle_normalizes_numpy_boolean_options():
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalTaskRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    client.relational('g').predict_task(
        pd.DataFrame({'ENTITY': [1], 'TARGET': [2]}),
        pd.DataFrame({'ENTITY': [1]}),
        task_type='regression',
        entity_table='users',
        return_embeddings=np.bool_(True),
        use_prediction_time=np.bool_(False),
    )

    assert adapter.captured.options['return_embeddings'] is True
    assert adapter.captured.options['use_prediction_time'] is False


def test_rfm_handle_can_silence_progress_output():
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    client.relational('g').predict('PREDICT x', verbose=False)

    assert adapter.captured.options == {'verbose': False}


def test_rfm_handle_forwards_engine_arguments_it_does_not_name():
    r"""The wrapper enumerates the engine's keywords, so anything it has not
    listed -- including whatever the engine gains next -- would otherwise be a
    ``TypeError`` with no way through.
    """
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    client.relational('g').predict('PREDICT x', some_new_engine_knob=7)

    assert adapter.captured.options == {'some_new_engine_knob': 7}


def test_rfm_task_handle_exposes_link_prediction_and_column_exclusion():
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalTaskRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    client.relational('g').predict_task(
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
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    client.relational('g').predict(
        'PREDICT x', [1], run_mode='normal', anchor_time='2025-06-01'
    )
    via_handle = adapter.captured

    typed = NemotronRelationalRequest(
        graph='g',
        query='PREDICT x',
        indices=[1],
        run_mode='normal',
        options={'anchor_time': '2025-06-01'},
    )

    assert via_handle == typed


def test_tabicl_returns_bound_handle():
    client = StructuredClient(url='http://nim.test')
    handle = client.tabular(
        pd.DataFrame({'y': [0, 1]}), target='y', task='classification'
    )
    assert isinstance(handle, TabularModel)


def test_tabicl_handle_end_to_end_through_client():
    result = pd.DataFrame({'prediction': [1]})
    adapter = _CapturingAdapter(
        'nemotron-tabular', NemotronTabularRequest, result
    )
    client = _client_with(adapter)

    ctx = pd.DataFrame({'x': [1, 2], 'y': [0, 1]})
    rows = pd.DataFrame({'x': [3]})
    out = client.tabular(ctx, target='y', task='classification').predict(
        rows, positive_class='1'
    )

    assert out is result
    req = adapter.captured
    assert isinstance(req, NemotronTabularRequest)
    assert req.model == 'nemotron-tabular'
    assert req.context.equals(ctx)
    assert req.predict.equals(rows)
    assert req.task == 'classification'
    assert req.target == 'y'
    assert req.positive_class == '1'


def test_tabicl_handle_defaults_are_minimal():
    adapter = _CapturingAdapter(
        'nemotron-tabular', NemotronTabularRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    ctx = pd.DataFrame({'x': [1], 'y': [0]})
    client.tabular(ctx, target='y', task='regression').predict(
        pd.DataFrame({'x': [2]})
    )

    req = adapter.captured
    assert req.outputs == ['prediction']
    assert req.positive_class is None
    assert req.max_results is None


def test_handle_still_dispatches_by_request_type():
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronTabularRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    from nemotron_structured import StructuredError

    with pytest.raises(StructuredError):
        client.relational('g').predict('PREDICT x')


def test_public_predict_is_not_exposed():
    client = StructuredClient(url='http://nim.test')
    assert not hasattr(client, 'predict')
    assert hasattr(client, 'relational')
    assert hasattr(client, 'tabular')


def test_rfm_handle_forwards_explain():
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    client.relational('g').predict('PREDICT x FOR t.id=1', explain=True)
    assert adapter.captured.explain is True

    client.relational('g').predict('PREDICT x FOR t.id=1')
    assert adapter.captured.explain is False


def test_rfm_handle_forwards_num_neighbors_and_num_hops():
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    client.relational('g').predict(
        'PREDICT x', num_neighbors=[8, 8], num_hops=3
    )

    assert adapter.captured.options == {
        'num_neighbors': [8, 8],
        'num_hops': 3,
    }


@pytest.mark.parametrize(
    ('option', 'value', 'expected'),
    [
        ('num_hops', 'two', 'integer between 1 and 6'),
        ('num_hops', 0, 'integer between 1 and 6'),
        ('num_hops', 7, 'integer between 1 and 6'),
        ('num_hops', True, 'integer between 1 and 6'),
        ('max_pq_iterations', 'ten', 'integer greater than or equal to 1'),
        ('max_pq_iterations', 0, 'integer greater than or equal to 1'),
        ('lag_timesteps', 'one', 'integer greater than or equal to 0'),
        ('lag_timesteps', -1, 'integer greater than or equal to 0'),
    ],
)
def test_rfm_handle_rejects_malformed_integer_options(
    option,
    value,
    expected,
):
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    with pytest.raises(
        StructuredError, match=rf'{option} must be an {expected}'
    ) as exc:
        client.relational('g').predict('PREDICT x', **{option: value})

    assert exc.value.code == 'INVALID_REQUEST'
    assert adapter.captured is None


def test_rfm_task_handle_rejects_malformed_num_hops():
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalTaskRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    with pytest.raises(
        StructuredError, match=r'num_hops.*integer between 1 and 6'
    ):
        client.relational('g').predict_task(
            pd.DataFrame({'ENTITY': [1], 'TARGET': [2]}),
            pd.DataFrame({'ENTITY': [1]}),
            task_type='regression',
            entity_table='users',
            num_hops='two',
        )

    assert adapter.captured is None


@pytest.mark.parametrize(
    ('option', 'value'),
    [
        ('num_hops', np.int64(2)),
        ('max_pq_iterations', pd.Series([3], dtype='Int64').iloc[0]),
        ('lag_timesteps', np.int32(1)),
    ],
)
def test_rfm_handle_normalizes_integral_options(option, value):
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    client.relational('g').predict('PREDICT x', **{option: value})

    assert adapter.captured.options[option] == int(value)
    assert type(adapter.captured.options[option]) is int


def test_rfm_task_handle_normalizes_integral_num_hops():
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalTaskRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    client.relational('g').predict_task(
        pd.DataFrame({'ENTITY': [1], 'TARGET': [2]}),
        pd.DataFrame({'ENTITY': [1]}),
        task_type='regression',
        entity_table='users',
        num_hops=np.int64(2),
    )

    assert adapter.captured.options['num_hops'] == 2
    assert type(adapter.captured.options['num_hops']) is int


@pytest.mark.parametrize(
    'num_neighbors',
    [[-1], ['eight'], [1.5], [True], [1] * 7, (8, 8)],
)
def test_rfm_handle_rejects_malformed_num_neighbors(num_neighbors):
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    with pytest.raises(
        StructuredError,
        match=(
            'num_neighbors must be None or a list of at most 6 non-negative '
            'integers'
        ),
    ) as exc:
        client.relational('g').predict('PREDICT x', num_neighbors=num_neighbors)

    assert exc.value.code == 'INVALID_REQUEST'
    assert adapter.captured is None


def test_rfm_task_handle_rejects_malformed_num_neighbors():
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalTaskRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    with pytest.raises(
        StructuredError, match=r'num_neighbors.*non-negative integers'
    ):
        client.relational('g').predict_task(
            pd.DataFrame({'ENTITY': [1], 'TARGET': [2]}),
            pd.DataFrame({'ENTITY': [1]}),
            task_type='regression',
            entity_table='users',
            num_neighbors=[-1],
        )

    assert adapter.captured is None


@pytest.mark.parametrize(
    'num_neighbors',
    [
        None,
        [],
        [0],
        [np.int64(8)],
        [pd.Series([8], dtype='Int64').iloc[0]],
    ],
)
def test_rfm_handle_preserves_supported_num_neighbors(num_neighbors):
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    client.relational('g').predict('PREDICT x', num_neighbors=num_neighbors)

    expected = (
        None if num_neighbors is None else [int(item) for item in num_neighbors]
    )
    assert adapter.captured.options['num_neighbors'] == expected
    if expected:
        assert type(adapter.captured.options['num_neighbors'][0]) is int


def test_rfm_handle_forwards_explain_config():
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    cfg = {'skip_summary': True}
    client.relational('g').predict('PREDICT x FOR t.id=1', explain=cfg)
    assert adapter.captured.explain == cfg


def test_tabicl_handle_forwards_request_id():
    adapter = _CapturingAdapter(
        'nemotron-tabular', NemotronTabularRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    ctx = pd.DataFrame({'x': [1], 'y': [0]})
    client.tabular(ctx, target='y', task='classification').predict(
        pd.DataFrame({'x': [2]}), request_id='trace-123'
    )

    assert adapter.captured.request_id == 'trace-123'


def test_rfm_handle_forwards_batch_size():
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    client.relational('g').predict(
        'PREDICT x FOR EACH t.id',
        list(range(1500)),
        batch_size='max',
        num_retries=2,
    )

    req = adapter.captured
    assert req.batch_size == 'max'
    assert req.num_retries == 2


def test_rfm_handle_batch_defaults_off():
    """batch_size defaults to None, not 'max': an unqualified predict must not
    enter batch mode. The adapter branches on ``batch_size is not None``, so
    flipping this default silently changes how every existing call executes --
    this makes such a flip a visible, deliberate edit."""
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    client.relational('g').predict('PREDICT x')

    assert adapter.captured.batch_size is None
    assert adapter.captured.num_retries == 1


def test_rfm_handle_predict_task_end_to_end_through_client():
    result = pd.DataFrame({'ENTITY': [3], 'PREDICTION': ['pro']})
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalTaskRequest, result
    )
    client = _client_with(adapter)

    context = pd.DataFrame({'ENTITY': [1, 2], 'TARGET': ['free', 'pro']})
    predict = pd.DataFrame({'ENTITY': [3]})
    out = client.relational('my-graph').predict_task(
        context=context,
        predict=predict,
        task_type='multiclass_classification',
        entity_table='users',
        run_mode='best',
        num_neighbors=[8, 8],
    )

    assert out is result
    req = adapter.captured
    assert isinstance(req, NemotronRelationalTaskRequest)
    assert req.model == 'nemotron-relational'
    assert req.graph == 'my-graph'
    assert req.context is context
    assert req.predict is predict
    assert req.task_type == 'multiclass_classification'
    assert req.entity_table == 'users'
    assert req.run_mode == 'best'
    assert req.options == {'num_neighbors': [8, 8]}


def test_rfm_handle_predict_task_defaults_are_minimal():
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalTaskRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    client.relational('g').predict_task(
        context=pd.DataFrame({'ENTITY': [1], 'TARGET': ['a']}),
        predict=pd.DataFrame({'ENTITY': [2]}),
        task_type='regression',
        entity_table='users',
    )

    req = adapter.captured
    assert req.entity_column == 'ENTITY'
    assert req.target_column == 'TARGET'
    assert req.time_column is None
    assert req.num_forecasts == 1
    assert req.step_size is None
    assert req.run_mode == 'fast'
    assert req.options == {}


def test_rfm_handle_predict_task_only_forwards_set_options():
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalTaskRequest, pd.DataFrame()
    )
    client = _client_with(adapter)

    client.relational('g').predict_task(
        context=pd.DataFrame({'ENTITY': [1], 'TARGET': ['a']}),
        predict=pd.DataFrame({'ENTITY': [2]}),
        task_type='regression',
        entity_table='users',
        inference_config={'num_estimators': 2},
    )

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
    """These are the shapes a pandas user reaches for before building a frame.
    Each used to reach ``frame.columns`` inside the adapter and raise
    ``AttributeError``, which ``except StructuredError`` does not catch and which
    names neither the argument nor the type it should have been.
    """
    client = StructuredClient(url='http://nim.test')
    with pytest.raises(StructuredError) as excinfo:
        client.tabular(value, target='y', task='classification')
    assert excinfo.value.code == 'INVALID_REQUEST'
    assert 'context must be a pandas DataFrame' in str(excinfo.value)
    assert type(value).__name__ in str(excinfo.value)


@pytest.mark.parametrize('value', _NOT_A_FRAME)
def test_tabicl_handle_rejects_a_non_frame_predict(value):
    adapter = _CapturingAdapter(
        'nemotron-tabular', NemotronTabularRequest, pd.DataFrame()
    )
    client = _client_with(adapter)
    handle = client.tabular(
        pd.DataFrame({'a': [1.0], 'y': [0]}), target='y', task='classification'
    )

    with pytest.raises(StructuredError) as excinfo:
        handle.predict(value)
    assert excinfo.value.code == 'INVALID_REQUEST'
    assert 'predict must be a pandas DataFrame' in str(excinfo.value)


@pytest.mark.parametrize('argument', ['context', 'predict'])
def test_rfm_handle_predict_task_rejects_a_non_frame(argument):
    adapter = _CapturingAdapter(
        'nemotron-relational', NemotronRelationalTaskRequest, pd.DataFrame()
    )
    client = _client_with(adapter)
    frames = {
        'context': pd.DataFrame({'ENTITY': [1], 'TARGET': ['a']}),
        'predict': pd.DataFrame({'ENTITY': [2]}),
    }
    frames[argument] = {'ENTITY': [1]}

    with pytest.raises(StructuredError) as excinfo:
        client.relational('g').predict_task(
            **frames, task_type='regression', entity_table='users'
        )
    assert excinfo.value.code == 'INVALID_REQUEST'
    assert f'{argument} must be a pandas DataFrame' in str(excinfo.value)


def test_a_dict_or_series_is_told_how_to_become_a_row():
    client = StructuredClient(url='http://nim.test')
    for value in ({'a': 1, 'y': 0}, pd.Series({'a': 1, 'y': 0})):
        with pytest.raises(StructuredError) as excinfo:
            client.tabular(value, target='y', task='classification')
        assert 'pd.DataFrame([row])' in str(excinfo.value)


def test_frames_and_subclasses_are_still_accepted():
    r"""The guard must not reject what already worked. A ``DataFrame``
    subclass is a legitimate frame and passes ``isinstance``.
    """

    class _MyFrame(pd.DataFrame):
        pass

    adapter = _CapturingAdapter(
        'nemotron-tabular', NemotronTabularRequest, pd.DataFrame()
    )
    client = _client_with(adapter)
    context = _MyFrame({'a': [1.0, 2.0], 'y': [0, 1]})

    client.tabular(context, target='y', task='classification').predict(
        _MyFrame({'a': [3.0]})
    )
    assert adapter.captured is not None

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
from kumorfm.api.pquery import ValidatedPredictiveQuery
from kumorfm.api.pquery.AST import (
    Aggregation,
    Column,
    Condition,
    Constant,
    DateOffsetRange,
    Filter,
    LogicalOperation,
)
from kumorfm.api.typing import BoolOp, Dtype

import kumorfm.kumolib as kumolib
from kumorfm.rfm import Graph
from kumorfm.rfm.backend.local import LocalSampler
from kumorfm.rfm.base.utils import Timestamp


def test_kumolib_sampler_validation() -> None:
    node_types = ['a', 'b']
    edge_types = [('a', 'to', 'b'), ('b', 'rev_to', 'a')]
    colptr = {
        'a__to__b': np.array([0, 1, 2, 0, 1, 2]),
        'b__rev_to__a': np.array([0, 1, 2, 0, 1, 2]),
    }
    row = {
        'a__to__b': np.array([0, 1, 1, 0]),
        'b__rev_to__a': np.array([0, 1, 1, 0]),
    }
    node_time = {
        'a': np.array([1, 2]),
        'b': np.array([1, 2]),
    }
    num_neighbors = {
        'a__to__b': np.array([3, 3]),
        'b__rev_to__a': np.array([3, 3]),
    }
    seed_node = np.array([0, 1])
    seed_time = np.array([0, 1])
    with pytest.raises(RuntimeError, match='is not 1D'):
        sampler = kumolib.NeighborSampler(node_types, edge_types, {
            'a__to__b': 'invalid',
            'b__to__a': None
        }, row, node_time, 123)

    with pytest.raises(RuntimeError, match='invalid'):
        sampler = kumolib.NeighborSampler(node_types, edge_types, colptr, row,
                                          node_time, 123)
        sampler.sample({'invalid': 1}, {}, 'a', seed_node, seed_time)

    with pytest.raises(RuntimeError, match='Invalid entity table'):
        sampler = kumolib.NeighborSampler(node_types, edge_types, colptr, row,
                                          node_time, 123)
        sampler.sample(num_neighbors, {}, 'k', seed_node, seed_time)

    with pytest.raises(RuntimeError, match='size mismatch'):
        sampler = kumolib.NeighborSampler(node_types, edge_types, colptr, row,
                                          node_time, 123)
        sampler.sample(num_neighbors, {}, 'a', seed_node, np.array([0]))

    with pytest.raises(RuntimeError, match='same edge type'):
        sampler = kumolib.NeighborSampler(node_types, edge_types, colptr, row,
                                          node_time, 123)
        sampler.sample(num_neighbors,
                       {'a__to__b': np.array([[0, 5], [None, 0]])}, 'a',
                       seed_node, seed_time)


def test_kumolib_sampler_temporal() -> None:
    node_types = ['A', 'B']
    edge_types = [('A', 'to', 'B'), ('B', 'to', 'A')]
    colptr = {
        'A__to__B': np.array([0, 2, 4]),
        'B__to__A': np.array([0, 1, 2, 3]),
    }
    row = {
        'A__to__B': np.array([0, 2, 0, 2]),
        'B__to__A': np.array([0, 1, 1]),
    }
    node_time = {
        'A': np.array([1, 0, 3]),
        'B': np.array([2, 1]),
    }

    sampler = kumolib.NeighborSampler(node_types, edge_types, colptr, row,
                                      node_time)

    num_neighbors = {
        'A__to__B': [1, 1],
        'B__to__A': [2, 1],
    }
    entity_table = 'A'
    seed_node = np.array([0, 1, 2])
    seed_time = np.array([3, 1, 3])
    (
        row,
        col,
        node_id,
        batch,
        num_sampled_nodes,
        num_sampled_edges,
    ) = sampler.sample(num_neighbors, {}, entity_table, seed_node, seed_time)
    assert row['A__to__B'].shape[0] == sum(num_sampled_edges['A__to__B'])
    assert row['B__to__A'].shape[0] == sum(num_sampled_edges['B__to__A'])
    assert col['A__to__B'].shape[0] == sum(num_sampled_edges['A__to__B'])
    assert col['B__to__A'].shape[0] == sum(num_sampled_edges['B__to__A'])
    assert node_id['A'].shape[0] == sum(num_sampled_nodes['A'])
    assert node_id['B'].shape[0] == sum(num_sampled_nodes['B'])
    assert row['A__to__B'].tolist() == [3, 4, 2]
    assert row['B__to__A'].tolist() == [0, 1, 2]
    assert col['A__to__B'].tolist() == [0, 1, 2]
    assert col['B__to__A'].tolist() == [0, 1, 2]
    assert node_id['A'].tolist() == [0, 1, 2, 2, 0]
    assert node_id['B'].tolist() == [0, 1, 1]
    assert batch['A'].tolist() == [0, 1, 2, 0, 1]
    assert batch['B'].tolist() == [0, 1, 2]
    assert num_sampled_nodes['A'].tolist() == [3, 0, 2]
    assert num_sampled_nodes['B'].tolist() == [0, 3, 0]
    assert num_sampled_edges['A__to__B'].tolist() == [0, 3]
    assert num_sampled_edges['B__to__A'].tolist() == [3, 0]


def test_kumolib_sampler_static() -> None:
    node_types = ['A', 'B']
    edge_types = [('A', 'to', 'B'), ('B', 'to', 'A')]
    colptr = {
        'A__to__B': np.array([0, 2, 4]),
        'B__to__A': np.array([0, 1, 2, 3]),
    }
    row = {
        'A__to__B': np.array([0, 2, 0, 2]),
        'B__to__A': np.array([0, 1, 1]),
    }

    sampler = kumolib.NeighborSampler(node_types, edge_types, colptr, row, [])

    num_neighbors = {
        'A__to__B': np.array([1, -1]),
        'B__to__A': np.array([1, 1]),
    }
    seed_node = np.array([0, 1, 2])
    entity_table = 'A'
    (
        row,
        col,
        node_id,
        batch,
        num_sampled_nodes,
        num_sampled_edges,
    ) = sampler.sample(num_neighbors, {}, entity_table, seed_node,
                       np.array([]))
    # Outputs are non-deterministic but this specific
    # config is deterministic for this graph
    assert row['A__to__B'].tolist() == [0, 3, 4, 5, 6, 2]
    assert row['B__to__A'].tolist() == [0, 1, 2]
    assert col['A__to__B'].tolist() == [0, 0, 1, 1, 2, 2]
    assert col['B__to__A'].tolist() == [0, 1, 2]
    assert node_id['A'].tolist() == [0, 1, 2, 2, 0, 2, 0]
    assert node_id['B'].tolist() == [0, 1, 1]
    assert batch['A'].tolist() == [0, 1, 2, 0, 1, 1, 2]
    assert batch['B'].tolist() == [0, 1, 2]
    assert num_sampled_nodes['A'].tolist() == [3, 0, 4]
    assert num_sampled_nodes['B'].tolist() == [0, 3, 0]
    assert num_sampled_edges['A__to__B'].tolist() == [0, 6]
    assert num_sampled_edges['B__to__A'].tolist() == [3, 0]


def test_kumolib_sampler_temporal_range() -> None:
    node_types = ['A', 'B']
    edge_types = [('A', 'to', 'B'), ('B', 'to', 'A')]
    colptr = {
        'A__to__B': np.array([0, 2, 4]),
        'B__to__A': np.array([0, 1, 2, 3]),
    }
    row = {
        'A__to__B': np.array([0, 2, 0, 2]),
        'B__to__A': np.array([0, 1, 1]),
    }
    node_time = {
        'A': np.array([1, 0, 3]),
        'B': np.array([2, 1]),
    }

    sampler = kumolib.NeighborSampler(node_types, edge_types, colptr, row,
                                      node_time)

    sampling_time_offsets = {
        'A__to__B': np.array([[0, 1], [-1, 2]]),
        'B__to__A': [[None, 0], [0, 0]],
    }
    entity_table = 'A'
    seed_node = np.array([0, 1, 2])
    seed_time = np.array([3, 1, 3])
    (
        row,
        col,
        node_id,
        batch,
        num_sampled_nodes,
        num_sampled_edges,
    ) = sampler.sample({}, sampling_time_offsets, entity_table, seed_node,
                       seed_time)
    assert row['A__to__B'].shape[0] == sum(num_sampled_edges['A__to__B'])
    assert row['B__to__A'].shape[0] == sum(num_sampled_edges['B__to__A'])
    assert col['A__to__B'].shape[0] == sum(num_sampled_edges['A__to__B'])
    assert col['B__to__A'].shape[0] == sum(num_sampled_edges['B__to__A'])
    assert node_id['A'].shape[0] == sum(num_sampled_nodes['A'])
    assert node_id['B'].shape[0] == sum(num_sampled_nodes['B'])
    assert row['A__to__B'].tolist() == [3, 4, 5, 2]
    assert row['B__to__A'].tolist() == [0, 1, 2]
    assert col['A__to__B'].tolist() == [0, 1, 1, 2]
    assert col['B__to__A'].tolist() == [0, 1, 2]
    assert node_id['A'].tolist() == [0, 1, 2, 2, 0, 2]
    assert node_id['B'].tolist() == [0, 1, 1]
    assert batch['A'].tolist() == [0, 1, 2, 0, 1, 1]
    assert batch['B'].tolist() == [0, 1, 2]
    assert num_sampled_nodes['A'].tolist() == [3, 0, 3]
    assert num_sampled_nodes['B'].tolist() == [0, 3, 0]
    assert num_sampled_edges['A__to__B'].tolist() == [0, 4]
    assert num_sampled_edges['B__to__A'].tolist() == [3, 0]


def test_sample_entity_table_with_int_entity_ids() -> None:
    df_dict = {
        'USERS':
        pd.DataFrame({
            'USER_ID': [1, 2, 3, 4, 5],
            'AGE': [20, 30, 40, 50, 60],
        }),
        'ORDERS':
        pd.DataFrame({
            'ORDER_ID': range(10),
            'USER_ID': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            'AMOUNT': range(10),
            'TIME': pd.date_range('2025-01-01', periods=10, freq='D'),
        }),
    }
    graph = Graph.from_data(df_dict, verbose=False)
    sampler = LocalSampler(graph, verbose=False)
    df = sampler._sample_entity_table(
        table_name='USERS',
        columns={'USER_ID', 'AGE'},
        num_rows=100,
        entity_ids=[2, 4],
    )
    assert set(df['USER_ID'].tolist()) == {2, 4}
    assert len(df) == 2


def test_sample_entity_table_with_string_entity_ids() -> None:
    df_dict = {
        'USERS':
        pd.DataFrame({
            'USER_ID': ['alice', 'bob', 'charlie'],
            'AGE': [20, 30, 40],
        }),
        'ORDERS':
        pd.DataFrame({
            'ORDER_ID': range(3),
            'USER_ID': ['alice', 'bob', 'charlie'],
            'AMOUNT': [10, 20, 30],
        }),
    }
    graph = Graph.from_data(df_dict, verbose=False)
    sampler = LocalSampler(graph, verbose=False)

    df = sampler._sample_entity_table(
        table_name='USERS',
        columns={'USER_ID', 'AGE'},
        num_rows=100,
        entity_ids=['bob'],
    )
    assert df['USER_ID'].tolist() == ['bob']
    assert len(df) == 1


def test_sample_target() -> None:
    df_dict: dict[str, pd.DataFrame] = {}
    df_dict['USERS'] = pd.DataFrame({
        'USER_ID': [10, 10, 0],
        'AGE': [30, 30, 20],
    })
    df_dict['ORDERS'] = pd.DataFrame({
        'USER_ID': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'AMOUNT':
        range(18),
        'TIME':
        pd.date_range('2025-01-01', periods=18, unit='ns'),
    })
    df_dict['VIEWS'] = pd.DataFrame({
        'USER_ID': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'AMOUNT':
        range(18),
        'TIME':
        pd.date_range('2025-01-01', periods=18, unit='ns'),
    })

    graph = Graph.from_data(df_dict, verbose=False)
    sampler = LocalSampler(graph, verbose=False)

    query = ValidatedPredictiveQuery(
        target_ast=LogicalOperation(
            left=Condition(
                target=Aggregation(
                    aggr='SUM',
                    target=Column(fqn='ORDERS.AMOUNT'),
                    aggr_time_range=DateOffsetRange(2, 7),
                ),
                op='>',
                value=Constant(value='5', dtype_maybe=Dtype.int),
            ),
            bool_op=BoolOp.AND,
            right=Condition(
                target=Column(fqn='USERS.USER_ID'),
                op='=',
                value=Constant(value='0', dtype_maybe=Dtype.int),
            ),
        ),
        entity_ast=Filter(
            target=Column(fqn='USERS.USER_ID'),
            condition=LogicalOperation(
                left=Condition(
                    target=Aggregation(
                        aggr='SUM',
                        target=Column(fqn='ORDERS.AMOUNT'),
                        aggr_time_range=DateOffsetRange(None, -2),
                    ),
                    op='>',
                    value=Constant(value='5', dtype_maybe=Dtype.int),
                ),
                bool_op=BoolOp.OR,
                right=Condition(
                    target=Aggregation(
                        aggr='SUM',
                        target=Column(fqn='VIEWS.AMOUNT'),
                        aggr_time_range=DateOffsetRange(-7, 0),
                    ),
                    op='<=',
                    value=Constant(value='5', dtype_maybe=Dtype.int),
                ),
            ),
        ),
    )

    train, test = sampler.sample_target(
        query,
        num_train_examples=2,
        train_anchor_time=Timestamp('2025-01-08'),
        num_train_trials=20,
        num_test_examples=0,
        test_anchor_time=Timestamp('2025-01-06'),
        num_test_trials=20,
    )

    perm = train.entity_pkey.argsort()
    pd.testing.assert_series_equal(
        train.entity_pkey[perm].reset_index(drop=True),
        pd.Series([0, 10], name='USER_ID'),
    )
    pd.testing.assert_series_equal(
        train.anchor_time,
        pd.to_datetime(pd.Series(['2025-01-08'] * 2)).astype('datetime64[ns]'),
    )
    pd.testing.assert_series_equal(
        train.target[perm].reset_index(drop=True),
        pd.Series([True, False]),
    )
    assert len(test.entity_pkey) == 0
    assert len(test.anchor_time) == 0
    assert len(test.target) == 0

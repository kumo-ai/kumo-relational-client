# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sqlite3
from pathlib import Path

import pytest
from kumo_relational_engine.api.pquery import ValidatedPredictiveQuery
from kumo_relational_engine.rfm import Graph

pytest.importorskip(
    'adbc_driver_sqlite', reason="'sqlite' extension not installed"
)

from kumo_relational_engine.tfm import KumoTabular

BINARY_TEMPORAL = (
    'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id'
)
BINARY_STATIC = 'PREDICT users.age > 30 FOR EACH users.user_id'
MULTICLASS_STATIC = 'PREDICT users.status FOR EACH users.user_id'
REGRESSION_TEMPORAL = (
    'PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.user_id'
)
REGRESSION_STATIC = 'PREDICT users.age FOR EACH users.user_id'

LINK_PREDICTION = (
    'PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) RANK TOP 5 '
    'FOR EACH users.user_id'
)
MULTILABEL_CLASSIFY = (
    'PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) CLASSIFY '
    'FOR EACH users.user_id'
)
FORECAST = (
    'PREDICT SUM(orders.amount, 0, 7, days) FORECAST 4 TIMEFRAMES '
    'FOR EACH users.user_id'
)
ASSUMING = (
    'PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.user_id '
    'ASSUMING COUNT(orders.*, 0, 30, days) > 3'
)


def _create_database(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.execute(
        'CREATE TABLE users ('
        '  user_id INTEGER PRIMARY KEY,'
        '  age INTEGER,'
        '  status TEXT)'
    )
    connection.execute(
        'CREATE TABLE items (  item_id INTEGER PRIMARY KEY,  category TEXT)'
    )
    connection.execute(
        'CREATE TABLE orders ('
        '  order_id INTEGER PRIMARY KEY,'
        '  user_id INTEGER,'
        '  item_id INTEGER,'
        '  amount REAL,'
        '  ts TEXT)'
    )
    connection.executemany(
        'INSERT INTO users VALUES (?, ?, ?)',
        [(i, 20 + i % 40, 'ABC'[i % 3]) for i in range(50)],
    )
    connection.executemany(
        'INSERT INTO items VALUES (?, ?)',
        [(i, ['burger', 'pizza', 'fries'][i % 3]) for i in range(9)],
    )
    connection.executemany(
        'INSERT INTO orders VALUES (?, ?, ?, ?, ?)',
        [
            (i, i % 50, i % 9, 10.0 + i % 7, f'2024-01-{i % 28 + 1:02d}')
            for i in range(200)
        ],
    )
    connection.commit()
    connection.close()
    return path


@pytest.fixture(scope='module')
def shop_graph(tmp_path_factory: pytest.TempPathFactory) -> Graph:
    path = _create_database(tmp_path_factory.mktemp('tfm') / 'shop.db')
    return Graph.from_sqlite(
        path,
        tables=[
            dict(name='users', primary_key='user_id'),
            dict(name='items', primary_key='item_id'),
            dict(name='orders', primary_key='order_id', time_column='ts'),
        ],
        edges=[
            ('orders', 'user_id', 'users'),
            ('orders', 'item_id', 'items'),
        ],
        verbose=False,
    )


@pytest.mark.parametrize(
    'query',
    [
        pytest.param(BINARY_TEMPORAL, id='binary-temporal-aggregation'),
        pytest.param(BINARY_STATIC, id='binary-static-column'),
        pytest.param(MULTICLASS_STATIC, id='multiclass-categorical-column'),
        pytest.param(REGRESSION_TEMPORAL, id='regression-temporal-aggregation'),
        pytest.param(REGRESSION_STATIC, id='regression-numerical-column'),
    ],
)
def test_the_gate_passes_the_tasks_the_tabular_model_serves(
    shop_graph: Graph,
    query: str,
) -> None:
    parsed = KumoTabular(shop_graph)._parse_query(query)

    assert isinstance(parsed, ValidatedPredictiveQuery)


@pytest.mark.parametrize(
    ('query', 'expected'),
    [
        pytest.param(
            LINK_PREDICTION,
            'do not support ranking or link prediction',
            id='link-prediction-rank-top-k',
        ),
        pytest.param(
            MULTILABEL_CLASSIFY,
            'do not support multilabel tasks',
            id='multilabel-classify',
        ),
        pytest.param(
            FORECAST,
            'do not support forecasting',
            id='forecast-timeframes',
        ),
        pytest.param(
            ASSUMING,
            'do not support counterfactuals',
            id='assuming-counterfactual',
        ),
    ],
)
def test_the_gate_rejects_the_tasks_the_tabular_model_cannot_answer(
    shop_graph: Graph,
    query: str,
    expected: str,
) -> None:
    model = KumoTabular(shop_graph)

    with pytest.raises(ValueError, match=expected) as excinfo:
        model._parse_query(query)

    message = str(excinfo.value)
    assert 'Tabular foundation model queries' in message
    # The diagnostic names the tabular model, not the relational one, even
    # though the relational rules reject several of these shapes too.
    assert 'Foundation Model queries' not in message


@pytest.mark.parametrize(
    'query',
    [
        pytest.param(LINK_PREDICTION, id='link-prediction-rank-top-k'),
        pytest.param(FORECAST, id='forecast-timeframes'),
        pytest.param(ASSUMING, id='assuming-counterfactual'),
    ],
)
def test_the_gate_is_the_only_thing_rejecting_these(
    shop_graph: Graph,
    query: str,
) -> None:
    # Regression guard: each of these parses cleanly for the relational
    # model, so a failure here would mean the gate had stopped being the
    # reason the tabular pathway turns them away.
    from kumo_relational_engine.rfm.query_parser import parse_query_locally

    parsed = parse_query_locally(query, shop_graph._to_api_graph_definition())

    assert isinstance(parsed, ValidatedPredictiveQuery)


def test_kumo_tabular_validates_before_it_declines_to_predict(
    shop_graph: Graph,
) -> None:
    model = KumoTabular(shop_graph)

    with pytest.raises(NotImplementedError, match='land in a later release'):
        model.predict(BINARY_TEMPORAL, indices=[1, 2])

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from kumo_relational_engine.api.pquery import ValidatedPredictiveQuery
from kumo_relational_engine.api.typing import Stype
from kumo_relational_engine.rfm import Graph
from kumo_relational_engine.rfm.query_parser import parse_query_locally
from kumo_relational_engine.tfm import KumoTabular

BINARY_TEMPORAL = (
    'PREDICT COUNT(ORDERS.*, 0, 30, days) > 0 FOR EACH USERS.USER_ID'
)
BINARY_STATIC = 'PREDICT USERS.AGE > 30 FOR EACH USERS.USER_ID'
MULTICLASS_STATIC = 'PREDICT USERS.STATUS FOR EACH USERS.USER_ID'
REGRESSION_TEMPORAL = (
    'PREDICT SUM(ORDERS.AMOUNT, 0, 30, days) FOR EACH USERS.USER_ID'
)
REGRESSION_STATIC = 'PREDICT USERS.AGE FOR EACH USERS.USER_ID'

LINK_PREDICTION = (
    'PREDICT LIST_DISTINCT(ORDERS.ITEM_ID, 0, 30, days) RANK TOP 5 '
    'FOR EACH USERS.USER_ID'
)
MULTILABEL_CLASSIFY = (
    'PREDICT LIST_DISTINCT(ORDERS.ITEM_ID, 0, 30, days) CLASSIFY '
    'FOR EACH USERS.USER_ID'
)
FORECAST = (
    'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) FORECAST 4 TIMEFRAMES '
    'FOR EACH USERS.USER_ID'
)
ASSUMING = (
    'PREDICT SUM(ORDERS.AMOUNT, 0, 30, days) FOR EACH USERS.USER_ID '
    'ASSUMING COUNT(ORDERS.*, 0, 30, days) > 3'
)


@pytest.fixture()
def shop_graph() -> Graph:
    df_dict = {}
    df_dict['USERS'] = pd.DataFrame(
        {
            'USER_ID': list(range(50)),
            'AGE': [20 + i % 40 for i in range(50)],
            'STATUS': ['ABC'[i % 3] for i in range(50)],
        }
    )
    df_dict['ITEMS'] = pd.DataFrame(
        {
            'ITEM_ID': list(range(9)),
            'CATEGORY': [['burger', 'pizza', 'fries'][i % 3] for i in range(9)],
        }
    )
    df_dict['ORDERS'] = pd.DataFrame(
        {
            'ORDER_ID': list(range(200)),
            'USER_ID': [i % 50 for i in range(200)],
            'ITEM_ID': [i % 9 for i in range(200)],
            'AMOUNT': [10.0 + i % 7 for i in range(200)],
            'TIME': pd.to_datetime(
                [f'2024-01-{i % 28 + 1:02d}' for i in range(200)]
            ),
        }
    )

    graph = Graph.from_data(df_dict, verbose=False)
    graph['USERS']['AGE'].stype = Stype.numerical
    graph['ORDERS']['AMOUNT'].stype = Stype.numerical
    return graph


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
    # Reaching the not-implemented raise is how a query says it got through
    # the gate: `predict` validates first and declines only afterwards.
    with pytest.raises(NotImplementedError, match='land in a later release'):
        KumoTabular(shop_graph).predict(query, indices=[1, 2])


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
        model.predict(query)

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
    parsed = parse_query_locally(query, shop_graph._to_api_graph_definition())

    assert isinstance(parsed, ValidatedPredictiveQuery)


def test_an_already_validated_query_is_not_parsed_again(
    shop_graph: Graph,
) -> None:
    # A `ValidatedPredictiveQuery` has been through a validator already, so
    # `predict` takes it as given and goes straight to declining.
    query = parse_query_locally(
        LINK_PREDICTION, shop_graph._to_api_graph_definition()
    )

    with pytest.raises(NotImplementedError, match='land in a later release'):
        KumoTabular(shop_graph).predict(query)

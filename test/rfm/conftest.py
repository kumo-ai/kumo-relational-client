import os

import pandas as pd
import pytest
from kumoapi.pquery import ValidatedPredictiveQuery
from kumoapi.pquery.AST import (
    Aggregation,
    Column,
    Condition,
    Constant,
    DateOffsetRange,
)
from kumoapi.typing import (
    AggregationType,
    Dtype,
    MemberOp,
    ProblemType,
    RelOp,
    Stype,
)

from kumoai import rfm


@pytest.fixture(scope='session')
def rfm_init() -> None:
    rfm.init(
        url='https://rfm2-uat.staging-rfm.kumorfm.ai/api',
        api_key=os.environ['KUMO_RFM_STAGING_API_KEY'],
    )


@pytest.fixture(autouse=True)
def _rfm_init(request: pytest.FixtureRequest) -> None:
    r"""Lazily initializes RFM client."""
    if request.node.get_closest_marker('rfm_backend'):
        request.getfixturevalue('rfm_init')  # Session-wide RFM initialization.


@pytest.fixture()
def user_store_graph() -> rfm.Graph:
    df_dict = {}
    df_dict['USERS'] = pd.DataFrame({
        'USER_ID': [0, 1, 2, 3],
        'AGE': [20, 30, 40, float('NaN')],
        'GENDER': ['male', 'female', 'female', None],
        'STATUS': ['A', 'B', 'A', 'C'],
    })
    df_dict['ORDERS'] = pd.DataFrame({
        'ORDER_ID': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'USER_ID': [0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3],
        'STORE_ID': [0, 1, 0, 1, 2, 2, 0, 1, 2, 0, 1, 1, 2],
        'AMOUNT':
        [10, 15, float('NaN'), 20, 25, 30, 10, 25, 20, 10, 15, 15, 20],
        'CAT': [10, 15,
                float('NaN'), 20, 25, 30, 11, 26, 27, 28, 29, 32, 33],
        'TIME': [
            '2025-01-01',
            '2024-12-20',
            '2025-01-03',
            '2025-01-02',
            '2025-01-03',
            '2025-01-04',
            '2025-01-09',
            '2025-01-02',
            '2025-01-02',
            '2025-01-01',
            '2025-01-02',
            '2025-01-03',
            '2025-01-04',
        ],
    })
    df_dict['STORES'] = pd.DataFrame({
        'STORE_ID': [0, 1, 2],
        'CAT': ['burger', 'pizza', 'fries'],
    })

    graph = rfm.Graph.from_data(df_dict, verbose=False)
    graph['USERS']['AGE'].stype = Stype.numerical
    graph['ORDERS']['AMOUNT'].stype = Stype.numerical
    return graph


@pytest.fixture()
def string_user_graph() -> rfm.Graph:
    df_dict = {}
    df_dict['USERS'] = pd.DataFrame({
        'USER_ID': ['user_a', 'user_b', 'user_c', 'user_d'],
        'NAME': ['Alice', 'Bob', 'Charlie', 'David'],
    })
    df_dict['ORDERS'] = pd.DataFrame({
        'ORDER_ID': [0, 1, 2, 3, 4],
        'USER_ID': ['user_a', 'user_a', 'user_b', 'user_c', 'user_d'],
        'TIME': [
            '2024-01-01',
            '2025-01-02',
            '2025-01-03',
            '2025-01-04',
            '2025-01-05',
        ],
    })

    return rfm.Graph.from_data(df_dict, verbose=False)


@pytest.fixture()
def churn() -> ValidatedPredictiveQuery:
    return ValidatedPredictiveQuery(
        target_ast=Condition(
            target=Aggregation(
                aggr=AggregationType.COUNT,
                target=Column(fqn='ORDERS.*'),
                aggr_time_range=DateOffsetRange(0, 1),
            ),
            op=RelOp.EQ,
            value=0,
        ),
        entity_ast=Column(fqn='USERS.USER_ID'),
        rfm_entity_ids=Condition(
            target=Column(fqn='USERS.USER_ID'),
            op=MemberOp.IN,
            value=Constant.from_value([0, 1, 2, 3]),
        ),
    )


@pytest.fixture()
def ltv() -> ValidatedPredictiveQuery:
    return ValidatedPredictiveQuery(
        target_ast=Aggregation(
            aggr=AggregationType.SUM,
            target=Column(fqn='ORDERS.AMOUNT'),
            aggr_time_range=DateOffsetRange(0, 7),
        ),
        entity_ast=Column(fqn='USERS.USER_ID'),
        rfm_entity_ids=Condition(
            target=Column(fqn='USERS.USER_ID'),
            op=MemberOp.IN,
            value=Constant.from_value([0, 1, 2, 3]),
        ),
    )


@pytest.fixture()
def forecast() -> ValidatedPredictiveQuery:
    q = ValidatedPredictiveQuery(
        target_ast=Aggregation(
            aggr=AggregationType.SUM,
            target=Column(fqn='ORDERS.AMOUNT'),
            aggr_time_range=DateOffsetRange(0, 1),
        ),
        entity_ast=Column(fqn='USERS.USER_ID'),
        rfm_entity_ids=Condition(
            target=Column(fqn='USERS.USER_ID'),
            op=MemberOp.IN,
            value=Constant.from_value([0]),
        ),
        problem_type=ProblemType.FORECAST,
    )
    q.num_forecasts = 4
    return q


@pytest.fixture()
def forecast_single_step() -> ValidatedPredictiveQuery:
    q = ValidatedPredictiveQuery(
        target_ast=Aggregation(
            aggr=AggregationType.SUM,
            target=Column(fqn='ORDERS.AMOUNT'),
            aggr_time_range=DateOffsetRange(0, 1),
        ),
        entity_ast=Column(fqn='USERS.USER_ID'),
        rfm_entity_ids=Condition(
            target=Column(fqn='USERS.USER_ID'),
            op=MemberOp.IN,
            value=Constant.from_value([0]),
        ),
        problem_type=ProblemType.FORECAST,
    )
    q.num_forecasts = 1
    return q


@pytest.fixture()
def age() -> ValidatedPredictiveQuery:
    return ValidatedPredictiveQuery(
        target_ast=Column(fqn='USERS.AGE', stype_maybe=Stype.numerical),
        entity_ast=Column(fqn='USERS.USER_ID'),
        rfm_entity_ids=Condition(
            target=Column(fqn='USERS.USER_ID'),
            op=MemberOp.IN,
            value=Constant.from_value([2, 3]),
        ),
    )


@pytest.fixture()
def assuming() -> ValidatedPredictiveQuery:
    return ValidatedPredictiveQuery(
        target_ast=Aggregation(
            aggr=AggregationType.SUM,
            target=Column(fqn='ORDERS.AMOUNT'),
            aggr_time_range=DateOffsetRange(0, 7),
        ),
        entity_ast=Column(fqn='USERS.USER_ID'),
        rfm_entity_ids=Condition(
            target=Column(fqn='USERS.USER_ID'),
            op=MemberOp.IN,
            value=Constant.from_value([0, 1, 2, 3]),
        ),
        whatif_ast=Condition(
            target=Aggregation(
                aggr=AggregationType.SUM,
                target=Column(fqn='ORDERS.AMOUNT'),
                aggr_time_range=DateOffsetRange(0, 7),
            ),
            op=RelOp.GEQ,
            value=Constant(value='10', dtype_maybe=Dtype.int),
        ),
    )


@pytest.fixture()
def ltv_offset(request: pytest.FixtureRequest) -> ValidatedPredictiveQuery:
    start, steps = request.param if hasattr(request, 'param') else (0, 1)
    return ValidatedPredictiveQuery(
        target_ast=Aggregation(
            aggr=AggregationType.SUM,
            target=Column(fqn='ORDERS.AMOUNT'),
            aggr_time_range=DateOffsetRange(start, start + steps),
        ),
        entity_ast=Column(fqn='USERS.USER_ID'),
        rfm_entity_ids=Condition(
            target=Column(fqn='USERS.USER_ID'),
            op=MemberOp.IN,
            value=Constant.from_value([0, 1, 2, 3]),
        ),
    )

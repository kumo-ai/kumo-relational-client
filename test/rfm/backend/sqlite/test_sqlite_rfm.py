import pytest
from kumoapi.pquery import ValidatedPredictiveQuery
from kumoapi.pquery.AST import (
    Aggregation,
    Column,
    Condition,
    Constant,
    DateOffsetRange,
)
from kumoapi.typing import AggregationType, Dtype

from kumoai.rfm import Graph, KumoRFM

try:
    from kumoai.rfm.backend.sqlite import Connection, SQLiteTable
except ImportError:
    pytest.skip("'sqlite' extension not installed", allow_module_level=True)


def test_rfm(connection: Connection) -> None:
    graph = Graph(
        tables=[
            SQLiteTable(
                connection,
                name='USER',
                source_name='USERS',
                columns=[
                    dict(name='user_id', expr="2*USER_ID"),
                    'IS_FLAG',
                    'AGE',
                    'GENDER',
                    dict(name='date_of_birth', expr="DOB"),
                ],
                primary_key='user_id',
                time_column='date_of_birth',
            ),
            SQLiteTable(
                connection,
                name='ORDERS',
                columns=[
                    dict(name='user_id', expr="2*USER_ID"),
                    'ITEM_ID',
                    dict(name='normalized_price', expr="PRICE/100"),
                    dict(name='order_date', expr="DATE"),
                ],
                time_column='order_date',
            ),
            SQLiteTable(
                connection,
                name='ITEM',
                source_name='ITEMS',
                columns=[
                    'ITEM_ID',
                    dict(name='CAT', expr="CONCAT(CATEGORY, '-', CATEGORY)"),
                ],
            ),
        ],
        edges=[
            ('ORDERS', 'user_id', 'USER'),
            ('ORDERS', 'ITEM_ID', 'ITEM'),
        ],
    )

    with pytest.warns(UserWarning, match="Missing 1 index in 1 table"):
        model = KumoRFM(graph)

    query = ValidatedPredictiveQuery(
        target_ast=Aggregation(
            aggr=AggregationType.COUNT,
            target=Column(fqn='ORDERS.*'),
            aggr_time_range=DateOffsetRange(0, 1),
        ),
        entity_ast=Column(fqn='USER.user_id'),
        rfm_entity_ids=Condition(
            target=Column(fqn='USER.user_id'),
            op='=',
            value=Constant(value='0', dtype_maybe=Dtype.int),
        ),
    )

    task_table = model._get_task_table(
        query=query,
        indices=[0, 2, 4, 6],
        anchor_time=model._sampler.get_max_time(),
    )
    model._get_context(task_table)

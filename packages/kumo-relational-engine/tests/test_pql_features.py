# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sqlite3
from pathlib import Path

import pytest
from kumo_relational_engine.api.graph import GraphDefinition
from kumo_relational_engine.api.pquery.AST import (
    Aggregation,
    ASTNode,
    Column,
    Filter,
    Join,
)
from kumo_relational_engine.pql.feature import (
    ParsedFeature,
    parse_feature,
    validate_features,
)
from kumo_relational_engine.rfm import Graph

pytest.importorskip(
    'adbc_driver_sqlite', reason="'sqlite' extension not installed"
)

ENTITY = 'users.user_id'

WINDOWED_AGGREGATION = 'SUM(orders.price, -90, 0, days)'
FILTERED_AGGREGATION = (
    "COUNT(orders.* WHERE orders.status = 'complete', -30, 0, days)"
)
BARE_COLUMN = 'users.age'


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
        '  price REAL,'
        '  status TEXT,'
        '  ts TEXT)'
    )
    # An island: no edge reaches it from `users`, so a feature that reads it
    # is unreachable from the entity rather than merely misspelled.
    connection.execute(
        'CREATE TABLE tickets (  ticket_id INTEGER PRIMARY KEY,  note TEXT)'
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
        'INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)',
        [
            (
                i,
                i % 50,
                i % 9,
                10.0 + i % 7,
                'complete',
                f'2024-01-{i % 28 + 1:02d}',
            )
            for i in range(200)
        ],
    )
    connection.executemany(
        'INSERT INTO tickets VALUES (?, ?)',
        [(i, 'late delivery') for i in range(5)],
    )
    connection.commit()
    connection.close()
    return path


@pytest.fixture(scope='module')
def shop_graph(tmp_path_factory: pytest.TempPathFactory) -> GraphDefinition:
    path = _create_database(tmp_path_factory.mktemp('features') / 'shop.db')
    graph = Graph.from_sqlite(
        path,
        tables=[
            dict(name='users', primary_key='user_id'),
            dict(name='items', primary_key='item_id'),
            dict(name='orders', primary_key='order_id', time_column='ts'),
            dict(name='tickets', primary_key='ticket_id'),
        ],
        edges=[
            ('orders', 'user_id', 'users'),
            ('orders', 'item_id', 'items'),
        ],
        verbose=False,
    )
    return graph._to_api_graph_definition()


# Parsing #####################################################################


@pytest.mark.parametrize(
    ('expression', 'node_type'),
    [
        pytest.param(
            WINDOWED_AGGREGATION, Aggregation, id='backward-window-aggregation'
        ),
        pytest.param(
            FILTERED_AGGREGATION, Aggregation, id='filtered-aggregation'
        ),
        pytest.param(BARE_COLUMN, Column, id='bare-column-on-entity-table'),
    ],
)
def test_a_fragment_parses_to_the_node_it_names(
    expression: str,
    node_type: type[ASTNode],
) -> None:
    feature = parse_feature(expression, ENTITY)

    assert isinstance(feature.ast, node_type)
    # The fragment is the feature's name downstream, so it survives verbatim.
    assert feature.expression == expression
    assert feature.entity == ENTITY


def test_a_windowed_aggregation_keeps_its_window() -> None:
    feature = parse_feature(WINDOWED_AGGREGATION, ENTITY)

    assert isinstance(feature.ast, Aggregation)
    assert feature.ast.aggr_time_range.start == -90
    assert feature.ast.aggr_time_range.end == 0
    assert feature.ast.get_target_column_name() == 'orders.price'


def test_a_filtered_aggregation_keeps_its_filter() -> None:
    feature = parse_feature(FILTERED_AGGREGATION, ENTITY)

    assert isinstance(feature.ast, Aggregation)
    assert isinstance(feature.ast.target, Filter)
    assert feature.ast.target.target.fqn == 'orders.*'


def test_a_fragment_is_located_in_itself_not_in_the_wrapper() -> None:
    # The parse runs on `PREDICT <fragment> FOR EACH <entity>`, which starts
    # the fragment at column 8. Nothing the caller ever sees may say so.
    feature = parse_feature(WINDOWED_AGGREGATION, ENTITY)

    location = feature.ast.get_location()

    assert (location.start_row, location.start_col) == (1, 0)
    assert location.end_col == len(WINDOWED_AGGREGATION) - 1


def test_a_typo_is_reported_where_the_caller_wrote_it() -> None:
    # 'dayz' begins at offset 26 of the fragment, and at offset 34 of the
    # synthetic query the parser actually saw.
    expression = 'SUM(orders.price, -90, 0, dayz)'
    assert expression.index('dayz') == 26

    with pytest.raises(ValueError) as excinfo:
        parse_feature(expression, ENTITY)

    message = str(excinfo.value)
    assert 'Line 1, col 26;' in message
    assert 'col 34' not in message
    assert "Invalid aggregation time unit 'dayz'" in message


def test_a_typo_at_the_start_is_reported_at_column_zero() -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_feature('SUMM(orders.price, -90, 0, days)', ENTITY)

    assert 'Line 1, col 0;' in str(excinfo.value)


def test_a_syntax_error_never_mentions_the_synthetic_wrapper() -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_feature('SUMM(orders.price, -90, 0, days)', ENTITY)

    message = str(excinfo.value)
    assert 'PREDICT' not in message
    assert 'FOR EACH' not in message


@pytest.mark.parametrize(
    'expression',
    [
        pytest.param('SUM(orders.price, -90', id='truncated-mid-arguments'),
        pytest.param(
            'SUM(orders.price, -90, 0, days', id='missing-closing-parenthesis'
        ),
    ],
)
def test_a_truncated_fragment_is_reported_as_incomplete(
    expression: str,
) -> None:
    # The parser fails on the ` FOR EACH ` the wrapper appended, which the
    # caller neither wrote nor can fix. Saying so is the only useful reading.
    with pytest.raises(ValueError) as excinfo:
        parse_feature(expression, ENTITY)

    message = str(excinfo.value)
    assert 'Incomplete feature expression' in message
    assert 'ends before the expression is complete' in message
    assert f'row 1, column {len(expression)}' in message
    assert 'FOR EACH' not in message


# Validation ##################################################################


@pytest.mark.parametrize(
    'expression',
    [
        pytest.param(WINDOWED_AGGREGATION, id='backward-window-aggregation'),
        pytest.param(FILTERED_AGGREGATION, id='filtered-aggregation'),
        pytest.param(BARE_COLUMN, id='bare-column-on-entity-table'),
    ],
)
def test_the_valid_shapes_validate(
    shop_graph: GraphDefinition,
    expression: str,
) -> None:
    response = validate_features(
        [parse_feature(expression, ENTITY)], shop_graph
    )

    assert response.ok, response.message()


def test_a_backward_window_validates_where_a_target_would_not(
    shop_graph: GraphDefinition,
) -> None:
    # Regression guard for the reason this module calls the validators one at
    # a time: the full query validator holds its target to
    # `allow_past_looking=False`, so the same expression as a label is
    # rejected outright. A feature is a backward window by definition.
    from kumo_relational_engine.rfm.query_parser import parse_query_locally

    query = f'PREDICT {WINDOWED_AGGREGATION} FOR EACH {ENTITY}'
    with pytest.raises(ValueError, match='cannot have an aggregation in'):
        parse_query_locally(query, shop_graph)

    response = validate_features(
        [parse_feature(WINDOWED_AGGREGATION, ENTITY)], shop_graph
    )

    assert response.ok, response.message()


def test_a_timestamp_column_is_not_rejected_here(
    shop_graph: GraphDefinition,
) -> None:
    # Dropping a raw time column is a model-boundary concern, not a PQL one.
    response = validate_features(
        [parse_feature('MAX(orders.ts, -90, 0, days)', ENTITY)], shop_graph
    )

    assert response.ok, response.message()


@pytest.mark.parametrize(
    ('expression', 'expected'),
    [
        pytest.param(
            'SUM(ordrs.price, -90, 0, days)',
            "Table 'ordrs' does not exist in the graph",
            id='unknown-table',
        ),
        pytest.param(
            'SUM(orders.prise, -90, 0, days)',
            "Column 'prise' does not exist in table 'orders'",
            id='unknown-column',
        ),
        pytest.param(
            'tickets.note',
            'There is no foreign key from table users to table tickets',
            id='table-unreachable-from-entity',
        ),
        pytest.param(
            'SUM(orders.status, -90, 0, days)',
            'SUM can only operate on integers and floats',
            id='sum-over-a-categorical-column',
        ),
        pytest.param(
            'SUM(orders.price, 0, 30, days)',
            'contains a future-looking time range',
            id='window-reaching-into-the-future',
        ),
    ],
)
def test_an_invalid_feature_is_rejected_with_a_useful_diagnostic(
    shop_graph: GraphDefinition,
    expression: str,
    expected: str,
) -> None:
    response = validate_features(
        [parse_feature(expression, ENTITY)], shop_graph
    )

    assert not response.ok
    message = response.error_message()
    assert expected in message
    # Every diagnostic names the feature it is about, because a caller
    # declares a list of them and the row and column mean nothing until they
    # know which one to look at.
    assert f'Feature {expression!r}' in message


def test_a_diagnostic_points_into_the_fragment(
    shop_graph: GraphDefinition,
) -> None:
    expression = 'SUM(ordrs.price, -90, 0, days)'
    assert expression.index('ordrs') == 4

    response = validate_features(
        [parse_feature(expression, ENTITY)], shop_graph
    )

    assert 'row 1, column 4:' in response.error_message()


def test_every_feature_in_a_batch_is_reported(
    shop_graph: GraphDefinition,
) -> None:
    features = [
        parse_feature(WINDOWED_AGGREGATION, ENTITY),
        parse_feature('SUM(ordrs.price, -90, 0, days)', ENTITY),
        parse_feature('SUM(orders.prise, -90, 0, days)', ENTITY),
    ]

    response = validate_features(features, shop_graph)

    assert len(response.errors) == 2
    assert "Table 'ordrs'" in response.error_message()
    assert "Column 'prise'" in response.error_message()


def test_validation_resolves_the_join_from_the_entity(
    shop_graph: GraphDefinition,
) -> None:
    feature = parse_feature(WINDOWED_AGGREGATION, ENTITY)
    assert isinstance(feature.ast, Aggregation)

    assert validate_features([feature], shop_graph).ok

    # Join inferral rewrites the node in place, and `ParsedFeature.ast` reads
    # through to the rewritten one rather than holding the stale node.
    assert isinstance(feature.ast, Join)
    assert feature.ast.lhs_key == 'users.user_id'
    assert feature.ast.rhs_key == 'orders.user_id'


def test_a_feature_carries_the_entity_it_was_parsed_for() -> None:
    feature = parse_feature(BARE_COLUMN, ENTITY)

    assert isinstance(feature, ParsedFeature)
    assert feature.query.entity_column == ENTITY

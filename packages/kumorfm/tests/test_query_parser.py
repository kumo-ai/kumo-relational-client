# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from unittest.mock import patch

import pytest
from kumorfm.api.pquery import ValidatedPredictiveQuery
from kumorfm.api.typing import ProblemType

from kumorfm.rfm import Graph, KumoRFM
from kumorfm.rfm.query_parser import parse_query_locally


@dataclass(frozen=True)
class ExpectedQuery:
    canonical: str
    entity_ids: list[int | str]
    target_ast: str
    entity_ast: str = 'USERS.USER_ID'
    whatif_ast: str | None = None
    problem_type: ProblemType | None = None
    num_forecasts: int = 1
    top_k: int | None = None


def test_kumo_rfm_parse_query_accepts_validated_query(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    model = KumoRFM(user_store_graph, verbose=False)

    assert model._parse_query(ltv) is ltv


def test_kumo_rfm_parse_query_delegates_strings_to_local_parser(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    model = KumoRFM(user_store_graph, verbose=False)
    query = 'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) FOR USERS.USER_ID=0'

    with patch(
        'kumorfm.rfm.rfm.parse_query_locally',
        return_value=ltv,
    ) as mock_parse:
        assert model._parse_query(query) is ltv

    mock_parse.assert_called_once_with(query, model._graph_def)


def test_parse_query_locally_returns_validated_query(
    user_store_graph: Graph,
) -> None:
    query = parse_query_locally(
        'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) FOR USERS.USER_ID=0',
        user_store_graph._to_api_graph_definition(),
    )

    assert isinstance(query, ValidatedPredictiveQuery)
    assert query.to_string() == (
        'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) FOR USERS.USER_ID = 0')
    assert query.get_rfm_entity_id_list() == [0]


def test_demo_prefix_offsets_assuming_location(
    user_store_graph: Graph,
) -> None:
    from kumorfm.pql.parser.parser import PQLParser, QueryValidationType
    from kumorfm.pql.validator.rfm_validator import RfmValidator

    prefix = 'EXPLAIN '
    parsed = PQLParser(
        query_validation_type=QueryValidationType.RFM_DEMO,
    ).to_parsed_predictive_query(
        prefix + 'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) '
        'FOR USERS.USER_ID = 0 '
        'ASSUMING SUM(ORDERS.AMOUNT, 0, 7, days) >= 10')
    assert parsed.whatif_ast is not None
    original_start_col = parsed.whatif_ast.location.start_col

    RfmValidator(
        user_store_graph._to_api_graph_definition(),
        QueryValidationType.RFM_DEMO,
    ).update_location_interval(parsed)

    assert parsed.whatif_ast.location.start_col == (
        original_start_col + len(prefix))


@pytest.mark.parametrize(
    ('query', 'expected'),
    [
        pytest.param(
            'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) FOR USERS.USER_ID=0',
            ExpectedQuery(
                canonical=('PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) '
                           'FOR USERS.USER_ID = 0'),
                entity_ids=[0],
                target_ast='SUM(ORDERS.AMOUNT, 0, 7, days)',
            ),
            id='single-numeric-entity-id',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.AMOUNT, 0, 7) '
            'FOR USERS.USER_ID IN (0, 1, 3)',
            ExpectedQuery(
                canonical=('PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) '
                           'FOR USERS.USER_ID IN (0, 1, 3)'),
                entity_ids=[0, 1, 3],
                target_ast='SUM(ORDERS.AMOUNT, 0, 7, days)',
            ),
            id='default-time-unit',
        ),
        pytest.param(
            'predict sum(ORDERS.AMOUNT, 0, 7, days) # target\n'
            'for USERS.USER_ID in (0, 1, 3)',
            ExpectedQuery(
                canonical=('PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) '
                           'FOR USERS.USER_ID IN (0, 1, 3)'),
                entity_ids=[0, 1, 3],
                target_ast='SUM(ORDERS.AMOUNT, 0, 7, days)',
            ),
            id='comments-and-keyword-case',
        ),
        pytest.param(
            'PREDICT COUNT(ORDERS.*, 0, 7, days) '
            'FOR USERS.USER_ID IN (0, 1, 3)',
            ExpectedQuery(
                canonical=('PREDICT COUNT(ORDERS.*, 0, 7, days) '
                           'FOR USERS.USER_ID IN (0, 1, 3)'),
                entity_ids=[0, 1, 3],
                target_ast='COUNT(ORDERS.*, 0, 7, days)',
            ),
            id='count-wildcard',
        ),
        pytest.param(
            'PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 7, days) '
            'RANK TOP 2 FOR USERS.USER_ID IN (0, 1, 3)',
            ExpectedQuery(
                canonical=('PREDICT LIST_DISTINCT(ORDERS.STORE_ID, '
                           '0, 7, days) RANK TOP 2 FOR USERS.USER_ID '
                           'IN (0, 1, 3)'),
                entity_ids=[0, 1, 3],
                target_ast='LIST_DISTINCT(ORDERS.STORE_ID, 0, 7, days)',
                problem_type=ProblemType.RANK,
                top_k=2,
            ),
            id='rank-link-prediction-top-k',
        ),
        pytest.param(
            'PREDICT AVG(ORDERS.AMOUNT, 0, 7, days) FOR USERS.USER_ID = 0',
            ExpectedQuery(
                canonical=('PREDICT AVG(ORDERS.AMOUNT, 0, 7, days) '
                           'FOR USERS.USER_ID = 0'),
                entity_ids=[0],
                target_ast='AVG(ORDERS.AMOUNT, 0, 7, days)',
            ),
            id='avg-aggregation',
        ),
        pytest.param(
            'PREDICT MIN(ORDERS.AMOUNT, 0, 7, days) FOR USERS.USER_ID = 0',
            ExpectedQuery(
                canonical=('PREDICT MIN(ORDERS.AMOUNT, 0, 7, days) '
                           'FOR USERS.USER_ID = 0'),
                entity_ids=[0],
                target_ast='MIN(ORDERS.AMOUNT, 0, 7, days)',
            ),
            id='min-aggregation',
        ),
        pytest.param(
            'PREDICT MAX(ORDERS.AMOUNT, 0, 7, days) FOR USERS.USER_ID = 0',
            ExpectedQuery(
                canonical=('PREDICT MAX(ORDERS.AMOUNT, 0, 7, days) '
                           'FOR USERS.USER_ID = 0'),
                entity_ids=[0],
                target_ast='MAX(ORDERS.AMOUNT, 0, 7, days)',
            ),
            id='max-aggregation',
        ),
        pytest.param(
            'PREDICT COUNT(ORDERS.* WHERE ORDERS.AMOUNT > 10.5, '
            '0, 7, days) FOR USERS.USER_ID IN (0, 1, 3)',
            ExpectedQuery(
                canonical=('PREDICT COUNT(ORDERS.* WHERE ORDERS.AMOUNT > '
                           '10.5, 0, 7, days) FOR USERS.USER_ID IN '
                           '(0, 1, 3)'),
                entity_ids=[0, 1, 3],
                target_ast=('COUNT(ORDERS.* WHERE ORDERS.AMOUNT > 10.5, '
                            '0, 7, days)'),
            ),
            id='target-filter-decimal',
        ),
        pytest.param(
            'PREDICT COUNT(ORDERS.* WHERE ORDERS.AMOUNT IN (10, 15), '
            '0, 7, days) FOR USERS.USER_ID IN (0, 1, 3)',
            ExpectedQuery(
                canonical=('PREDICT COUNT(ORDERS.* WHERE ORDERS.AMOUNT IN '
                           '(10, 15), 0, 7, days) FOR USERS.USER_ID IN '
                           '(0, 1, 3)'),
                entity_ids=[0, 1, 3],
                target_ast=('COUNT(ORDERS.* WHERE ORDERS.AMOUNT IN '
                            '(10, 15), 0, 7, days)'),
            ),
            id='target-filter-membership',
        ),
        pytest.param(
            'PREDICT COUNT(ORDERS.* WHERE NOT ORDERS.AMOUNT IS NULL, '
            '0, 7, days) FOR USERS.USER_ID IN (0, 1, 3)',
            ExpectedQuery(
                canonical=('PREDICT COUNT(ORDERS.* WHERE NOT '
                           '(ORDERS.AMOUNT IS NULL), 0, 7, days) '
                           'FOR USERS.USER_ID IN (0, 1, 3)'),
                entity_ids=[0, 1, 3],
                target_ast=('COUNT(ORDERS.* WHERE NOT '
                            '(ORDERS.AMOUNT IS NULL), 0, 7, days)'),
            ),
            id='target-filter-null-negation',
        ),
        pytest.param(
            'PREDICT COUNT(ORDERS.* WHERE ORDERS.AMOUNT >= 10 '
            'AND ORDERS.AMOUNT <= 25, 0, 7, days) '
            'FOR USERS.USER_ID IN (0, 1, 3)',
            ExpectedQuery(
                canonical=('PREDICT COUNT(ORDERS.* WHERE '
                           '(ORDERS.AMOUNT >= 10) AND '
                           '(ORDERS.AMOUNT <= 25), 0, 7, days) '
                           'FOR USERS.USER_ID IN (0, 1, 3)'),
                entity_ids=[0, 1, 3],
                target_ast=('COUNT(ORDERS.* WHERE (ORDERS.AMOUNT >= 10) '
                            'AND (ORDERS.AMOUNT <= 25), 0, 7, days)'),
            ),
            id='target-filter-logical-and',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) > 25 '
            'FOR USERS.USER_ID IN (0, 1, 3)',
            ExpectedQuery(
                canonical=('PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) > 25 '
                           'FOR USERS.USER_ID IN (0, 1, 3)'),
                entity_ids=[0, 1, 3],
                target_ast='SUM(ORDERS.AMOUNT, 0, 7, days) > 25',
            ),
            id='target-threshold-condition',
        ),
        pytest.param(
            'PREDICT USERS.AGE FOR USERS.USER_ID IN (2, 3)',
            ExpectedQuery(
                canonical='PREDICT USERS.AGE FOR USERS.USER_ID IN (2, 3)',
                entity_ids=[2, 3],
                target_ast='USERS.AGE',
            ),
            id='static-numeric-target',
        ),
        pytest.param(
            'PREDICT USERS.STATUS = "A" FOR USERS.USER_ID IN (0, 1, 3)',
            ExpectedQuery(
                canonical=('PREDICT USERS.STATUS = "A" '
                           'FOR USERS.USER_ID IN (0, 1, 3)'),
                entity_ids=[0, 1, 3],
                target_ast='USERS.STATUS = "A"',
            ),
            id='static-categorical-condition',
        ),
        pytest.param(
            'PREDICT COUNT(ORDERS.*, 0, 7, days) '
            'FOR USERS.USER_ID IN (0, 1, 3) '
            'WHERE USERS.STATUS = "A" OR USERS.AGE >= 30',
            ExpectedQuery(
                canonical=('PREDICT COUNT(ORDERS.*, 0, 7, days) '
                           'FOR USERS.USER_ID IN (0, 1, 3) WHERE '
                           '(USERS.STATUS = "A") OR (USERS.AGE >= 30)'),
                entity_ids=[0, 1, 3],
                target_ast='COUNT(ORDERS.*, 0, 7, days)',
                entity_ast=('USERS.USER_ID WHERE (USERS.STATUS = "A") '
                            'OR (USERS.AGE >= 30)'),
            ),
            id='entity-filter-static-or',
        ),
        pytest.param(
            'PREDICT COUNT(ORDERS.*, 0, 7, days) '
            'FOR USERS.USER_ID IN (0, 1, 3) '
            'WHERE COUNT(ORDERS.*, -7, 0, days) > 0',
            ExpectedQuery(
                canonical=('PREDICT COUNT(ORDERS.*, 0, 7, days) '
                           'FOR USERS.USER_ID IN (0, 1, 3) WHERE '
                           'COUNT(ORDERS.*, -7, 0, days) > 0'),
                entity_ids=[0, 1, 3],
                target_ast='COUNT(ORDERS.*, 0, 7, days)',
                entity_ast=('USERS.USER_ID WHERE COUNT(ORDERS.*, '
                            '-7, 0, days) > 0'),
            ),
            id='entity-filter-temporal-aggregation',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) '
            'FOR USERS.USER_ID IN (0, 1, 3) '
            'ASSUMING SUM(ORDERS.AMOUNT, 0, 7, days) >= 10',
            ExpectedQuery(
                canonical=('PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) '
                           'FOR USERS.USER_ID IN (0, 1, 3) ASSUMING '
                           'SUM(ORDERS.AMOUNT, 0, 7, days) >= 10'),
                entity_ids=[0, 1, 3],
                target_ast='SUM(ORDERS.AMOUNT, 0, 7, days)',
                whatif_ast='SUM(ORDERS.AMOUNT, 0, 7, days) >= 10',
            ),
            id='assuming-whatif-condition',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.AMOUNT, 0, 1, days) '
            'FORECAST 4 TIMEFRAMES FOR USERS.USER_ID = 0',
            ExpectedQuery(
                canonical=('PREDICT SUM(ORDERS.AMOUNT, 0, 1, days) '
                           'FORECAST 4 TIMEFRAMES FOR USERS.USER_ID = 0'),
                entity_ids=[0],
                target_ast='SUM(ORDERS.AMOUNT, 0, 1, days)',
                problem_type=ProblemType.FORECAST,
                num_forecasts=4,
            ),
            id='forecast',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) FOR EACH USERS.USER_ID',
            ExpectedQuery(
                canonical=('PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) '
                           'FOR EACH USERS.USER_ID'),
                entity_ids=[],
                target_ast='SUM(ORDERS.AMOUNT, 0, 7, days)',
            ),
            id='for-each-entity',
        ),
    ],
)
def test_parse_valid_queries_against_rfm_graph(
    user_store_graph: Graph,
    query: str,
    expected: ExpectedQuery,
) -> None:
    parsed = KumoRFM(user_store_graph, verbose=False)._parse_query(query)

    assert isinstance(parsed, ValidatedPredictiveQuery)
    assert parsed.to_string() == expected.canonical
    assert parsed.get_rfm_entity_id_list() == expected.entity_ids
    assert str(parsed.target_ast) == expected.target_ast
    assert str(parsed.entity_ast) == expected.entity_ast
    assert parsed.problem_type == expected.problem_type
    assert parsed.num_forecasts == expected.num_forecasts
    assert parsed.top_k == expected.top_k
    if expected.whatif_ast is None:
        assert parsed.whatif_ast is None
    else:
        assert str(parsed.whatif_ast) == expected.whatif_ast


@pytest.mark.parametrize(
    ('query', 'expected_ids', 'expected_target'),
    [
        pytest.param(
            'PREDICT COUNT(ORDERS.*, 0, 365, days) '
            'FOR USERS.USER_ID = "user_a"',
            ['user_a'],
            'COUNT(ORDERS.*, 0, 365, days)',
            id='single-double-quoted-string-id',
        ),
        pytest.param(
            'PREDICT COUNT(ORDERS.*, 0, 365, days) '
            'FOR USERS.USER_ID IN ("user_a", "user_c")',
            ['user_a', 'user_c'],
            'COUNT(ORDERS.*, 0, 365, days)',
            id='double-quoted-string-id-list',
        ),
        pytest.param(
            "PREDICT COUNT(ORDERS.*, 0, 365, days) "
            "FOR USERS.USER_ID IN ('user_a', 'user_c')",
            ['user_a', 'user_c'],
            'COUNT(ORDERS.*, 0, 365, days)',
            id='single-quoted-string-id-list',
        ),
        pytest.param(
            'PREDICT USERS.NAME = "Alice" '
            'FOR USERS.USER_ID IN ("user_a", "user_c")',
            ['user_a', 'user_c'],
            'USERS.NAME = "Alice"',
            id='string-condition-target',
        ),
    ],
)
def test_parse_string_entity_ids(
    string_user_graph: Graph,
    query: str,
    expected_ids: list[str],
    expected_target: str,
) -> None:
    parsed = KumoRFM(string_user_graph, verbose=False)._parse_query(query)

    assert parsed.get_rfm_entity_id_list() == expected_ids
    assert str(parsed.target_ast) == expected_target


@pytest.mark.parametrize(
    ('query', 'expected_message'),
    [
        pytest.param(
            'SUM(ORDERS.AMOUNT, 0, 7, days) FOR USERS.USER_ID = 0',
            'Query does not start with the PREDICT keyword',
            id='missing-predict',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.MISSING, 0, 7, days) '
            'FOR USERS.USER_ID = 0',
            "Column 'MISSING' does not exist in table 'ORDERS'",
            id='unknown-column',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) '
            'FOR USERSS.USER_ID = 0',
            "Table 'USERSS' does not exist in the graph",
            id='unknown-table',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) FOR USERS.AGE = 20',
            "Entity 'USERS.AGE' is not a primary key",
            id='entity-not-primary-key',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.*, 0, 7, days) FOR USERS.USER_ID = 0',
            'Wildcard column ORDERS.* appeared in an invalid position',
            id='wildcard-outside-count',
        ),
        pytest.param(
            'PREDICT ORDERS.AMOUNT FOR USERS.USER_ID = 0',
            "Couldn't pick a unique ORDERS.AMOUNT for each USERS",
            id='many-to-one-target-without-aggregation',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.AMOUNT, 7, 0, days) FOR USERS.USER_ID = 0',
            "start date offset '7' needs to be smaller",
            id='invalid-time-range',
        ),
        pytest.param(
            'PREDICT FIRST(ORDERS.AMOUNT, 0, 7, days) '
            'FOR USERS.USER_ID = 0',
            'Aggregation FIRST type is not supported',
            id='unsupported-aggregation',
        ),
        pytest.param(
            'PREDICT COUNT(ORDERS.* WHERE STORES.CAT = "pizza", '
            '0, 7, days) FOR USERS.USER_ID IN (0, 1, 3)',
            'Unsupported implicit join',
            id='unsupported-implicit-join',
        ),
        pytest.param(
            'PREDICT COUNT(ORDERS.*, 0, 7, days) RANK TOP 2 '
            'FOR USERS.USER_ID = 0',
            "Problem type 'RANK' is only supported",
            id='unsupported-rank-problem-type',
        ),
        pytest.param(
            'PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 7, days) '
            'TOP 100 FOR USERS.USER_ID IN (0, 1, 3)',
            'TOP K requires RANK',
            id='top-k-without-rank',
        ),
        pytest.param(
            'PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 7, days) '
            'RANK TOP 0 FOR USERS.USER_ID IN (0, 1, 3)',
            'TOP K value must be at least 1',
            id='top-k-zero',
        ),
        pytest.param(
            'PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 7, days) '
            'RANK TOP -1 FOR USERS.USER_ID IN (0, 1, 3)',
            'TOP K value must be at least 1',
            id='top-k-negative',
        ),
        pytest.param(
            'PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 7, days) '
            'RANK TOP 21 FOR USERS.USER_ID IN (0, 1, 3)',
            'Top k 21 exceeds maximum',
            id='top-k-exceeds-rfm-limit',
        ),
        pytest.param(
            'PREDICT USERS.AGE FORECAST 4 TIMEFRAMES FOR USERS.USER_ID = 0',
            'FORECAST requires a temporal target',
            id='forecast-static-target',
        ),
        pytest.param(
            'PREDICT SUM(ORDERS.AMOUNT, 0, 7, days) '
            'FOR USERS.USER_ID IN ("0")',
            'Array with single element is not supported',
            id='single-element-in-list',
        ),
    ],
)
def test_parse_invalid_queries_report_useful_errors(
    user_store_graph: Graph,
    query: str,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        KumoRFM(user_store_graph, verbose=False)._parse_query(query)

    message = str(exc_info.value)
    assert query in message
    assert expected_message in message

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from kumo_relational_engine.api.task import TaskType
from kumo_relational_engine.rfm import Graph

pytest.importorskip(
    'adbc_driver_sqlite', reason="'sqlite' extension not installed"
)

from kumo_relational_engine.tfm import KumoTabular

# The sqlite backend cannot seed its random sampling and says so every time a
# target is drawn. That is a property of the backend, not of label generation,
# and the session turns UserWarning into an error, so it is silenced here. The
# tests below never assert on *which* entities the sampler picked.
pytestmark = pytest.mark.filterwarnings(
    'ignore:.*seeded random sampling.*:UserWarning'
)

# The toy shop below is built so that every label can be computed by hand.
#
# Users 0..11 each fire an event every 14 days from 2024-01-01 to 2024-05-09
# for an amount of `user_id + 1`; user 99 fires a single event on 2024-01-02
# and then goes quiet forever.
#
# The latest event in the graph is therefore on 2024-05-09, so a
# `(0, 30, days)` query anchors its prediction examples 30 days earlier, at
# 2024-04-09, and its context examples one further window back, at
# 2024-03-10:
#
#   * the prediction window (2024-04-09, 2024-05-09] holds the events of
#     04-11, 04-25 and 05-09: three of them, summing to 3 * (user_id + 1);
#   * the context window (2024-03-10, 2024-04-09] holds the events of 03-14
#     and 03-28: two of them, summing to 2 * (user_id + 1);
#   * user 99 has no event in either window.
FIRST_EVENT = datetime.date(2024, 1, 1)
EVENT_DAYS = (0, 14, 31, 45, 59, 73, 87, 101, 115, 129)
NUM_USERS = 12
QUIET_USER = 99
QUIET_USER_AGE = 40

LATEST_EVENT = pd.Timestamp('2024-05-09')
PRED_ANCHOR = pd.Timestamp('2024-04-09')
CONTEXT_ANCHOR = pd.Timestamp('2024-03-10')

SUM_TEMPORAL = 'PREDICT SUM(events.amount, 0, 30, days) FOR EACH users.user_id'
COUNT_TEMPORAL = 'PREDICT COUNT(events.*, 0, 30, days) FOR EACH users.user_id'
ACTIVE_TEMPORAL = (
    'PREDICT COUNT(events.*, 0, 30, days) > 0 FOR EACH users.user_id'
)
AGE_STATIC = 'PREDICT users.age FOR EACH users.user_id'

# Hand-computed: two events in the context window, three in the prediction
# window, and nothing at all for the entity that went quiet in January.
EXPECTED_CONTEXT_SUM = {user: 2.0 * (user + 1) for user in range(NUM_USERS)}
EXPECTED_CONTEXT_SUM[QUIET_USER] = 0.0
EXPECTED_PRED_SUM = {user: 3.0 * (user + 1) for user in range(NUM_USERS)}
EXPECTED_PRED_SUM[QUIET_USER] = 0.0
EXPECTED_AGE = {user: 20 + user for user in range(NUM_USERS)}
EXPECTED_AGE[QUIET_USER] = QUIET_USER_AGE

ALL_USERS = {*range(NUM_USERS), QUIET_USER}


def _create_database(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.execute(
        'CREATE TABLE users ('
        '  user_id INTEGER PRIMARY KEY,'
        '  age INTEGER,'
        '  status TEXT)'
    )
    connection.execute(
        'CREATE TABLE events ('
        '  event_id INTEGER PRIMARY KEY,'
        '  user_id INTEGER,'
        '  amount REAL,'
        '  ts TEXT)'
    )
    connection.executemany(
        'INSERT INTO users VALUES (?, ?, ?)',
        [(user, 20 + user, 'ABC'[user % 3]) for user in range(NUM_USERS)]
        + [(QUIET_USER, QUIET_USER_AGE, 'A')],
    )
    events = [
        (
            user * len(EVENT_DAYS) + step,
            user,
            float(user + 1),
            str(FIRST_EVENT + datetime.timedelta(days=day)),
        )
        for user in range(NUM_USERS)
        for step, day in enumerate(EVENT_DAYS)
    ]
    events.append(
        (NUM_USERS * len(EVENT_DAYS), QUIET_USER, 500.0, '2024-01-02')
    )
    connection.executemany('INSERT INTO events VALUES (?, ?, ?, ?)', events)
    # Index the foreign key. Building a KumoTabular builds the backend
    # sampler, which warns about a sqlite database without one, and the
    # session turns that warning into an error.
    connection.execute('CREATE INDEX events_user_id ON events (user_id)')
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
            dict(name='events', primary_key='event_id', time_column='ts'),
        ],
        edges=[('events', 'user_id', 'users')],
        verbose=False,
    )


@pytest.fixture(scope='module')
def model(shop_graph: Graph) -> KumoTabular:
    return KumoTabular(shop_graph, verbose=False)


def _anchors(frame: pd.DataFrame) -> set[pd.Timestamp]:
    return set(pd.to_datetime(frame['ANCHOR_TIMESTAMP']))


def _labels(frame: pd.DataFrame) -> dict[int, float]:
    return dict(zip(frame['ENTITY'], frame['TARGET']))


def test_a_temporal_query_labels_its_context_one_window_before_the_anchor(
    model: KumoTabular,
) -> None:
    task = model._generate_labels(model._parse_query(SUM_TEMPORAL), 6)

    assert task.task_type == TaskType.REGRESSION
    assert task.num_context_examples == 6

    context = task._context_df
    # Every context example is anchored at the same timestamp, one 30-day
    # window before the prediction anchor, and carries the sum of the two
    # events that fall in the window after it.
    assert _anchors(context) == {CONTEXT_ANCHOR}
    assert _labels(context) == {
        entity: EXPECTED_CONTEXT_SUM[entity] for entity in context['ENTITY']
    }

    # Which entities are drawn into the context is up to the sampler -- the
    # sqlite backend cannot seed it -- but they are real entities, labelled
    # once each.
    assert set(context['ENTITY']) <= ALL_USERS
    assert not context['ENTITY'].duplicated().any()


def test_a_temporal_query_labels_every_entity_at_the_prediction_anchor(
    model: KumoTabular,
) -> None:
    task = model._generate_labels(model._parse_query(SUM_TEMPORAL), 6)

    prediction = task._pred_df
    # There are only 13 entities, so all of them fit under the prediction
    # budget and the whole frame is hand-checkable.
    assert _anchors(prediction) == {PRED_ANCHOR}
    assert _labels(prediction) == EXPECTED_PRED_SUM


def test_an_explicit_anchor_time_moves_both_windows(
    model: KumoTabular,
) -> None:
    # 2024-03-20 puts the prediction window at (03-20, 04-19], which holds
    # the events of 03-28 and 04-11, and the context window at
    # (02-19, 03-20], which holds the events of 02-29 and 03-14: two events
    # apiece rather than the three the derived anchor sees.
    task = model._generate_labels(
        model._parse_query(SUM_TEMPORAL),
        6,
        anchor_time=pd.Timestamp('2024-03-20'),
    )

    assert _anchors(task._pred_df) == {pd.Timestamp('2024-03-20')}
    assert _anchors(task._context_df) == {pd.Timestamp('2024-02-19')}
    assert _labels(task._pred_df) == {
        **{user: 2.0 * (user + 1) for user in range(NUM_USERS)},
        QUIET_USER: 0.0,
    }


def test_a_static_query_reads_the_label_off_the_entity_row(
    model: KumoTabular,
) -> None:
    task = model._generate_labels(model._parse_query(AGE_STATIC), 5)

    assert task.task_type == TaskType.REGRESSION
    assert task.num_context_examples == 5

    context, prediction = task._context_df, task._pred_df
    # A static query has no aggregation window, so the label is the column
    # itself and does not depend on the anchor. The anchor is still recorded,
    # as the one timestamp the graph ends at, and is the same for the context
    # and the prediction examples rather than a window apart.
    assert _anchors(context) == {LATEST_EVENT}
    assert _anchors(prediction) == {LATEST_EVENT}
    assert _labels(context) == {
        entity: EXPECTED_AGE[entity] for entity in context['ENTITY']
    }
    assert _labels(prediction) == {
        entity: EXPECTED_AGE[entity] for entity in prediction['ENTITY']
    }
    # A static query partitions the entities: no entity is both a context
    # example and a prediction example.
    assert not set(context['ENTITY']) & set(prediction['ENTITY'])


@pytest.mark.parametrize(
    ('query', 'expected'),
    [
        pytest.param(SUM_TEMPORAL, 0.0, id='sum-of-no-rows'),
        pytest.param(COUNT_TEMPORAL, 0.0, id='count-of-no-rows'),
        pytest.param(ACTIVE_TEMPORAL, False, id='count-of-no-rows-is-not-gt-0'),
    ],
)
def test_an_entity_with_no_events_in_the_window_gets_the_empty_aggregate(
    model: KumoTabular,
    query: str,
    expected: float | bool,
) -> None:
    # The quiet entity's only event is on 2024-01-02, months before either
    # window opens. The engine reads the empty window as zero rather than as
    # a missing label, which is what makes `COUNT(...) > 0` a usable churn
    # target: the entity that did nothing is a negative, not a hole.
    task = model._generate_labels(model._parse_query(query), 6)

    labels = _labels(task._pred_df)

    assert labels[QUIET_USER] == expected
    assert not pd.isna(labels[QUIET_USER])
    # The entities that did fire events in the window are not zero, so the
    # zero above is the empty window and not a label that failed to compute.
    assert all(labels[user] != expected for user in range(NUM_USERS))


@pytest.mark.parametrize('context_size', [1, 3, 7])
def test_context_size_is_the_number_of_context_examples(
    model: KumoTabular,
    context_size: int,
) -> None:
    task = model._generate_labels(
        model._parse_query(SUM_TEMPORAL), context_size
    )

    assert task.num_context_examples == context_size


def test_context_size_has_no_default(model: KumoTabular) -> None:
    query = model._parse_query(SUM_TEMPORAL)

    with pytest.raises(TypeError, match='context_size'):
        model._generate_labels(query)  # type: ignore[call-arg]

    with pytest.raises(TypeError, match='context_size'):
        model.predict(SUM_TEMPORAL)  # type: ignore[call-arg]


def test_context_size_must_be_positive(model: KumoTabular) -> None:
    query = model._parse_query(SUM_TEMPORAL)

    with pytest.raises(ValueError, match='must be greater than zero'):
        model._generate_labels(query, 0)


def test_predict_generates_labels_before_it_declines_to_predict(
    model: KumoTabular,
) -> None:
    with pytest.raises(NotImplementedError, match='land in a later release'):
        model.predict(SUM_TEMPORAL, indices=[1, 2], context_size=6)

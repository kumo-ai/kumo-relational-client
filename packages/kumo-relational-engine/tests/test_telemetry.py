# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What a prediction says about itself, and what it must not say."""

import json
import logging

import kumo_relational_engine.rfm as rfm
import pandas as pd
import pytest
from kumo_relational_engine.rfm.rfm import KumoRelational
from kumo_relational_engine.rfm.telemetry import (
    PredictionRecord,
    emit,
    recorded,
)

QUERY = 'PREDICT COUNT(orders.*, 0, 30) FOR customers.customer_id IN (1, 2, 3)'


@pytest.fixture
def model() -> KumoRelational:
    customers = pd.DataFrame(
        {
            'customer_id': range(60),
            'seg': ['gold', 'silver', 'bronze'] * 20,
        }
    )
    orders = pd.DataFrame(
        {
            'order_id': range(600),
            'customer_id': [i % 60 for i in range(600)],
            'amt': [10.0 + (i % 37) for i in range(600)],
            'ts': pd.to_datetime(
                [
                    f'2025-{1 + (i // 50) % 12:02d}-{1 + i % 28:02d}'
                    for i in range(600)
                ]
            ),
        }
    )
    graph = rfm.Graph.from_data(
        {'customers': customers, 'orders': orders},
        infer_metadata=True,
        verbose=False,
    )
    graph['customers'].primary_key = 'customer_id'
    graph['orders'].primary_key = 'order_id'
    if not graph.edges:
        graph.link('orders', 'customer_id', 'customers')
    return KumoRelational(graph, verbose=False)


def _written(caplog: pytest.LogCaptureFixture) -> list[dict]:
    return [
        json.loads(record.getMessage().removeprefix('rfm.predict '))
        for record in caplog.records
        if record.getMessage().startswith('rfm.predict ')
    ]


def test_a_prediction_names_the_graph_it_ran_against(
    model: KumoRelational,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        KumoRelational,
        'predict_task',
        lambda self, task, **kw: pd.DataFrame({'ENTITY': [1, 2, 3]}),
    )

    with caplog.at_level(logging.INFO, logger='kumo_relational_engine'):
        model.predict(QUERY, verbose=False)

    written = _written(caplog)[0]
    assert len(written['graph_fingerprint']) == 64
    assert written['entities'] == 3
    assert written['outcome'] == 'returned'
    assert written['query_type'] == 'temporal'
    assert written['seconds'] >= 0


def test_a_prediction_that_raised_is_still_recorded(
    model: KumoRelational,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    r"""The run worth explaining afterwards is the one that failed."""

    def boom(self: KumoRelational, task: object, **kw: object) -> None:
        raise RuntimeError('NIM returned 504')

    monkeypatch.setattr(KumoRelational, 'predict_task', boom)

    with caplog.at_level(logging.INFO, logger='kumo_relational_engine'):
        with pytest.raises(RuntimeError, match='504'):
            model.predict(QUERY, verbose=False)

    written = _written(caplog)[0]
    assert written['outcome'] == 'raised'
    assert written['error'] == 'RuntimeError: NIM returned 504'


def test_the_batch_size_in_force_is_the_one_recorded(
    model: KumoRelational,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        KumoRelational,
        'predict_task',
        lambda self, task, **kw: pd.DataFrame({'ENTITY': [1, 2, 3]}),
    )

    with caplog.at_level(logging.INFO, logger='kumo_relational_engine'):
        model.predict(QUERY, verbose=False)
        with model.batch_mode(100):
            model.predict(QUERY, verbose=False)

    assert [w['batch_size'] for w in _written(caplog)] == ['None', '100']


def test_a_re_typed_graph_is_a_different_graph(model: KumoRelational) -> None:
    r"""A fingerprint that named the graph as it once was would mislead."""
    before = model._graph_fingerprint()

    model._graph['customers']['seg'].stype = 'text'

    assert model._graph_fingerprint() != before


def test_a_record_that_cannot_be_written_does_not_fail_the_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    r"""Telemetry describes a prediction; it must never be what breaks one."""

    class Unserializable:
        def __repr__(self) -> str:
            raise RuntimeError('no')

    record = PredictionRecord()
    record.entities = Unserializable()  # type: ignore[assignment]

    with caplog.at_level(logging.INFO, logger='kumo_relational_engine'):
        emit(record)

    assert not _written(caplog)


def test_a_cancelled_call_is_recorded_too(
    caplog: pytest.LogCaptureFixture,
) -> None:
    r"""A prediction killed by a timeout raises through, not from, ``except``."""
    with caplog.at_level(logging.INFO, logger='kumo_relational_engine'):
        with pytest.raises(KeyboardInterrupt):
            with recorded(PredictionRecord(entities=3)):
                raise KeyboardInterrupt

    written = _written(caplog)[0]
    assert written['outcome'] == 'raised'
    assert written['error'] == 'KeyboardInterrupt: '

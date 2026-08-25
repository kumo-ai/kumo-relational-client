# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The Kumo Relational examples that ship beside the contract.

These had no test at all: only the tabular examples were replayed, so the
relational half of the wire format was described by the contract and checked by
nobody. A full replay is not possible from this package, because a relational
request is assembled by the engine from a graph rather than by the client, so
what is pinned here is the envelope every one of them shares and the pieces the
client itself is responsible for.
"""

from __future__ import annotations

import pytest
from conftest import canonical_examples_available, load_canonical_example

from kumo_relational_client.adapters.relational import KumoRelationalAdapter

RELATIONAL = 'kumo-relational'

PREDICTION_EXAMPLES = [
    'prediction_kumo_relational_binary_classification.json',
    'prediction_kumo_relational_forecasting.json',
    'prediction_kumo_relational_temporal_link_prediction.json',
]


@canonical_examples_available
@pytest.mark.parametrize('example_file', PREDICTION_EXAMPLES)
def test_a_prediction_example_is_addressed_to_this_client(
    example_file: str,
) -> None:
    payload = load_canonical_example(RELATIONAL, example_file)

    assert payload['model'] == KumoRelationalAdapter().capabilities().model


@canonical_examples_available
@pytest.mark.parametrize('example_file', PREDICTION_EXAMPLES)
def test_a_prediction_example_carries_the_whole_envelope(
    example_file: str,
) -> None:
    r"""A prediction says what to run, on what, and what to return; dropping
    any one of those is a change in the contract, not a detail."""
    payload = load_canonical_example(RELATIONAL, example_file)

    assert {'model', 'task', 'schema', 'context', 'predict', 'output'} <= set(
        payload
    )
    assert payload['task']['target']['column_name']
    assert payload['task']['entity_table_names']


@canonical_examples_available
def test_creating_a_session_carries_context_but_asks_for_nothing() -> None:
    r"""A session is the context alone: it runs no inference, so it has no
    predict or output block."""
    payload = load_canonical_example(
        RELATIONAL,
        'session_create_kumo_relational_binary_classification.json',
    )

    assert {'model', 'task', 'schema', 'context'} <= set(payload)
    assert 'predict' not in payload
    assert 'output' not in payload


@canonical_examples_available
def test_predicting_in_a_session_carries_no_context() -> None:
    r"""The counterpart: the session already holds the context, so a
    prediction against it re-sends only what to score."""
    payload = load_canonical_example(
        RELATIONAL,
        'session_prediction_kumo_relational_binary_classification.json',
    )

    assert {'predict', 'output'} <= set(payload)
    assert 'context' not in payload
    assert 'schema' not in payload


@canonical_examples_available
def test_a_forecast_step_is_an_integer_number_of_nanoseconds() -> None:
    r"""The unit is not stated on the wire, so a client that reads it as
    seconds is wrong by a factor of a billion and nothing says so."""
    payload = load_canonical_example(
        RELATIONAL, 'prediction_kumo_relational_forecasting.json'
    )
    step = payload['task']['step_size']

    assert isinstance(step, int)
    assert step == 7 * 24 * 60 * 60 * 1_000_000_000


@canonical_examples_available
def test_a_temporal_link_prediction_ranks_between_two_entity_tables() -> None:
    payload = load_canonical_example(
        RELATIONAL,
        'prediction_kumo_relational_temporal_link_prediction.json',
    )

    assert len(payload['task']['entity_table_names']) == 2
    assert payload['task']['top_k'] >= 1

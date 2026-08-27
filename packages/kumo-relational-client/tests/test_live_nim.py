# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pandas as pd
import pytest

from kumo_relational_client import RelationalClient

_ENV_VAR = 'KUMO_RELATIONAL_NIM_BASE_URL'
_KEY_VAR = 'KUMO_RELATIONAL_NIM_API_KEY'

pytestmark = [
    pytest.mark.live_nim,
    pytest.mark.skipif(
        not os.environ.get(_ENV_VAR),
        reason=f'set {_ENV_VAR} to run live TFM NIM tests',
    ),
]


@pytest.fixture(scope='module')
def nim_url() -> str:
    return os.environ[_ENV_VAR].rstrip('/')


@pytest.fixture
def client(nim_url: str):
    # A deployment may front the NIM with an authenticating gateway; the
    # contract itself leaves NIMs unauthenticated, so the key is optional.
    with RelationalClient(
        url=nim_url, api_key=os.environ.get(_KEY_VAR)
    ) as client:
        yield client


def test_nim_reports_ready(client: RelationalClient):
    assert client.health_ready() is True


def test_nim_serves_the_relational_model(client: RelationalClient):
    caps = client.capabilities('kumo-relational')
    assert caps.model == 'kumo-relational'
    assert 'regression' in caps.tasks


def test_relational_predict_returns_a_row_per_entity(client: RelationalClient):
    relational = pytest.importorskip('kumo_relational_client.relational')

    users = pd.DataFrame({'user_id': range(1, 41)})
    orders = pd.DataFrame(
        {
            'order_id': range(1, 401),
            'user_id': [1 + (i % 40) for i in range(400)],
            'price': [5.0 + i % 50 for i in range(400)],
            'ts': pd.Timestamp('2024-01-01')
            + pd.to_timedelta([i % 180 for i in range(400)], unit='D'),
        }
    )
    graph = relational.Graph.from_data(
        {'users': users, 'orders': orders}, verbose=False
    )

    entities = [1, 2, 3]
    frame = client.relational(graph).predict(
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
        indices=entities,
        verbose=False,
    )

    assert len(frame) == len(entities)
    assert 'ENTITY' in frame.columns
    assert frame['PREDICTION'].notna().all()

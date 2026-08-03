# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from kumorfm.api.pquery import ValidatedPredictiveQuery
from kumorfm.exceptions import HTTPException
from kumorfm.rfm import Graph, KumoRFM

_CARDINALITY_DETAIL = (
    '{"detail": "categorical cardinality 15001 exceeds limit 10000."}')


class FailingAPI:
    r"""Fake RFMAPI whose predict always fails with a fixed status, counting
    how many times the retry loop actually called it.
    """
    def __init__(self, status: int, detail: str = 'boom') -> None:
        self.status = status
        self.detail = detail
        self.n_predict = 0

    def predict(self, request: Any, *, entity_ids: Any = None,
                instance_ids: Any = None,
                anchor_times: Any = None) -> Any:
        self.n_predict += 1
        raise HTTPException(self.status, self.detail)


def _model(graph: Graph, api: FailingAPI) -> KumoRFM:
    model = KumoRFM(graph, verbose=False)
    model._client = api  # type: ignore[assignment]
    return model


def _predict(model: KumoRFM, query: ValidatedPredictiveQuery,
             num_retries: int = 2) -> str:
    with model.retry(num_retries=num_retries):
        with pytest.raises(RuntimeError) as info:
            model.predict(query, indices=[0, 1], verbose=False)
    return str(info.value)


def test_rejected_request_is_not_retried(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    api = FailingAPI(422, _CARDINALITY_DETAIL)
    message = _predict(_model(user_store_graph, api), ltv)
    assert api.n_predict == 1
    assert 'categorical cardinality 15001 exceeds limit 10000' in message
    assert 'create an issue' not in message


def test_unavailable_nim_is_retried(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    api = FailingAPI(503)
    message = _predict(_model(user_store_graph, api), ltv, num_retries=2)
    assert api.n_predict == 3
    assert 'temporarily unavailable' in message


def test_rate_limit_is_retried(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    api = FailingAPI(429)
    _predict(_model(user_store_graph, api), ltv, num_retries=1)
    assert api.n_predict == 2


def test_cardinality_rejection_carries_the_remedy_end_to_end(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    r"""This fixture's payload is far below the limit, so the columns cannot be
    named; the remedy still has to survive the whole predict path. Naming the
    columns is covered against real payloads in ``test_explain_summary``.
    """
    api = FailingAPI(422, _CARDINALITY_DETAIL)
    message = _predict(_model(user_store_graph, api), ltv)
    assert 'Stype.ID' in message
    assert 'does not lift the limit' in message
    assert 'instance_table' not in message

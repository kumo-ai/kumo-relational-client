# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Sequence

import pandas as pd

from nvidia_sdfm.requests import KumoRFMRequest, TabICLRequest

if TYPE_CHECKING:
    from nvidia_sdfm.base import PredictResult
    from nvidia_sdfm.kumorfm import ExplainConfig

_UNSET: Any = object()


class RFMModel:
    r"""A KumoRFM handle bound to a graph, offering the familiar
    ``model.predict(query, ...)`` call from the old SDK.

    Returned by :meth:`SDFMClient.kumorfm`. It is a thin, stateless wrapper: the
    graph is supplied once, and each :meth:`predict` builds a
    :class:`KumoRFMRequest` and runs it through the owning client, so advanced
    knobs are plain keyword arguments instead of an ``options`` dict.

    >>> model = client.kumorfm(graph)  # doctest: +SKIP
    >>> model.predict("PREDICT ... FOR ...", [1, 2, 3], run_mode="fast")
    """

    def __init__(self, client: Any, graph: Any) -> None:
        self._client = client
        self._graph = graph

    def predict(
        self,
        query: str,
        indices: Sequence[Any] | None = None,
        *,
        run_mode: str = 'fast',
        explain: bool | ExplainConfig | dict[str, Any] = False,
        batch_size: int | Literal['max'] | None = None,
        num_retries: int = 1,
        anchor_time: Any = _UNSET,
        context_anchor_time: Any = _UNSET,
        use_prediction_time: Any = _UNSET,
        lag_timesteps: Any = _UNSET,
        num_neighbors: Any = _UNSET,
        num_hops: Any = _UNSET,
        inference_config: Any = _UNSET,
        return_embeddings: Any = _UNSET,
        random_seed: Any = _UNSET,
        max_pq_iterations: Any = _UNSET,
    ) -> PredictResult:
        options = {
            name: value
            for name, value in (
                ('anchor_time', anchor_time),
                ('context_anchor_time', context_anchor_time),
                ('use_prediction_time', use_prediction_time),
                ('lag_timesteps', lag_timesteps),
                ('num_neighbors', num_neighbors),
                ('num_hops', num_hops),
                ('inference_config', inference_config),
                ('return_embeddings', return_embeddings),
                ('random_seed', random_seed),
                ('max_pq_iterations', max_pq_iterations),
            )
            if value is not _UNSET
        }
        request = KumoRFMRequest(
            graph=self._graph,
            query=query,
            indices=indices,
            run_mode=run_mode,
            explain=explain,
            batch_size=batch_size,
            num_retries=num_retries,
            options=options,
        )
        return self._client._predict(request)

    def __repr__(self) -> str:
        return 'RFMModel()'


class TabICLModel:
    r"""A TabICL handle bound to a labelled context table.

    Returned by :meth:`SDFMClient.tabicl`. The context, task and target are
    supplied once; each :meth:`predict` scores a new table of unlabelled rows,
    building a :class:`TabICLRequest` under the owning client.

    >>> model = client.tabicl(context_df, target="y", task="classification")  # doctest: +SKIP
    >>> model.predict(new_rows)
    """

    def __init__(
        self,
        client: Any,
        context: pd.DataFrame,
        task: str,
        target: str,
    ) -> None:
        self._client = client
        self._context = context
        self._task = task
        self._target = target

    def predict(
        self,
        predict: pd.DataFrame,
        *,
        outputs: Any = _UNSET,
        positive_class: Any = _UNSET,
        prediction_statistic: Any = _UNSET,
        quantile_levels: Any = _UNSET,
        score_format: Any = _UNSET,
        embedding_dtype: Any = _UNSET,
        max_results: Any = _UNSET,
        request_id: Any = _UNSET,
    ) -> pd.DataFrame:
        extra = {
            name: value
            for name, value in (
                ('outputs', outputs),
                ('positive_class', positive_class),
                ('prediction_statistic', prediction_statistic),
                ('quantile_levels', quantile_levels),
                ('score_format', score_format),
                ('embedding_dtype', embedding_dtype),
                ('max_results', max_results),
                ('request_id', request_id),
            )
            if value is not _UNSET
        }
        request = TabICLRequest(
            context=self._context,
            predict=predict,
            task=self._task,
            target=self._target,
            **extra,
        )
        return self._client._predict(request)

    def __repr__(self) -> str:
        return 'TabICLModel()'

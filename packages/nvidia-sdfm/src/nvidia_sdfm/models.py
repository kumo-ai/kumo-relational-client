# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Sequence

import pandas as pd

from nvidia_sdfm.requests import (
    KumoRFMRequest,
    KumoRFMTaskRequest,
    TabICLRequest,
)

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

    def predict_task(
        self,
        context: pd.DataFrame,
        predict: pd.DataFrame,
        *,
        task_type: str,
        entity_table: str | Sequence[str],
        run_mode: str = 'fast',
        explain: bool | ExplainConfig | dict[str, Any] = False,
        batch_size: int | Literal['max'] | None = None,
        num_retries: int = 1,
        entity_column: str = 'ENTITY',
        target_column: str = 'TARGET',
        time_column: str | None = None,
        num_forecasts: int = 1,
        step_size: int | None = None,
        num_neighbors: Any = _UNSET,
        num_hops: Any = _UNSET,
        inference_config: Any = _UNSET,
        use_prediction_time: Any = _UNSET,
        return_embeddings: Any = _UNSET,
        random_seed: Any = _UNSET,
    ) -> PredictResult:
        r"""Predict from a caller-supplied train table instead of a PQL query.

        ``context`` holds the labelled (train) rows and ``predict`` the rows to
        score, both referencing entities of ``entity_table`` in the bound graph
        (a ``(source, target)`` pair for temporal link prediction). Columns
        default to ``ENTITY`` / ``TARGET`` / ``ANCHOR_TIMESTAMP`` (the
        prediction-output convention) and are overridable.

        ``task_type`` is one of ``binary_classification``,
        ``multiclass_classification``, ``regression``, ``forecasting`` or
        ``temporal_link_prediction`` (also on ``capabilities("kumo-rfm").tasks``).

        >>> model = client.kumorfm(graph)  # doctest: +SKIP
        >>> model.predict_task(
        ...     context=train_df, predict=predict_df,
        ...     task_type="multiclass_classification", entity_table="users")
        """
        options = {
            name: value
            for name, value in (
                ('num_neighbors', num_neighbors),
                ('num_hops', num_hops),
                ('inference_config', inference_config),
                ('use_prediction_time', use_prediction_time),
                ('return_embeddings', return_embeddings),
                ('random_seed', random_seed),
            )
            if value is not _UNSET
        }
        request = KumoRFMTaskRequest(
            graph=self._graph,
            context=context,
            predict=predict,
            task_type=task_type,
            entity_table=entity_table,
            entity_column=entity_column,
            target_column=target_column,
            time_column=time_column,
            num_forecasts=num_forecasts,
            step_size=step_size,
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
        r"""Scores ``predict`` against the bound context table.

        Args:
            predict: The unlabelled rows to score. Must share the context's
                feature columns and hold at least one row.
            outputs: The fields to return, defaulting to ``['prediction']``.
                TabICL produces ``'probabilities'`` for a classification task
                and ``'quantiles'`` for a regression one; asking for a field
                the task cannot produce raises.
            positive_class: The class to treat as positive in a binary
                classification. Must be one of the classes in the context's
                target column. Currently ignored by the TabICL NIM.
            prediction_statistic: The statistic to reduce a regression
                prediction with, e.g. ``'mean'``.
            quantile_levels: The quantiles to return alongside a regression
                prediction, each strictly between 0 and 1.
            score_format: Requested encoding of ``'probabilities'``.
                Currently ignored by the TabICL NIM, which always returns an
                object keyed by class.
            embedding_dtype: Requested encoding of returned embeddings.
                Currently ignored by the TabICL NIM.
            max_results: Requested cap on the number of returned rows.
                Currently ignored by the TabICL NIM, which scores every row.
            request_id: An id to correlate this request with server logs. One
                is generated when omitted.

        A classification context may declare at most 10 distinct classes.
        """
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

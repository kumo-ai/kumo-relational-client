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
    TabICLSession,
)

if TYPE_CHECKING:
    from nvidia_sdfm.base import PredictResult
    from nvidia_sdfm.kumorfm import ExplainConfig

class _Unset:
    r"""Sentinel for "argument not supplied".

    A plain ``object()`` renders as ``<object object at 0x...>`` in ``help()``,
    IDE hovers and generated docs, for most of ``RFMModel.predict``'s
    parameters. This one has a readable ``repr``.
    """

    def __repr__(self) -> str:
        return '<unset>'


_UNSET: Any = _Unset()


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
        verbose: Any = _UNSET,
        **engine_options: Any,
    ) -> PredictResult:
        r"""Runs ``query`` against the bound graph.

        Args:
            query: The predictive query (PQL).
            indices: The entity primary keys to predict for. Overrides the
                indices given in the query. Predictions are generated for every
                index, whether or not it satisfies the query's entity filters.
            run_mode: ``'fast'``, ``'normal'`` or ``'best'`` -- how much
                context the model is given. Explanations require ``'fast'``.
            explain: ``True``, a ``kumorfm.ExplainConfig`` or its dict form.
                Limits the request to a single entity and makes this call
                return an ``Explanation`` instead of a DataFrame. Filling in
                ``Explanation.summary`` sends the query, predictions and raw
                subgraph cell values to an external LLM endpoint; pass
                ``explain=dict(skip_summary=True)`` to keep them local.
            batch_size: Entities per request, or ``'max'``. ``None`` sends
                every index in one request.
            num_retries: Retries the driver performs when the NIM rejects a
                request as too large.
            anchor_time: The anchor timestamp for the prediction. ``None`` uses
                the maximum timestamp in the data; ``'entity'`` uses each
                entity's own timestamp.
            context_anchor_time: The maximum anchor timestamp for context
                (in-context train) examples. ``None`` derives it from
                ``anchor_time``.
            use_prediction_time: Whether to use the anchor timestamp as an
                additional feature at prediction time.
            lag_timesteps: The number of past timesteps included as lagged
                features.
            num_neighbors: The number of neighbors to sample per hop. Takes
                precedence over ``num_hops``.
            num_hops: The number of hops to sample when generating the context.
            inference_config: Inference-time model configuration, e.g.
                ``dict(num_estimators=4, output_type='quantiles')``. See
                ``kumorfm.rfm.KumoRFM.predict`` for the supported keys; keys
                outside that set are dropped silently.
            return_embeddings: Whether to also return an embedding per
                prediction example.
            random_seed: A manual seed for pseudo-random sampling. The
                ``sqlite`` and ``snowflake`` backends cannot seed their random
                row sampling and warn once when a seed is given. Passing
                ``None`` re-samples neighborhoods per batch, which rules out
                the shared-context session a multi-batch prediction otherwise
                opens, so each batch re-uploads the full context.
            max_pq_iterations: The maximum number of label-collection
                iterations. Raise it when the query has strict entity filters.
            verbose: Whether to print progress. Pass ``False`` in a service, a
                pipeline or anything else parsing stdout.
            engine_options: Any further keyword argument the engine's
                ``predict`` accepts, forwarded unchanged. An argument it does
                not accept raises ``SdfmError('INVALID_REQUEST')``.

        Returns:
            The predictions as a ``pd.DataFrame``, or a ``kumorfm``
            ``Explanation`` when ``explain`` is set.
        """
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
                ('verbose', verbose),
            )
            if value is not _UNSET
        }
        options.update(engine_options)
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
        top_k: Any = _UNSET,
        exclude_cols_dict: Any = _UNSET,
        verbose: Any = _UNSET,
        **engine_options: Any,
    ) -> PredictResult:
        r"""Predict from a caller-supplied train table instead of a PQL query.

        ``context`` holds the labelled (train) rows and ``predict`` the rows to
        score, both referencing entities of ``entity_table`` in the bound graph
        (a ``(source, target)`` pair for temporal link prediction). Columns
        default to ``ENTITY`` / ``TARGET`` / ``ANCHOR_TIMESTAMP`` (the
        prediction-output convention) and are overridable.

        >>> model = client.kumorfm(graph)  # doctest: +SKIP
        >>> model.predict_task(
        ...     context=train_df, predict=predict_df,
        ...     task_type="multiclass_classification", entity_table="users")

        Args:
            context: The labelled (train) rows.
            predict: The rows to score.
            task_type: One of ``'binary_classification'``,
                ``'multiclass_classification'``, ``'regression'``,
                ``'forecasting'`` or ``'temporal_link_prediction'`` (also on
                ``capabilities('kumo-rfm').tasks``).
            entity_table: The graph table the ``entity_column`` values refer
                to, or a ``(source, target)`` pair for temporal link
                prediction.
            run_mode: As in :meth:`predict`.
            explain: As in :meth:`predict`.
            batch_size: As in :meth:`predict`.
            num_retries: As in :meth:`predict`.
            entity_column: The entity-id column in ``context`` / ``predict``.
            target_column: The label column in ``context``.
            time_column: The anchor-timestamp column. ``None`` uses
                ``ANCHOR_TIMESTAMP`` when either frame carries it, otherwise
                the entity table's own time column.
            num_forecasts: For ``task_type='forecasting'``, how many steps to
                forecast. Ignored otherwise.
            step_size: **Required for** ``task_type='forecasting'``: the
                spacing between forecast steps, as an integer number of
                **nanoseconds** -- e.g. ``int(pd.Timedelta(days=30).value)``
                for monthly steps. Omitting it is rejected by the NIM, not
                client-side. Ignored for every other task type.
            num_neighbors: As in :meth:`predict`.
            num_hops: As in :meth:`predict`.
            inference_config: As in :meth:`predict`.
            use_prediction_time: As in :meth:`predict`.
            return_embeddings: As in :meth:`predict`.
            random_seed: As in :meth:`predict`.
            top_k: For ``task_type='temporal_link_prediction'``, how many
                ranked items to return per entity. The PQL path spells this
                ``RANK TOP k``. Ignored for every other task type.
            exclude_cols_dict: Columns to withhold from the model input, as
                ``{table_name: [column, ...]}``.
            verbose: As in :meth:`predict`.
            engine_options: As in :meth:`predict`, for the engine's
                ``predict_task``.

        Returns:
            The predictions as a ``pd.DataFrame``, or a ``kumorfm``
            ``Explanation`` when ``explain`` is set.
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
                ('top_k', top_k),
                ('exclude_cols_dict', exclude_cols_dict),
                ('verbose', verbose),
            )
            if value is not _UNSET
        }
        options.update(engine_options)
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

    Reusing one handle for many scoring calls is the cheap path: the NIM pins
    the context after the first call, so later calls send only the rows to
    score. Against a NIM without session routes every call carries the context,
    as it always did; either way the predictions are the same.

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
        self._session = TabICLSession()

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
            session=self._session,
            **extra,
        )
        return self._client._predict(request)

    def __del__(self) -> None:
        # Releases the pinned context when the handle goes away. Best effort:
        # this runs on the garbage collector's schedule, possibly at
        # interpreter shutdown and possibly after the client was closed, and
        # the NIM reaps the session on its own TTL regardless.
        session = getattr(self, '_session', None)
        if session is None or session.id is None:
            return
        try:
            from nvidia_sdfm.adapters.tabicl import delete_session_quietly

            delete_session_quietly(self._client._transport, session.id)
        except Exception:
            pass

    def __repr__(self) -> str:
        return 'TabICLModel()'

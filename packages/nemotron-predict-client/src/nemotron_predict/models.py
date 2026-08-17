# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like

from nemotron_predict.errors import PredictError
from nemotron_predict.requests import (
    NemotronRelationalRequest,
    NemotronRelationalTaskRequest,
    NemotronTabularRequest,
    NemotronTabularSession,
)

if TYPE_CHECKING:
    from nemotron_predict.base import PredictResult
    from nemotron_predict.relational import ExplainConfig


def require_frame(value: Any, name: str) -> pd.DataFrame:
    r"""Reject a table argument that is not a ``DataFrame``, naming the caller's
    parameter.

    Every table these handles take is used as a frame immediately, so a
    ``dict``, a list of records, a numpy array or a ``Series`` -- the shapes a
    pandas user reaches for before building a frame -- otherwise surfaces as
    ``AttributeError: 'dict' object has no attribute 'columns'`` from inside an
    adapter, which ``except PredictError`` does not catch and which names neither
    the argument nor the type it should have been.
    """
    if not isinstance(value, pd.DataFrame):
        hint = (
            '; wrap a single row with pd.DataFrame([row])'
            if isinstance(value, (dict, pd.Series))
            else '; build one with pd.DataFrame(...)'
        )
        raise PredictError(
            f'{name} must be a pandas DataFrame, got '
            f'{type(value).__name__}{hint}',
            code='INVALID_REQUEST',
        )
    return value


class _Unset:
    r"""Sentinel for "argument not supplied".

    A plain ``object()`` renders as ``<object object at 0x...>`` in ``help()``,
    IDE hovers and generated docs, for most of ``RelationalModel.predict``'s
    parameters. This one has a readable ``repr``.
    """

    def __repr__(self) -> str:
        return '<unset>'


_UNSET: Any = _Unset()

_RFM_RUN_MODES = ('debug', 'fast', 'normal', 'best')


def _validate_run_mode(value: Any) -> None:
    r"""Name every accepted mode instead of leaking the engine enum error."""
    if not isinstance(value, str) or value not in _RFM_RUN_MODES:
        allowed = ', '.join(repr(mode) for mode in _RFM_RUN_MODES)
        raise PredictError(
            f'run_mode must be one of {allowed}, got '
            f'{type(value).__name__} {value!r}',
            code='INVALID_REQUEST',
        )


def _validate_num_neighbors(value: Any) -> list[int] | None:
    r"""Keep malformed sampling shapes out of the native RFM sampler."""
    valid = value is None or (
        isinstance(value, list)
        and len(value) <= 6
        and all(
            isinstance(item, Integral)
            and not isinstance(item, bool)
            and item >= 0
            for item in value
        )
    )
    if not valid:
        raise PredictError(
            'num_neighbors must be None or a list of at most 6 non-negative '
            f'integers, got {type(value).__name__} ({value!r})',
            code='INVALID_REQUEST',
        )
    if value is None:
        return None
    return [int(item) for item in value]


def _validate_integer_option(
    name: str,
    value: Any,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    r"""Reject booleans, wrong types and out-of-range integer options."""
    valid = isinstance(value, Integral) and not isinstance(value, bool)
    if valid:
        valid = value >= minimum and (maximum is None or value <= maximum)
    if valid:
        return int(value)
    expected = f'an integer greater than or equal to {minimum}'
    if maximum is not None:
        expected = f'an integer between {minimum} and {maximum}'
    raise PredictError(
        f'{name} must be {expected}, got {type(value).__name__} {value!r}',
        code='INVALID_REQUEST',
    )


def _validate_bool_option(name: str, value: Any) -> bool:
    r"""Reject truthy strings and integers before they change RFM behavior."""
    if not isinstance(value, (bool, np.bool_)):
        raise PredictError(
            f'{name} must be a bool, got {type(value).__name__} {value!r}',
            code='INVALID_REQUEST',
        )
    return bool(value)


def _validate_random_seed(value: Any) -> int | None:
    r"""Reject values that would otherwise leak NumPy SeedSequence errors."""
    if value is not None and (
        not isinstance(value, Integral) or isinstance(value, bool) or value < 0
    ):
        raise PredictError(
            'random_seed must be None or a non-negative integer, got '
            f'{type(value).__name__} ({value!r})',
            code='INVALID_REQUEST',
        )
    return None if value is None else int(value)


class RelationalModel:
    r"""A Nemotron Relational handle bound to a graph, offering the familiar
    ``model.predict(query, ...)`` call from the old client.

    Returned by :meth:`PredictClient.nemotron_relational`. It is a thin, stateless wrapper: the
    graph is supplied once, and each :meth:`predict` builds a
    :class:`NemotronRelationalRequest` and runs it through the owning client, so advanced
    knobs are plain keyword arguments instead of an ``options`` dict.

    >>> model = client.relational(graph)  # doctest: +SKIP
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
            run_mode: ``'debug'``, ``'fast'``, ``'normal'`` or ``'best'`` --
                how much context the model is given. Explanations require
                ``'fast'``.
            explain: ``True``, a ``nemotron_relational.ExplainConfig`` or its
                dict form.
                Limits the request to a single entity and makes this call
                return an ``Explanation`` instead of a DataFrame. Filling in
                ``Explanation.summary`` sends the query, predictions and raw
                subgraph cell values to an external LLM endpoint; pass
                ``explain=dict(skip_summary=True)`` to keep them local.
            batch_size: Entities per request, or ``'max'``. ``None`` sends
                every index in one request.
            num_retries: Application-level retries of a failed prediction
                request, with exponential backoff. Only transient failures
                (408, 429, 500, 502, 503, 504, timeouts, dropped connections)
                are retried; a refusal such as an oversized payload is not.
                Distinct from
                ``PredictClient(max_retries=...)``, which retries at the
                transport.
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
                ``nemotron_relational.rfm.NemotronRelational.predict`` for the supported keys; a key
                outside that set is rejected rather than dropped.
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
                not accept raises ``PredictError('INVALID_REQUEST')``.

        Returns:
            The predictions as a ``pd.DataFrame``, or a ``nemotron_relational``
            ``Explanation`` when ``explain`` is set (the frame is then on its
            ``prediction`` attribute).

            **The row count depends on the task**, so do not assume one row per
            index. Every row carries ``ENTITY`` and ``ANCHOR_TIMESTAMP``; join
            on ``ENTITY`` rather than on position.

            =========================== ====================================
            Task                        Rows and columns
            =========================== ====================================
            binary classification       One row per entity. ``PREDICTION``,
                                        ``FALSE_PROB``, ``TRUE_PROB``.
            regression                  One row per entity. ``PREDICTION``.
            multiclass classification   One row per entity *per class*.
                                        ``CLASS``, ``SCORE``, ``PREDICTED``.
            ``RANK TOP k``              *k* rows per entity, best first.
            (temporal link prediction)  ``CLASS``, ``SCORE``.
            =========================== ====================================

            A query whose target is a categorical column is a multiclass task,
            so it is reached from here as well as from :meth:`predict_task`.

            ``return_embeddings=True`` adds an ``EMBEDDINGS`` column without
            changing the row count. See ``docs/reference/prediction-output.md``.
        """
        if indices is not None and not is_list_like(indices):
            raise PredictError(
                'indices must be a list-like collection of entity IDs, got '
                f'{type(indices).__name__} ({indices!r}); for one entity ID, '
                f'pass indices=[{indices!r}]',
                code='INVALID_REQUEST',
            )
        if num_neighbors is not _UNSET:
            num_neighbors = _validate_num_neighbors(num_neighbors)
        if num_hops is not _UNSET:
            num_hops = _validate_integer_option(
                'num_hops', num_hops, minimum=1, maximum=6
            )
        if max_pq_iterations is not _UNSET:
            max_pq_iterations = _validate_integer_option(
                'max_pq_iterations', max_pq_iterations, minimum=1
            )
        if lag_timesteps is not _UNSET:
            lag_timesteps = _validate_integer_option(
                'lag_timesteps', lag_timesteps, minimum=0
            )
        if return_embeddings is not _UNSET:
            return_embeddings = _validate_bool_option(
                'return_embeddings', return_embeddings
            )
        if use_prediction_time is not _UNSET:
            use_prediction_time = _validate_bool_option(
                'use_prediction_time', use_prediction_time
            )
        if random_seed is not _UNSET:
            random_seed = _validate_random_seed(random_seed)
        _validate_run_mode(run_mode)
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
        request = NemotronRelationalRequest(
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

        >>> model = client.relational(graph)  # doctest: +SKIP
        >>> model.predict_task(
        ...     context=train_df, predict=predict_df,
        ...     task_type="multiclass_classification", entity_table="users")

        Args:
            context: The labelled (train) rows.
            predict: The rows to score.
            task_type: One of ``'binary_classification'``,
                ``'multiclass_classification'``, ``'regression'``,
                ``'forecasting'`` or ``'temporal_link_prediction'`` (also on
                ``capabilities('nemotron-relational').tasks``).
            entity_table: The graph table the ``entity_column`` values refer
                to, or a ``(source, target)`` pair for temporal link
                prediction.
            run_mode: As in :meth:`predict`.
            explain: As in :meth:`predict`.
            batch_size: As in :meth:`predict`.
            num_retries: As in :meth:`predict`.
            entity_column: The entity-id column in ``context`` / ``predict``.
            target_column: The label column in ``context``. For
                ``task_type='temporal_link_prediction'`` each value must be a
                *list* of target ids, not a single id; a scalar is rejected
                with "Link prediction target values must be stringlist
                arrays".
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
            The predictions as a ``pd.DataFrame``, or a ``nemotron_relational``
            ``Explanation`` when ``explain`` is set (the frame is then on its
            ``prediction`` attribute).

            **The row count depends on ``task_type``**, so do not assume one
            row per row of ``predict``. Every row carries ``ENTITY`` and
            ``ANCHOR_TIMESTAMP``; join on ``ENTITY`` rather than on position.

            ============================= ==================================
            ``task_type``                 Rows and columns
            ============================= ==================================
            ``binary_classification``     One row per entity. ``PREDICTION``,
                                          ``FALSE_PROB``, ``TRUE_PROB``.
            ``regression``                One row per entity. ``PREDICTION``.
            ``multiclass_classification`` One row per entity *per class*,
                                          best first. ``CLASS``, ``SCORE``,
                                          ``PREDICTED``.
            ``temporal_link_prediction``  ``top_k`` rows per entity, best
                                          first. ``CLASS``, ``SCORE``.
            ``forecasting``               ``num_forecasts`` rows per entity.
                                          ``PREDICTION``, ``FORECAST_STEP``.
            ============================= ==================================

            For multiclass, ``PREDICTED`` is ``True`` on the single winning
            row per entity, so ``frame[frame['PREDICTED']]`` recovers one row
            per entity. ``return_embeddings=True`` adds an ``EMBEDDINGS``
            column without changing the row count. See
            ``docs/reference/prediction-output.md``.
        """
        if num_neighbors is not _UNSET:
            num_neighbors = _validate_num_neighbors(num_neighbors)
        if num_hops is not _UNSET:
            num_hops = _validate_integer_option(
                'num_hops', num_hops, minimum=1, maximum=6
            )
        if return_embeddings is not _UNSET:
            return_embeddings = _validate_bool_option(
                'return_embeddings', return_embeddings
            )
        if use_prediction_time is not _UNSET:
            use_prediction_time = _validate_bool_option(
                'use_prediction_time', use_prediction_time
            )
        if random_seed is not _UNSET:
            random_seed = _validate_random_seed(random_seed)
        _validate_run_mode(run_mode)
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
        request = NemotronRelationalTaskRequest(
            graph=self._graph,
            context=require_frame(context, 'context'),
            predict=require_frame(predict, 'predict'),
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
        return 'RelationalModel()'


class TabularModel:
    r"""A Nemotron Tabular handle bound to a labelled context table.

    Returned by :meth:`PredictClient.tabicl`. The context, task and target are
    supplied once; each :meth:`predict` scores a new table of unlabelled rows,
    building a :class:`NemotronTabularRequest` under the owning client.

    Reusing one handle for many scoring calls is the cheap path: the NIM pins
    the context after the first call, so later calls send only the rows to
    score. Against a NIM without session routes every call carries the context,
    as it always did; either way the predictions are the same.

    >>> model = client.tabular(context_df, target="y", task="classification")  # doctest: +SKIP
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
        self._session = NemotronTabularSession()

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
                Nemotron Tabular produces ``'probabilities'`` for a classification task
                and ``'quantiles'`` for a regression one; asking for a field
                the task cannot produce raises.
            positive_class: The class to treat as positive in a binary
                classification. Must be one of the classes in the context's
                target column. Currently ignored by the Nemotron Tabular NIM.
            prediction_statistic: The statistic to reduce a regression
                prediction with, e.g. ``'mean'``.
            quantile_levels: The quantiles to return alongside a regression
                prediction, each strictly between 0 and 1.
            score_format: Requested encoding of ``'probabilities'``.
                Currently ignored by the Nemotron Tabular NIM, which always returns an
                object keyed by class.
            embedding_dtype: Requested encoding of returned embeddings.
                Currently ignored by the Nemotron Tabular NIM.
            max_results: Requested cap on the number of returned rows.
                Currently ignored by the Nemotron Tabular NIM, which scores every row.
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
        request = NemotronTabularRequest(
            context=self._context,
            predict=require_frame(predict, 'predict'),
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
            from nemotron_predict.adapters.tabular import delete_session_quietly

            delete_session_quietly(self._client._transport, session.id)
        except Exception:
            pass

    def __repr__(self) -> str:
        return 'TabularModel()'

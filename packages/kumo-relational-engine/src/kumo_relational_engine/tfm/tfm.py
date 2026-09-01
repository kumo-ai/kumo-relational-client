# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Sequence
from typing import Literal

import pandas as pd

from kumo_relational_engine.api.pquery import (
    QueryType,
    ValidatedPredictiveQuery,
)
from kumo_relational_engine.pql.parser.parser import QueryValidationType
from kumo_relational_engine.rfm import Graph, TaskTable
from kumo_relational_engine.rfm.base import DataBackend, Sampler
from kumo_relational_engine.rfm.query_parser import parse_query_locally
from kumo_relational_engine.rfm.rfm import (
    _MAX_TEST_SIZE,
    _OPTIMIZABLE_BACKENDS,
    _RANDOM_SEED,
    _check_anchor_time,
    _TaskSetupMixin,
)
from kumo_relational_engine.utils import ProgressLogger

_NOT_YET_IMPLEMENTED = (
    'KumoTabular cannot answer a query yet. This release validates a '
    'predictive query against the graph and the tabular task gate and '
    'generates its labels; feature assembly and prediction land in a later '
    'release. Use KumoRelational to make predictions today.'
)

# How many candidate rows the sampler is allowed to try per example it is
# asked for. Mirrors the ``max_pq_iterations`` default of
# :meth:`KumoRelational._get_task_table`.
_MAX_PQ_ITERATIONS = 10


class KumoTabular(_TaskSetupMixin):
    r"""Run Kumo Tabular predictions over a relational graph.

    :class:`KumoTabular` is the tabular counterpart of
    :class:`~kumo_relational_engine.rfm.KumoRelational`: it takes the same
    :class:`~kumo_relational_engine.rfm.Graph` and the same PQL. It processes
    the data, converts it to a single table, and builds the context table the
    tabular foundation model predicts from. The model scores one label per
    entity, so it serves binary classification, multiclass classification and
    regression.

    .. code-block:: python

        from kumo_relational_engine.rfm import Graph
        from kumo_relational_engine.tfm import KumoTabular

        graph = Graph.from_data({
            'users': df_users,
            'orders': df_orders,
        })

        tabular = KumoTabular(graph)

        query = ("PREDICT COUNT(orders.*, 0, 30, days) > 0 "
                 "FOR EACH users.user_id")
        result = tabular.predict(query, indices=[1, 2], context_size=100)

    Args:
        graph: The graph.
        verbose: Whether to print verbose output.
        optimize: If set to ``True``, will optimize the underlying data backend
            for optimal querying. For example, for transactional database
            backends, will create any missing indices. Requires write-access to
            the data backend. Only the :obj:`"sqlite"` and :obj:`"duckdb"`
            backends implement this; passing it on any other backend warns.
    """

    def __init__(
        self,
        graph: Graph,
        verbose: bool | ProgressLogger = True,
        optimize: bool = False,
    ) -> None:
        graph = graph.validate()
        self._graph_def = graph._to_api_graph_definition()

        # NOTE: This backend dispatch is a copy of the one in
        # :meth:`KumoRelational.__init__`. The two are meant to stay
        # identical; folding them into one factory is deferred until the
        # tabular pathway knows which of the backends it actually serves.
        if optimize and graph.backend not in _OPTIMIZABLE_BACKENDS:
            warnings.warn(
                f"'optimize=True' has no effect on the "
                f"'{graph.backend.value}' backend; it is implemented "
                f'only for '
                f'{sorted(b.value for b in _OPTIMIZABLE_BACKENDS)}'
            )

        if graph.backend == DataBackend.LOCAL:
            from kumo_relational_engine.rfm.backend.local import LocalSampler

            self._sampler: Sampler = LocalSampler(graph, verbose)
        elif graph.backend == DataBackend.SQLITE:
            from kumo_relational_engine.rfm.backend.sqlite import SQLiteSampler

            self._sampler = SQLiteSampler(graph, verbose, optimize)
        elif graph.backend == DataBackend.DUCKDB:
            from kumo_relational_engine.rfm.backend.duckdb import DuckDBSampler

            self._sampler = DuckDBSampler(graph, verbose, optimize)
        elif graph.backend == DataBackend.SNOWFLAKE:
            from kumo_relational_engine.rfm.backend.snow import SnowSampler

            self._sampler = SnowSampler(graph, verbose)
        elif graph.backend == DataBackend.DATABRICKS:
            from kumo_relational_engine.rfm.backend.databricks import (
                DatabricksSampler,
            )

            self._sampler = DatabricksSampler(graph, verbose)
        else:
            raise NotImplementedError

    def predict(
        self,
        query: str | ValidatedPredictiveQuery,
        indices: Sequence[str] | Sequence[float] | Sequence[int] | None = None,
        *,
        context_size: int,
    ) -> pd.DataFrame:
        r"""Returns predictions for a predictive query.

        Args:
            query: The predictive query.
            indices: The entity primary keys to predict for. Will override the
                indices given as part of the predictive query.
            context_size: How many in-context examples to label. There is no
                default: the tabular model has no run modes to read one from,
                so the caller states the budget.

        Raises:
            ValueError: If the query is not valid for the tabular model.
            NotImplementedError: Always, once the query validates and its
                labels are generated. Prediction is not wired up yet.
        """
        if isinstance(query, str):
            parse_query_locally(
                query,
                self._graph_def,
                QueryValidationType.TFM,
            )
        raise NotImplementedError(
            'KumoTabular cannot answer a query yet. This release validates a '
            'predictive query against the graph and the tabular task gate; '
            'label generation, feature assembly and prediction land in a '
            'later release. Use KumoRelational to make predictions today.'
        )

    def _generate_labels(
        self,
        query: ValidatedPredictiveQuery,
        context_size: int,
        anchor_time: pd.Timestamp |  None = None,
        context_anchor_time: pd.Timestamp | None = None,
    ) -> TaskTable:
        r"""Turns a validated query into a :class:`TaskTable`: in-context
        examples carrying their ground-truth label, and the prediction
        examples the model is asked about.

        Args:
            query: The validated predictive query.
            context_size: How many in-context examples to label. Fewer come
                back when the data cannot support that many; the sampler warns
                or raises when it falls short.
            anchor_time: The anchor timestamp of the prediction examples. If
                ``None``, derived from the latest timestamp the query
                aggregates over. If ``"entity"``, each entity is anchored at
                its own timestamp, which only a static query supports.
            context_anchor_time: The anchor timestamp of the in-context
                examples. If ``None``, placed one aggregation window before
                ``anchor_time``, so labelling the context cannot read data
                from after the prediction anchor.
        """
        anchor_time = _check_anchor_time(anchor_time, 'anchor_time')
        context_anchor_time = _check_anchor_time(
            context_anchor_time, 'context_anchor_time'
        )
        if context_size < 1:
            raise ValueError(
                f"'context_size' must be greater than zero (got {context_size})"
            )

        task_type = self._get_task_type(
            query=query,
            edge_types=self._sampler.edge_types,
        )
        num_pred_examples = _MAX_TEST_SIZE[task_type]

        if query.target_ast.date_offset_range is None:
            step_offset = pd.DateOffset(0)
        else:
            step_offset = query.target_ast.date_offset_range.end_date_offset

        if anchor_time is None:
            anchor_time = self._get_default_anchor_time(query)
            anchor_time = anchor_time - step_offset * query.num_forecasts

        if isinstance(anchor_time, pd.Timestamp):
            if context_anchor_time == 'entity':
                raise ValueError(
                    "Anchor time 'entity' needs to be shared "
                    'for context and prediction examples'
                )
            if context_anchor_time is None:
                context_anchor_time = anchor_time - step_offset
            self._validate_time(
                query,
                anchor_time,
                context_anchor_time,
                evaluate=True,
            )
        else:
            assert anchor_time == 'entity'
            if query.query_type != QueryType.STATIC:
                raise ValueError(
                    "Anchor time 'entity' is only valid for "
                    'static predictive queries'
                )
            if query.entity_table not in self._sampler.time_column_dict:
                raise ValueError(
                    f"Anchor time 'entity' requires the entity "
                    f"table '{query.entity_table}' to "
                    f'have a time column'
                )
            if isinstance(context_anchor_time, pd.Timestamp):
                raise ValueError(
                    "Anchor time 'entity' needs to be shared "
                    'for context and prediction examples'
                )
            context_anchor_time = 'entity'
        assert context_anchor_time is not None

        context, pred = self._sampler.sample_target(
            query=query,
            num_train_examples=context_size,
            train_anchor_time=context_anchor_time,
            num_train_trials=_MAX_PQ_ITERATIONS * context_size,
            num_test_examples=num_pred_examples,
            test_anchor_time=anchor_time,
            num_test_trials=_MAX_PQ_ITERATIONS * num_pred_examples,
            random_seed=_RANDOM_SEED,
        )
        context_pkey, context_time, context_y = context
        pred_pkey, pred_time, pred_y = pred

        context_df = pd.DataFrame({'ENTITY': context_pkey, 'TARGET': context_y})
        if isinstance(context_time, pd.Series):
            context_df['ANCHOR_TIMESTAMP'] = context_time
        pred_df = pd.DataFrame({'ENTITY': pred_pkey, 'TARGET': pred_y})
        if isinstance(pred_time, pd.Series):
            pred_df['ANCHOR_TIMESTAMP'] = pred_time

        return TaskTable(
            task_type=task_type,
            context_df=context_df,
            pred_df=pred_df,
            entity_table_name=(query.entity_table,),
            entity_column='ENTITY',
            target_column='TARGET',
            time_column='ANCHOR_TIMESTAMP'
            if isinstance(context_time, pd.Series)
            else TaskTable.ENTITY_TIME,
            num_forecasts=query.num_forecasts,
        )

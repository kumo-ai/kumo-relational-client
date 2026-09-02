# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import TYPE_CHECKING, Any

import pandas as pd

from kumo_relational_engine.api.pquery import (
    QueryType,
    ValidatedPredictiveQuery,
)
from kumo_relational_engine.api.pquery.AST import (
    Aggregation,
    Column,
    Condition,
    Join,
    LogicalOperation,
)
from kumo_relational_engine.api.task import TaskType
from kumo_relational_engine.api.typing import (
    AggregationType,
    ProblemType,
    Stype,
)
from kumo_relational_engine.core.utils import to_naive_utc

if TYPE_CHECKING:
    from kumo_relational_engine.rfm.base import Sampler


class TaskSetupMixin:
    r"""The task typing and anchor-time derivation a predictive query
    needs before its labels can be sampled.

    It lives on a mixin because :class:`KumoRelational` and the tabular
    :class:`~kumo_relational_engine.tfm.KumoTabular` both build a
    :class:`TaskTable` from the same query against the same sampler, and
    the two must agree on the task type and the anchor times or the same
    query would mean different things to the two models.
    """

    _sampler: 'Sampler'

    @staticmethod
    def check_anchor_time(value: Any, name: str) -> Any:
        r"""Reject an anchor time that is neither a ``Timestamp`` nor
        ``'entity'``, and return it converted to the timezone-naive UTC the
        graph is held in.

        A date *string* is what most pandas users reach for first, and it used
        to land on a bare ``assert`` with an empty message, and under
        ``python -O`` on no check at all. A timezone-*aware* ``Timestamp`` is
        what the client's own ``predict`` output carries, so it is converted
        rather than refused.
        """
        if value is None or isinstance(value, pd.Timestamp):
            return to_naive_utc(value)
        if isinstance(value, str) and value == 'entity':
            return value
        hint = (
            f'; try pd.Timestamp({value!r})' if isinstance(value, str) else ''
        )
        raise TypeError(
            f"'{name}' must be a pandas.Timestamp or the literal 'entity' (got "
            f'{type(value).__name__} {value!r}){hint}'
        )

    @staticmethod
    def get_task_type(
        query: ValidatedPredictiveQuery,
        edge_types: list[tuple[str, str, str]],
    ) -> TaskType:
        if query.problem_type == ProblemType.FORECAST:
            return TaskType.FORECASTING

        if isinstance(query.target_ast, (Condition, LogicalOperation)):
            return TaskType.BINARY_CLASSIFICATION

        target = query.target_ast
        if isinstance(target, Join):
            target = target.rhs_target
        if isinstance(target, Aggregation):
            if target.aggr == AggregationType.LIST_DISTINCT:
                table_name, col_name = target.get_target_column_name().split(
                    '.'
                )
                target_edge_types = [
                    edge_type
                    for edge_type in edge_types
                    if edge_type[0] == table_name and edge_type[1] == col_name
                ]
                if len(target_edge_types) != 1:
                    raise NotImplementedError(
                        f'Multilabel-classification queries based on '
                        f"'LIST_DISTINCT' are not supported yet. If you "
                        f'planned to write a link prediction query instead, '
                        f"make sure to register '{col_name}' as a "
                        f'foreign key.'
                    )
                return TaskType.TEMPORAL_LINK_PREDICTION

            return TaskType.REGRESSION

        assert isinstance(target, Column)

        if target.stype in {Stype.ID, Stype.categorical}:
            return TaskType.MULTICLASS_CLASSIFICATION

        if target.stype in {Stype.numerical}:
            return TaskType.REGRESSION

        raise NotImplementedError('Task type not yet supported')

    def get_default_anchor_time(
        self,
        query: ValidatedPredictiveQuery | None = None,
    ) -> pd.Timestamp:
        if query is not None and query.query_type == QueryType.TEMPORAL:
            aggr_table_names = [
                aggr.get_target_column_name().split('.')[0]
                for aggr in query.get_all_target_aggregations()
            ]
            return self._sampler.get_max_time(aggr_table_names)

        return self._sampler.get_max_time()

    def validate_time(
        self,
        query: ValidatedPredictiveQuery,
        anchor_time: pd.Timestamp,
        context_anchor_time: pd.Timestamp | None,
        evaluate: bool,
    ) -> None:

        if len(self._sampler.time_column_dict) == 0:
            return  # Graph without timestamps

        if query.query_type == QueryType.TEMPORAL:
            aggr_table_names = [
                aggr.get_target_column_name().split('.')[0]
                for aggr in query.get_all_target_aggregations()
            ]
            min_time = self._sampler.get_min_time(aggr_table_names)
            max_time = self._sampler.get_max_time(aggr_table_names)
        else:
            min_time = self._sampler.get_min_time()
            max_time = self._sampler.get_max_time()

        if anchor_time < min_time:
            raise ValueError(
                f"Anchor timestamp '{anchor_time}' is before "
                f"the earliest timestamp '{min_time}' in the "
                f'data.'
            )

        if context_anchor_time is not None and context_anchor_time < min_time:
            raise ValueError(
                f'Context anchor timestamp is too early or '
                f'aggregation time range is too large. To make '
                f'this prediction, we would need data back to '
                f"'{context_anchor_time}', however, your data "
                f"only contains data back to '{min_time}'."
            )

        if query.target_ast.date_offset_range is not None:
            end_offset = query.target_ast.date_offset_range.end_date_offset
        else:
            end_offset = pd.DateOffset(0)

        if (
            context_anchor_time is not None
            and context_anchor_time > anchor_time
        ):
            warnings.warn(
                f'Context anchor timestamp '
                f"(got '{context_anchor_time}') is set to a later "
                f'date than the prediction anchor timestamp '
                f"(got '{anchor_time}'). Please make sure this is "
                f'intended.'
            )
        elif (
            query.query_type == QueryType.TEMPORAL
            and context_anchor_time is not None
            and context_anchor_time + end_offset > anchor_time
        ):
            warnings.warn(
                f'Aggregation for context examples at timestamp '
                f"'{context_anchor_time}' will leak information "
                f'from the prediction anchor timestamp '
                f"'{anchor_time}'. Please make sure this is "
                f'intended.'
            )

        elif (
            context_anchor_time is not None
            and context_anchor_time - end_offset * query.num_forecasts
            < min_time
        ):
            _time = context_anchor_time - end_offset * query.num_forecasts
            warnings.warn(
                f'Context anchor timestamp is too early or '
                f'aggregation time range is too large. To form '
                f'proper input data, we would need data back to '
                f"'{_time}', however, your data only contains "
                f"data back to '{min_time}'."
            )

        if not evaluate and anchor_time > max_time + pd.DateOffset(days=1):
            warnings.warn(
                f"Anchor timestamp '{anchor_time}' is after the "
                f"latest timestamp '{max_time}' in the data. Please "
                f'make sure this is intended.'
            )

        if (
            evaluate
            and anchor_time > max_time - end_offset * query.num_forecasts
        ):
            raise ValueError(
                f'Anchor timestamp for evaluation is after the latest '
                f"supported timestamp '{max_time - end_offset}'."
            )

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_relational_engine.api.common import (
    ValidationError,
    ValidationResponse,
)
from kumo_relational_engine.api.pquery import ParsedPredictiveQuery
from kumo_relational_engine.api.pquery.AST import (
    Aggregation,
    ASTNode,
    Join,
)
from kumo_relational_engine.api.typing import (
    AggregationType,
    ProblemType,
    Stype,
)

TRAIN_A_MODEL = (
    'To make predictions with this query, use the relational foundation '
    'model or train a new model.'
)


class TabularValidator:
    r"""This class contains validation logic specific to tabular queries.

    The tabular foundation model answers one row at a time: it scores an
    entity against a single label column. Binary classification, multiclass
    classification and regression have that shape; ranking, multilabel
    classification, forecasting and counterfactuals do not, because each of
    them asks for a label whose width or horizon is not one value per entity.

    The rules below reject those four shapes, so a query the tabular model
    cannot answer fails at validation with a diagnostic naming the clause at
    fault, rather than at prediction with a shape mismatch.
    """

    def validate(
        self, parsed_query: ParsedPredictiveQuery
    ) -> ValidationResponse:
        r"""Validates that the query names a task the tabular model serves.

        Args:
            parsed_query: the input parsed query.

        Returns:
            List of encountered errors/warnings.
        """
        for rule in (
            self._validate_ranking,
            self._validate_multilabel,
            self._validate_forecast,
            self._validate_whatif,
        ):
            response = rule(parsed_query)
            if not response.ok:
                return response
        return ValidationResponse()

    def _validate_ranking(
        self, parsed_query: ParsedPredictiveQuery
    ) -> ValidationResponse:
        response = ValidationResponse()
        if (
            parsed_query.problem_type != ProblemType.RANK
            and parsed_query.top_k is None
        ):
            return response
        response.errors.append(
            ValidationError(
                title='Unsupported query structure',
                message=(
                    f'{parsed_query.target_ast.get_location().message_start}: '
                    f'Tabular foundation model queries do not support ranking '
                    f'or link prediction. Drop the "RANK TOP k" clause and '
                    f'predict a single label per entity instead. '
                    f'{TRAIN_A_MODEL}'
                ),
            )
        )
        return response

    def _validate_multilabel(
        self, parsed_query: ParsedPredictiveQuery
    ) -> ValidationResponse:
        response = ValidationResponse()
        target_ast = parsed_query.target_ast
        if isinstance(target_ast, Join):
            # Handles the implicit join a link prediction target carries.
            target_ast = target_ast.rhs_target

        is_list_distinct = self._has_list_distinct(target_ast)
        if not (
            is_list_distinct
            or parsed_query.problem_type == ProblemType.CLASSIFY
            or target_ast.stype == Stype.multicategorical
        ):
            return response

        clause = 'LIST_DISTINCT' if is_list_distinct else 'CLASSIFY'
        response.errors.append(
            ValidationError(
                title='Unsupported query structure',
                message=(
                    f'{target_ast.get_location().message_start}: '
                    f'Tabular foundation model queries do not support '
                    f'multilabel tasks, so "{clause}" cannot be used here. '
                    f'Predict a single categorical or numerical label per '
                    f'entity instead. {TRAIN_A_MODEL}'
                ),
            )
        )
        return response

    def _validate_forecast(
        self, parsed_query: ParsedPredictiveQuery
    ) -> ValidationResponse:
        response = ValidationResponse()
        if parsed_query.problem_type != ProblemType.FORECAST:
            return response
        response.errors.append(
            ValidationError(
                title='Unsupported query structure',
                message=(
                    f'{parsed_query.target_ast.get_location().message_start}: '
                    f'Tabular foundation model queries do not support '
                    f'forecasting. Drop the "FORECAST '
                    f'{parsed_query.num_forecasts} TIMEFRAMES" clause to '
                    f'predict a single aggregate over one time range. '
                    f'{TRAIN_A_MODEL}'
                ),
            )
        )
        return response

    def _validate_whatif(
        self, parsed_query: ParsedPredictiveQuery
    ) -> ValidationResponse:
        response = ValidationResponse()
        if parsed_query.whatif_ast is None:
            return response
        response.errors.append(
            ValidationError(
                title='Unsupported query structure',
                message=(
                    f'{parsed_query.whatif_ast.get_location().message_start}: '
                    f'Tabular foundation model queries do not support '
                    f'counterfactuals. Drop the "ASSUMING" clause and predict '
                    f'against the observed data instead. {TRAIN_A_MODEL}'
                ),
            )
        )
        return response

    def _has_list_distinct(self, node: ASTNode) -> bool:
        if (
            isinstance(node, Aggregation)
            and node.aggr == AggregationType.LIST_DISTINCT
        ):
            return True
        return any(self._has_list_distinct(child) for child in node.children)

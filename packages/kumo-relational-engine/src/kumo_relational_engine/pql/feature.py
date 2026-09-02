# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from dataclasses import dataclass

from kumo_relational_engine.api.common import (
    ValidationError,
    ValidationResponse,
    ValidationWarning,
)
from kumo_relational_engine.api.graph import GraphDefinition
from kumo_relational_engine.api.pquery import ParsedPredictiveQuery
from kumo_relational_engine.api.pquery.AST import (
    Aggregation,
    ASTNode,
    Column,
    Condition,
    Join,
    LogicalOperation,
)
from kumo_relational_engine.api.pquery.AST.location_interval import (
    ASTQueryLocationInterval,
)
from kumo_relational_engine.api.typing import ProblemType
from kumo_relational_engine.pql.parser.error_translator import ErrorTranslator
from kumo_relational_engine.pql.parser.parser import (
    PQLParser,
    QueryValidationType,
)
from kumo_relational_engine.pql.validator.join_validator import JoinValidator
from kumo_relational_engine.pql.validator.predictive_query_validator import (
    PredictiveQueryValidator,
)
from kumo_relational_engine.pql.validator.time_range_validator import (
    TimeRangeValidator,
)
from kumo_relational_engine.pql.validator.type_validator import TypeValidator
from kumo_relational_engine.pql.validator.utils import merge


FeatureAST = LogicalOperation | Join | Condition | Column | Aggregation


@dataclass
class ParsedFeature:
    r"""One model input feature, parsed out of its PQL expression.

    Args:
        expression: The fragment exactly as the caller wrote it. This is the
            feature's canonical name downstream, so it is kept verbatim
            rather than normalised.
        entity: Fully qualified name of the entity the feature is computed
            for, as ``table.column``.
        query: The synthetic ``PREDICT <expression> FOR EACH <entity>`` query
            the fragment was parsed inside of. Validation reuses it to reach
            the entity, which the query validators need and a bare fragment
            does not carry.
    """

    expression: str
    entity: str
    query: ParsedPredictiveQuery

    @property
    def ast(self) -> FeatureAST:
        r"""The AST node the fragment parsed to.

        Read through the host query rather than stored, because join
        inferral replaces the node with a :class:`Join` wrapping it.
        """
        return self.query.target_ast


def parse_feature(expression: str, entity: str) -> ParsedFeature:
    r"""Parses one feature expression into its AST node.

    The fragment is parsed by wrapping it as
    ``PREDICT <expression> FOR EACH <entity>`` and running the ordinary
    :class:`PQLParser` over that, so a feature and a target are accepted by
    exactly the same grammar. Every position in a diagnostic is walked back
    out of the wrapper and into the fragment, so a caller who never wrote
    ``PREDICT`` is never told about a column that only exists because of it.

    Args:
        expression: The feature expression, e.g.
            ``'SUM(orders.price, -90, 0, days)'``.
        entity: Fully qualified name of the entity, as ``table.column``.

    Returns:
        The parsed feature.

    Raises:
        ValueError: If the expression is not a valid PQL expression. The
            message locates the problem within `expression`.
    """
    query = f'{'PREDICT '}{expression}{' FOR EACH '}{entity}'
    parser = PQLParser(query_validation_type=QueryValidationType.TABULAR)
    _, _, errors = parser.parse_raw(query)
    if len(errors) > 0:
        raise ValueError(_parse_error(errors, expression).message())

    parsed_query = parser.to_parsed_predictive_query(query)
    _shift_locations(parsed_query.target_ast)
    # The entity is not part of the fragment, so no position within the
    # fragment describes it. Say so rather than report a shifted one.
    _drop_locations(parsed_query.entity_ast)
    return ParsedFeature(
        expression=expression,
        entity=entity,
        query=parsed_query,
    )


def validate_features(
    features: Sequence[ParsedFeature],
    graph: GraphDefinition,
) -> ValidationResponse:
    r"""Validates parsed features against the graph they are computed on.

    Each feature is checked for tables and columns that exist, types the
    aggregation accepts, a time window the table can answer, and a path from
    the entity table to the table the feature reads. Every one of those is
    the query validators' own rule, called here on the feature node alone.

    The full :class:`PredictiveQueryValidator` is deliberately not used: it
    validates its target with ``allow_past_looking=False``, and a feature is
    a backward window by definition, so every windowed feature would be
    rejected with "TARGET cannot have an aggregation in the past". The same
    validators are therefore invoked one at a time, with the flags a feature
    needs rather than the flags a label needs.

    Args:
        features: The parsed features. Each carries the entity it belongs to.
        graph: The graph the features are written against.

    Returns:
        Every error and warning found, across all features.
    """
    response = ValidationResponse()
    for feature in features:
        response = merge(response, _validate_feature(feature, graph))
    return response


# Helpers #####################################################################


def _validate_feature(
    feature: ParsedFeature,
    graph: GraphDefinition,
) -> ValidationResponse:
    r"""Runs the component validators over one feature, in the order
    :meth:`PredictiveQueryValidator.validate_predictive_query` runs them.

    Each stage is a precondition for the next -- type inferral reads columns
    that column validation proved exist, and join inferral reads the time
    column that time range validation writes -- so a failing stage stops the
    feature rather than letting the next one raise.
    """
    query = feature.query
    query_validator = PredictiveQueryValidator(
        graph,
        query_validation_type=QueryValidationType.TABULAR,
    )
    response = merge(
        query_validator.validate_columns(query),
        query_validator.validate_wildcard(query),
    )
    if not response.ok:
        return _name_the_feature(response, feature)

    # Node-level type validation, not `validate_dtypes`/`validate_stypes`:
    # those two add the rules that make a node a predictable *label*, and
    # reject a timestamp or a list outright. A feature is an input, so those
    # rules do not apply to it; what remains is the type checking of the
    # expression itself, which does.
    type_validator = TypeValidator(graph)
    type_validator.infer_dtypes(query)
    type_validator.infer_stypes(query)
    response = merge(response, type_validator._validate_dtypes(feature.ast))
    response = merge(response, type_validator._validate_stypes(feature.ast))
    if not response.ok:
        return _name_the_feature(response, feature)

    response = merge(
        response,
        TimeRangeValidator(graph).validate_tree(
            feature.ast,
            'FEATURE',
            ProblemType.CLASSIFY,
            # A feature is the entity's history: it may look back as far as
            # it likes, and may not look forward at all, because the data it
            # would need does not exist at prediction time.
            allow_future_looking=False,
            allow_past_looking=True,
        ),
    )
    if not response.ok:
        return _name_the_feature(response, feature)

    response = merge(
        response,
        JoinValidator(graph).infer_and_validate_joins(query),
    )
    return _name_the_feature(response, feature)


def _name_the_feature(
    response: ValidationResponse,
    feature: ParsedFeature,
) -> ValidationResponse:
    r"""Prefixes every diagnostic with the feature it is about.

    A validator writes for a query, where there is one target and naming it
    is redundant. A caller who declared a dozen features needs to know which
    one is at fault before the row and column mean anything.
    """
    named = ValidationResponse(info_items=response.info_items)
    named.errors = [
        ValidationError(
            title=error.title,
            message=_name_one(error.message, feature),
        )
        for error in response.errors
    ]
    named.warnings = [
        ValidationWarning(
            title=warning.title,
            message=_name_one(warning.message, feature),
        )
        for warning in response.warnings
    ]
    return named


def _name_one(message: str, feature: ParsedFeature) -> str:
    # A node with no location renders an empty `message_start`, which leaves
    # the message opening with the separator on its own.
    return f'Feature {feature.expression!r}: {message.removeprefix(": ")}'


def _parse_error(errors: list, expression: str) -> ValidationResponse:
    r"""Turns the syntax errors of the synthetic query into fragment errors.

    Anything ANTLR reports past the end of the fragment is a complaint about
    the ``FOR EACH`` the wrapper appended, which the caller did not write and
    cannot fix. It means the fragment ran out before the expression was
    finished, so it is reported as exactly that.
    """
    end_row, end_col = _fragment_end(expression)
    for error in errors:
        row, col = _in_fragment(error.line, error.column)
        if (row, col) >= (end_row, end_col):
            return ValidationResponse(
                errors=[
                    ValidationError(
                        title='Incomplete feature expression',
                        message=(
                            f'row {end_row}, column {end_col}: Feature '
                            f'{expression!r} ends before the expression is '
                            f'complete. Check for a missing closing '
                            f'parenthesis, or for an aggregation given fewer '
                            f'arguments than it takes.'
                        ),
                    )
                ]
            )
        error.line, error.column = row, col
        # The lineage records which clause of the *synthetic* query the
        # parser was in, and every fragment sits in the target clause. Left
        # alone, it turns every syntax error into "the target (PREDICT)
        # clause in this query is empty or invalid", which names a keyword
        # the caller never wrote. Cleared, the translator falls back to a
        # summary that holds for a fragment, and the located specifics --
        # the ones that say which token was wrong -- are untouched.
        error.lineage = []

    response = ErrorTranslator().translate_errors(errors, expression)
    return ValidationResponse(
        errors=[
            ValidationError(
                title='Invalid feature expression',
                message=f'Feature {expression!r}: {error.message}',
            )
            for error in response.errors
        ]
    )


def _in_fragment(row: int, col: int) -> tuple[int, int]:
    r"""Maps an ANTLR position in the synthetic query into the fragment.

    Rows are 1-indexed and columns 0-indexed, both following ANTLR. Only the
    first row of the fragment carries the ``PREDICT`` prefix, so only it
    shifts; a fragment written across several lines keeps the rest as they
    are.
    """
    if row == 1:
        return row, col - len('PREDICT ')
    return row, col


def _fragment_end(expression: str) -> tuple[int, int]:
    r"""The position one character past the end of `expression`."""
    rows = expression.split('\n')
    return len(rows), len(rows[-1])


def _shift_locations(node: ASTNode) -> None:
    r"""Rewrites a subtree's locations from the synthetic query's coordinates
    into the fragment's, so that a diagnostic a validator writes points at
    the fragment the caller handed over.
    """
    for child in node.children:
        _shift_locations(child)
    if node.location is None:
        return
    start_row, start_col = _in_fragment(
        node.location.start_row, node.location.start_col
    )
    end_row, end_col = _in_fragment(
        node.location.end_row, node.location.end_col
    )
    node.location = ASTQueryLocationInterval(
        start_row=start_row,
        start_col=start_col,
        end_row=end_row,
        end_col=end_col,
        data_available=node.location.data_available,
    )


def _drop_locations(node: ASTNode) -> None:
    r"""Marks a subtree as carrying no location."""
    for child in node.children:
        _drop_locations(child)
    node.location = ASTQueryLocationInterval(0, 0, 0, 0, False)

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_relational_engine.api.common import (
    ValidationError,
    ValidationResponse,
)
from kumo_relational_engine.api.graph import ColumnKey, GraphDefinition
from kumo_relational_engine.api.pquery import ParsedPredictiveQuery
from kumo_relational_engine.api.pquery.AST import (
    Aggregation,
    ASTNode,
    Column,
    Condition,
    Constant,
    Filter,
    Join,
)
from kumo_relational_engine.api.pquery.AST.location_interval import (
    ASTQueryLocationInterval,
)
from kumo_relational_engine.api.typing import (
    AggregationType,
    Dtype,
    MemberOp,
    ProblemType,
    RelOp,
    StrOp,
    Stype,
)
from kumo_relational_engine.pql.parser.parser import QueryValidationType
from kumo_relational_engine.pql.validator.utils import (
    col_name,
    merge,
    table_name,
)

FOR = 'FOR'
FOR_EACH = 'FOR EACH'
MAX_ENTITY_LIST_LENGTH = 1000
MAX_TOP_K = 20


class RfmValidator:
    r"""This class contains validation logic specific to RFM queries.

    Args:
        graph: The graph that query is written for.
        query_validation_type: The rfm validation level flag.
    """

    def __init__(
        self,
        graph: GraphDefinition,
        query_validation_type: QueryValidationType = QueryValidationType.ENTERPRISE,
    ):
        self.graph = graph
        self.query_validation_type = query_validation_type

    def validate(
        self, parsed_query: ParsedPredictiveQuery
    ) -> ValidationResponse:
        r"""Validates the structure of RFM queries.

        Args:
            parsed_query: the input parsed query.

        Returns:
            List of encountered errors/warnings.
        """
        response = ValidationResponse()
        if parsed_query.rfm_query:
            # TODO: deprecate the `rfm_query` field. It does not need to be
            # user-facing
            assert self.query_validation_type.is_rfm()
        if self.query_validation_type.is_enterprise():
            # Ensure that non-RFM queries aren't using RFM syntax
            if parsed_query.for_each == FOR:
                response.errors.append(
                    ValidationError(
                        title='Invalid query structure',
                        message=(
                            '"FOR" clause is only supported for Kumo Relational '
                            'foundation model, use FOR EACH when '
                            'training your own model.'
                        ),
                    )
                )
            if parsed_query.rfm_entity_ids is not None:
                response.errors.append(
                    ValidationError(
                        title='Invalid query structure',
                        message=(
                            'Specifying entities is only supported for '
                            'Kumo Relational foundation model, not when '
                            'training your own model.'
                        ),
                    )
                )
            return response

        assert self.query_validation_type.is_rfm()

        if self.query_validation_type.is_tfm():
            response = merge(
                response, self._find_unsupported_tfm_tasks(parsed_query)
            )
            if not response.ok:
                return response

        # Disable multilabel everything
        target_ast = parsed_query.target_ast
        if isinstance(target_ast, Join):
            # Handles static link prediction
            target_ast = target_ast.rhs_target

        if (
            isinstance(target_ast, Column)
            and parsed_query.problem_type == ProblemType.CLASSIFY
        ):
            response.errors.append(
                ValidationError(
                    title='Unsupported query structure',
                    message=(
                        'Foundation Model queries do not support '
                        'static multilabel classification. To make '
                        'predictions with this query, train a new model.'
                    ),
                )
            )

        if (
            isinstance(target_ast, Aggregation)
            and target_ast.aggr == AggregationType.LIST_DISTINCT
        ):
            aggregated_col_name = target_ast.get_target_column_name()
            _, target_is_fkey = self._is_key_col(aggregated_col_name)

            # Disable link prediction for RFM_SDK_V2
            if target_is_fkey and self.query_validation_type.is_sdk_v2():
                response.errors.append(
                    ValidationError(
                        title='Unsupported query structure',
                        message=(
                            'Link prediction queries are not supported by '
                            'this version of KumoRelational.'
                        ),
                    )
                )
                return response

            if parsed_query.problem_type == ProblemType.CLASSIFY or (
                parsed_query.problem_type == ProblemType.RANK
                and target_ast.stype == Stype.multicategorical
                and not target_is_fkey
            ):
                response.errors.append(
                    ValidationError(
                        title='Unsupported query structure',
                        message=(
                            'Foundation Model queries do not support '
                            'multilabel tasks. Use `LIST_DISTINCT` on a '
                            'foreign key column to make a link prediction '
                            'query. To make predictions with this query, '
                            'train a new model.'
                        ),
                    )
                )
            # Limit LP top K
            elif parsed_query.problem_type == ProblemType.RANK:
                top_k = parsed_query.top_k
                if top_k > MAX_TOP_K:
                    response.errors.append(
                        ValidationError(
                            title='Top k is too large',
                            message=(
                                f'Top k {top_k} exceeds maximum '
                                f'foundation model supported value of '
                                f'{MAX_TOP_K}. To list more candidates, '
                                f'train a new model.'
                            ),
                        )
                    )
                if target_ast.aggr_time_range is None:
                    response.errors.append(
                        ValidationError(
                            title='Static Link Prediction not supported',
                            message=(
                                'LIST_DISTINCT without a time range is '
                                'not supported by the foundation model.'
                                'To make predictions with this query, '
                                'train a new model.'
                            ),
                        )
                    )

        # Disable one-hop away filters/targets
        response = merge(
            response, self._find_disabled_joins(parsed_query.entity_ast)
        )
        response = merge(
            response, self._find_disabled_joins(parsed_query.target_ast)
        )
        if parsed_query.whatif_ast is not None:
            response = merge(
                response, self._find_disabled_joins(parsed_query.whatif_ast)
            )
        # Disable unsupported string operations
        response = merge(
            response, self._find_disabled_ops(parsed_query.entity_ast)
        )
        response = merge(
            response, self._find_disabled_ops(parsed_query.target_ast)
        )
        if parsed_query.whatif_ast is not None:
            response = merge(
                response, self._find_disabled_ops(parsed_query.whatif_ast)
            )
        # Disable unsupported aggregations
        response = merge(
            response,
            self._find_disabled_aggrs(
                parsed_query.entity_ast,
                permit_list_distinct=False,
                nested_aggr=False,
            ),
        )
        response = merge(
            response,
            self._find_disabled_aggrs(
                parsed_query.target_ast,
                permit_list_distinct=True,
                nested_aggr=False,
            ),
        )
        if parsed_query.whatif_ast is not None:
            response = merge(
                response,
                self._find_disabled_aggrs(
                    parsed_query.whatif_ast,
                    permit_list_distinct=False,
                    nested_aggr=False,
                ),
            )
        # Disable unsupported key references
        response = merge(
            response,
            self._find_key_cols(
                parsed_query.entity_ast,
                permit_list_distinct=False,
                permit_fkey=False,
                permit_pkey=True,
            ),
        )
        response = merge(
            response,
            self._find_key_cols(
                parsed_query.target_ast,
                permit_list_distinct=True,
                permit_fkey=False,
                permit_pkey=False,
            ),
        )
        if parsed_query.whatif_ast is not None:
            response = merge(
                response,
                self._find_key_cols(
                    parsed_query.whatif_ast,
                    permit_list_distinct=False,
                    permit_fkey=False,
                    permit_pkey=False,
                ),
            )

        # Check the correct use of FOR
        if parsed_query.for_each == FOR:
            if parsed_query.rfm_entity_ids is None:
                response.errors.append(
                    ValidationError(
                        title='Invalid query structure',
                        message=(
                            'When using "FOR" in foundation model '
                            'queries, you must specify one or multiple '
                            'entities to make the prediction for '
                            '(e.g., "FOR table.column = <ID>")'
                        ),
                    )
                )
                return response

            # Check that entity IDs are defined correctly
            assert isinstance(parsed_query.rfm_entity_ids, Condition)
            message = parsed_query.rfm_entity_ids.get_location().message_start
            op = parsed_query.rfm_entity_ids.op
            assert isinstance(parsed_query.rfm_entity_ids.value, Constant)
            entity_ids = parsed_query.rfm_entity_ids.value.typed_value()
            if op == RelOp.EQ:
                if isinstance(entity_ids, list):
                    response.errors.append(
                        ValidationError(
                            title='Invalid query structure',
                            message=(
                                f'{message}: Expected a single entity ID, '
                                f'but got a list: '
                                + str(entity_ids)
                                .replace('[', '(')
                                .replace(']', ')')
                            ),
                        )
                    )
                    return response
            elif op == MemberOp.IN:
                if not isinstance(entity_ids, list):
                    response.errors.append(
                        ValidationError(
                            title='Invalid query structure',
                            message=(
                                f'{message}: Expected a list of entity IDs, '
                                f'but got {entity_ids}.'
                            ),
                        )
                    )
                else:
                    if len(entity_ids) > MAX_ENTITY_LIST_LENGTH:
                        entity_ids_repr = str(
                            entity_ids[:MAX_ENTITY_LIST_LENGTH]
                        )
                        entity_ids_repr = entity_ids_repr[:-1] + ', ... ]'
                        response.errors.append(
                            ValidationError(
                                title='The entity id list is too long',
                                message=(
                                    f'{message}: List of entity IDs '
                                    f'{entity_ids_repr} has length '
                                    f'{len(entity_ids)} which exceeds '
                                    f'maximum allowed length of '
                                    f'{MAX_ENTITY_LIST_LENGTH}.'
                                ),
                            )
                        )
            else:
                response.errors.append(
                    ValidationError(
                        title='Invalid query structure',
                        message=(
                            f'{message}: Expected entity IDs, with operators '
                            f'IN or =, but got {op.value}. Valid syntax for '
                            f'specifying entity IDs is '
                            f'"FOR table.column = <ID>" or '
                            f'"FOR table.column IN (<ID>, <ID>, ...)".'
                        ),
                    )
                )
        else:
            if parsed_query.rfm_entity_ids is not None:
                message = (
                    parsed_query.rfm_entity_ids.get_location().message_start
                )
                response.errors.append(
                    ValidationError(
                        title='Invalid query structure',
                        message=(
                            f'{message}: Entity IDs are not expected '
                            f'when `FOR EACH` syntax is used. Got '
                            f'{parsed_query.rfm_entity_ids}.'
                        ),
                    )
                )
        return response

    def _find_unsupported_tfm_tasks(
        self, parsed_query: ParsedPredictiveQuery
    ) -> ValidationResponse:
        r"""Validates that the query names a task the tabular model supports.

        Args:
            parsed_query: the input parsed query.

        Returns:
            List of encountered errors/warnings.
        """
        response = ValidationResponse()

        target_ast = parsed_query.target_ast
        if isinstance(target_ast, Join):
            # Handles the implicit join a link prediction target carries.
            target_ast = target_ast.rhs_target
        is_list_distinct = self._has_list_distinct(target_ast)

        if (
            parsed_query.problem_type == ProblemType.RANK
            or parsed_query.top_k is not None
        ):
            node = parsed_query.target_ast
            reason = (
                'Tabular foundation model queries do not support ranking or '
                'link prediction. Drop the "RANK TOP k" clause and predict a '
                'single label per entity instead.'
            )
        elif (
            is_list_distinct
            or parsed_query.problem_type == ProblemType.CLASSIFY
            or target_ast.stype == Stype.multicategorical
        ):
            node = target_ast
            clause = 'LIST_DISTINCT' if is_list_distinct else 'CLASSIFY'
            reason = (
                f'Tabular foundation model queries do not support multilabel '
                f'tasks, so "{clause}" cannot be used here. Predict a single '
                f'categorical or numerical label per entity instead.'
            )
        elif parsed_query.problem_type == ProblemType.FORECAST:
            node = parsed_query.target_ast
            reason = (
                f'Tabular foundation model queries do not support '
                f'forecasting. Drop the "FORECAST '
                f'{parsed_query.num_forecasts} TIMEFRAMES" clause to predict '
                f'a single aggregate over one time range.'
            )
        elif parsed_query.whatif_ast is not None:
            node = parsed_query.whatif_ast
            reason = (
                'Tabular foundation model queries do not support '
                'counterfactuals. Drop the "ASSUMING" clause and predict '
                'against the observed data instead.'
            )
        else:
            return response

        response.errors.append(
            ValidationError(
                title='Unsupported query structure',
                message=f'{node.get_location().message_start}: {reason}',
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

    def _find_disabled_joins(self, node: ASTNode) -> ValidationResponse:
        response = ValidationResponse()
        if isinstance(node, Join):
            if isinstance(node.rhs_target, Column):
                response.errors.append(
                    ValidationError(
                        title='Unsupported implicit join',
                        message=(
                            f'{node.get_location().message_start}: '
                            f'Static references to columns from other '
                            f'tables that implicitly contain a '
                            f'foreign key to primary key connection '
                            f'are not supported in foundation model queries. '
                            f'Your query implicitly requires a join '
                            f'{node.lhs_key} -> {node.rhs_key}. '
                            f'Please remove the reference to table '
                            f'{table_name(node.rhs_key)} and retry, or '
                            f'train a new model instead of the '
                            f'foundation model.'
                        ),
                    )
                )
        for child in node.children:
            response = merge(response, self._find_disabled_joins(child))
        return response

    def _is_key_col(self, fqn: str) -> tuple[bool, bool]:
        r"""Returns (is_pkey, is_fkey) for the given column FQN."""
        col_key = ColumnKey(table_name(fqn), col_name(fqn))
        is_pkey = self.graph.tables[table_name(fqn)].pkey == col_name(fqn)
        all_keys = []
        for col_group in self.graph.col_groups:
            all_keys.extend(list(col_group.columns))
        is_fkey = col_key in all_keys and not is_pkey
        return is_pkey, is_fkey

    def _find_disabled_ops(self, node: ASTNode) -> ValidationResponse:
        response = ValidationResponse()
        unsupported_str_ops = (
            StrOp.STARTS_WITH,
            StrOp.ENDS_WITH,
            StrOp.CONTAINS,
            StrOp.NOT_CONTAINS,
            RelOp.LEQ,
            RelOp.GEQ,
            RelOp.LT,
            RelOp.GT,
        )
        if isinstance(node, Condition):
            assert isinstance(node.value, Constant)
            if (
                node.value.dtype_maybe == Dtype.string
                and node.op in unsupported_str_ops
            ):
                op_rep = (
                    node.op.value if node.input_op is None else node.input_op
                )
                response.errors.append(
                    ValidationError(
                        title='Unsupported operation',
                        message=(
                            f'{node.get_location().message_start}: '
                            f'Operation {op_rep} between strings is '
                            f'not supported in the foundation model queries.'
                        ),
                    )
                )
            if (
                node.target.stype == Stype.ID
                and any(self._is_key_col(node.target.fqn))
                and self.query_validation_type.is_demo()
            ):
                response.errors.append(
                    ValidationError(
                        title='Unsupported operation',
                        message=(
                            f'{node.get_location().message_start}: '
                            f'Operations on primary key and foreign key '
                            f'columns are not supported in the foundation '
                            f'model queries.'
                        ),
                    )
                )
            elif not (
                node.target.stype in [Stype.numerical, Stype.categorical]
                or (
                    node.target.stype == Stype.timestamp
                    and self.query_validation_type.is_sdk()
                )
                or node.target.stype == Stype.ID
            ):
                response.errors.append(
                    ValidationError(
                        title='Unsupported operation',
                        message=(
                            f'{node.get_location().message_start}: '
                            f'Operations on a '
                            f'{node.target.stype.value} column are '
                            f'not supported in the foundation model queries '
                            f'yet.'
                        ),
                    )
                )
        for child in node.children:
            response = merge(response, self._find_disabled_ops(child))
        return response

    def _find_disabled_aggrs(
        self,
        node: ASTNode,
        permit_list_distinct: bool = False,
        nested_aggr: bool = False,
    ) -> ValidationResponse:
        response = ValidationResponse()
        unsupported_aggrs = (
            AggregationType.FIRST,
            AggregationType.LAST,
            AggregationType.COUNT_DISTINCT,
        )
        if isinstance(node, Aggregation):
            if node.aggr in unsupported_aggrs:
                response.errors.append(
                    ValidationError(
                        title='Unsupported aggregation',
                        message=(
                            f'{node.get_location().message_start}: '
                            f'Aggregation {node.aggr.value} type is not '
                            f'supported in the foundation model queries.'
                        ),
                    )
                )
            if (
                not node.target.dtype.is_numerical()
                and node.aggr != AggregationType.LIST_DISTINCT
            ):
                response.errors.append(
                    ValidationError(
                        title='Unsupported aggregation',
                        message=(
                            f'{node.get_location().message_start}: '
                            f'Aggregations of {node.target.stype.value} '
                            f'columns are not '
                            f'supported in the foundation model queries.'
                        ),
                    )
                )
            if (
                node.aggr == AggregationType.LIST_DISTINCT
                and not permit_list_distinct
            ):
                response.errors.append(
                    ValidationError(
                        title='Unsupported aggregation',
                        message=(
                            f'{node.get_location().message_start}: '
                            f'Aggregation {node.aggr.value} is not '
                            f'supported in the foundation model queries '
                            f'except for foreign key targets.'
                        ),
                    )
                )
            if nested_aggr:
                response.errors.append(
                    ValidationError(
                        title='Unsupported nested aggregation',
                        message=(
                            f'{node.get_location().message_start}: '
                            f'Nesting aggregations is not '
                            f'supported in the foundation model queries.'
                        ),
                    )
                )

            # NOTE: Set nested_aggr=True after processing current aggregation
            # to prevent further aggregations from being nested inside this
            # one. This ensures we catch cases where aggregations are used
            # inside other aggregations, which is not supported by the
            # relational foundation model.
            nested_aggr = True
        if isinstance(node, Condition):
            permit_list_distinct = False
        for child in node.children:
            response = merge(
                response,
                self._find_disabled_aggrs(
                    child,
                    permit_list_distinct=permit_list_distinct,
                    nested_aggr=nested_aggr,
                ),
            )
        return response

    def _find_key_cols(
        self,
        node: ASTNode,
        permit_list_distinct: bool = False,
        permit_fkey: bool = False,
        permit_pkey: bool = False,
    ) -> ValidationResponse:
        response = ValidationResponse()
        # The only two situations where we permit key references:
        # Entity definition and within a target LIST_DISTINCT
        if (
            isinstance(node, Aggregation)
            and node.aggr == AggregationType.LIST_DISTINCT
            and permit_list_distinct
        ):
            permit_fkey = True
        if isinstance(node, Aggregation) and node.aggr == AggregationType.COUNT:
            permit_fkey = True
            permit_pkey = True
        if isinstance(node, Condition):
            permit_list_distinct = False
            # DEMO deployment uses GE which cannot serve keys
            # client deployments have no such limitation
            permit_fkey = not self.query_validation_type.is_demo()
            permit_pkey = not self.query_validation_type.is_demo()
        if isinstance(node, Column) and col_name(node.fqn) != '*':
            target_is_pkey, target_is_fkey = self._is_key_col(node.fqn)
            if (target_is_pkey and not permit_pkey) or (
                target_is_fkey and not permit_fkey
            ):
                response.errors.append(
                    ValidationError(
                        title='Unsupported column reference',
                        message=(
                            f'{node.get_location().message_start}: '
                            f'References to primary keys and foreign keys are '
                            f'not supported in the foundation model queries '
                            f'except for foreign key targets in COUNT and '
                            f'LIST_DISTINCT targets.'
                        ),
                    )
                )
        # Filter requires special arg care
        if isinstance(node, Filter):
            response = merge(
                response,
                self._find_key_cols(
                    node.condition,
                    permit_list_distinct=permit_list_distinct,
                    permit_fkey=False,
                    permit_pkey=False,
                ),
            )
            response = merge(
                response,
                self._find_key_cols(
                    node.target,
                    permit_list_distinct=permit_list_distinct,
                    permit_fkey=permit_fkey,
                    permit_pkey=permit_pkey,
                ),
            )
            return response
        for child in node.children:
            response = merge(
                response,
                self._find_key_cols(
                    child,
                    permit_list_distinct=permit_list_distinct,
                    permit_fkey=permit_fkey,
                    permit_pkey=permit_pkey,
                ),
            )
        return response

    def update_location_interval(
        self, parsed_query: ParsedPredictiveQuery
    ) -> None:
        r"""If `parsed_query` has `self.evaluate` or `self.explain` set to
        :obj:`True`, updates the interval values across the entire PQuery
        to adjust for the removed prefix.

        Args:
            parsed_query (ParsedPredictiveQuery): Query to adjust locations to.
        """
        offset = 0
        if parsed_query.explain:
            offset += len('EXPLAIN ')
        if parsed_query.evaluate:
            offset += len('EVALUATE ')
        if offset == 0:
            return
        self._update_location_interval(parsed_query.target_ast, offset)
        self._update_location_interval(parsed_query.entity_ast, offset)
        if parsed_query.rfm_entity_ids is not None:
            self._update_location_interval(parsed_query.rfm_entity_ids, offset)
        if parsed_query.whatif_ast is not None:
            self._update_location_interval(parsed_query.whatif_ast, offset)

    def _update_location_interval(self, node: ASTNode, offset: int) -> None:
        if node.location.data_available:
            if node.location.start_row == 1:
                node.location = ASTQueryLocationInterval(
                    start_row=node.location.start_row,
                    start_col=node.location.start_col + offset,
                    end_row=node.location.end_row,
                    end_col=node.location.end_col,
                )
            if node.location.end_row == 1:
                node.location = ASTQueryLocationInterval(
                    start_row=node.location.start_row,
                    start_col=node.location.start_col,
                    end_row=node.location.end_row,
                    end_col=node.location.end_col + offset,
                )
        for child in node.children:
            self._update_location_interval(child, offset)

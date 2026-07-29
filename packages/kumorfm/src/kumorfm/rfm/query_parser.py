# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Any

from kumorfm.api.graph import GraphDefinition
from kumorfm.api.pquery import ValidatedPredictiveQuery


def parse_query_locally(
    query: str,
    graph_definition: GraphDefinition,
) -> ValidatedPredictiveQuery:
    try:
        from kumorfm.pql.parser.parser import PQLParser, QueryValidationType
        from kumorfm.pql.validator import PredictiveQueryValidator
    except ImportError as exc:
        raise ValueError(
            "String predictive queries require the vendored parser runtime "
            "dependencies. Install the SDK with its runtime dependencies, "
            "or pass a ValidatedPredictiveQuery instead.") from exc

    query_validation_type = QueryValidationType.RFM_SDK
    try:
        parsed_query = PQLParser(
            query_validation_type=query_validation_type,
        ).to_parsed_predictive_query(query)
        validator = PredictiveQueryValidator(
            graph=graph_definition,
            query_validation_type=query_validation_type,
        )
        validated_query, response = validator.validate_predictive_query(
            parsed_query)
    except Exception as exc:
        raise ValueError(f"Failed to parse query '{query}'. {exc}") from None

    if validated_query is None or not response.ok:
        raise ValueError(
            f"Failed to parse query '{query}'. {_response_message(response)}")

    if len(response.warnings) > 0:
        msg = '\n'.join([
            f'{i+1}. {warning.title}: {warning.message}'
            for i, warning in enumerate(response.warnings)
        ])
        warnings.warn(f"Encountered the following warnings during "
                      f"parsing:\n{msg}")

    return validated_query


def _response_message(response: Any) -> str:
    message = getattr(response, 'message', None)
    if callable(message):
        return str(message())
    return str(response)

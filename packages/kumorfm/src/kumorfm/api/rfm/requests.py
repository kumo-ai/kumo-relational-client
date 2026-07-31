# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from pydantic.dataclasses import dataclass

from kumorfm.api.common import ValidationResponse
from kumorfm.api.graph import GraphDefinition
from kumorfm.api.model_plan import RunMode
from kumorfm.api.pquery import ValidatedPredictiveQuery
from kumorfm.api.rfm import Context, Explanation, PQueryDefinition
from kumorfm.api.rfm.inference import InferenceConfig


@dataclass
class RFMValidateQueryRequest:
    query: str
    graph_definition: GraphDefinition


@dataclass
class RFMValidateQueryResponse:
    query_definition: PQueryDefinition
    validation_response: ValidationResponse


@dataclass
class RFMParseQueryRequest:
    query: str
    graph_definition: GraphDefinition


@dataclass
class RFMParseQueryResponse:
    query: ValidatedPredictiveQuery
    validation_response: ValidationResponse


@dataclass
class RFMPredictRequest:
    context: Context
    run_mode: RunMode
    use_prediction_time: bool = False
    query: str = ""
    inference_config: InferenceConfig | None = None
    return_embeddings: bool = False


@dataclass
class RFMExplanationResponse:
    prediction: dict[str, Any]
    summary: str
    details: Explanation
    warning: Optional[str] = None


@dataclass
class RFMPredictResponse:
    prediction: dict[str, Any]


@dataclass
class RFMEvaluateRequest:
    context: Context
    run_mode: RunMode
    metrics: Optional[List[str]] = None
    use_prediction_time: bool = False
    inference_config: InferenceConfig | None = None

    def __post_init__(self) -> None:
        if self.metrics is not None and len(self.metrics) == 0:
            self.metrics = None


@dataclass
class RFMEvaluateResponse:
    metrics: Dict[str, Optional[float]]


@dataclass
class ListPromptsResponse:
    prompts: list[str]


@dataclass
class PromptContentResponse:
    name: str
    content: str

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, cast

from pydantic.dataclasses import dataclass
from typing_extensions import Self

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

    def to_protobuf(self) -> Any:
        import kumorfm.api.rfm.protos.request_pb2 as _request_pb2

        request_pb2 = cast(Any, _request_pb2)

        msg = request_pb2.PredictRequest()
        self.context.fill_protobuf_(msg.context)
        msg.run_mode = getattr(request_pb2.RunMode, self.run_mode.upper())
        msg.use_prediction_time = self.use_prediction_time
        msg.query = self.query
        if self.inference_config is not None:
            self.inference_config.fill_protobuf(
                msg.inference_config,
                request_pb2,
            )
        msg.return_embeddings = self.return_embeddings

        return msg

    def serialize(self) -> bytes:
        import kumorfm.api.rfm.protos.request_pb2 as _request_pb2

        request_pb2 = cast(Any, _request_pb2)

        msg = request_pb2.PredictRequest()
        self.context.fill_protobuf_(msg.context)
        msg.run_mode = getattr(request_pb2.RunMode, self.run_mode.upper())
        msg.use_prediction_time = self.use_prediction_time
        msg.query = self.query
        if self.inference_config is not None:
            self.inference_config.fill_protobuf(
                msg.inference_config,
                request_pb2,
            )
        msg.return_embeddings = self.return_embeddings

        return self.to_protobuf().SerializeToString()

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        import kumorfm.api.rfm.protos.request_pb2 as _request_pb2

        request_pb2 = cast(Any, _request_pb2)

        msg = request_pb2.PredictRequest()
        msg.ParseFromString(data)

        return cls(
            context=Context.from_protobuf(msg.context),
            run_mode=RunMode(request_pb2.RunMode.Name(msg.run_mode).lower()),
            use_prediction_time=bool(msg.use_prediction_time),
            query=str(msg.query),
            inference_config=InferenceConfig.from_protobuf(
                msg.inference_config, request_pb2)
            if msg.HasField('inference_config') else None,
            return_embeddings=bool(msg.return_embeddings),
        )


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

    def to_protobuf(self) -> Any:
        import kumorfm.api.rfm.protos.request_pb2 as _request_pb2

        request_pb2 = cast(Any, _request_pb2)

        msg = request_pb2.EvaluateRequest()
        self.context.fill_protobuf_(msg.context)
        msg.run_mode = getattr(request_pb2.RunMode, self.run_mode.upper())
        if self.metrics is not None:
            msg.metrics.extend(self.metrics)
        msg.use_prediction_time = self.use_prediction_time
        if self.inference_config is not None:
            self.inference_config.fill_protobuf(
                msg.inference_config,
                request_pb2,
            )

        return msg

    def serialize(self) -> bytes:
        return self.to_protobuf().SerializeToString()

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        import kumorfm.api.rfm.protos.request_pb2 as _request_pb2

        request_pb2 = cast(Any, _request_pb2)

        msg = request_pb2.EvaluateRequest()
        msg.ParseFromString(data)

        return cls(
            context=Context.from_protobuf(msg.context),
            run_mode=RunMode(request_pb2.RunMode.Name(msg.run_mode).lower()),
            metrics=list(msg.metrics),
            use_prediction_time=bool(msg.use_prediction_time),
            inference_config=InferenceConfig.from_protobuf(
                msg.inference_config, request_pb2)
            if msg.HasField('inference_config') else None,
        )


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

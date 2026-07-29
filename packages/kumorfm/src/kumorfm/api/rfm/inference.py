# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import Any, Literal, cast

from pydantic import conint
from pydantic.dataclasses import dataclass

from kumorfm.api.task import TaskType

TargetTransform = Literal['clip', 'power', 'quantile']
RegressionOutput = Literal['mean', 'median', 'quantiles']


@dataclass
class InferenceConfig:
    num_estimators: conint(gt=0, le=4) = 1  # type: ignore
    column_shuffle: bool = False
    category_shuffle: bool = False
    hop_shuffle: bool = False

    @classmethod
    def from_task_type(cls, task_type: TaskType) -> 'InferenceConfig':
        if task_type.is_classification:
            return ClassificationInferenceConfig()
        if task_type in {TaskType.REGRESSION, TaskType.FORECASTING}:
            return RegressionInferenceConfig()
        return InferenceConfig()

    @staticmethod
    def _target_transform_from_proto(value: str) -> TargetTransform | None:
        if value == 'UNSPECIFIED':
            return None
        return cast(TargetTransform, value.lower())

    def fill_protobuf(self, msg: Any, request_pb2: Any) -> None:
        msg.num_estimators = int(self.num_estimators)
        msg.column_shuffle = bool(self.column_shuffle)
        msg.category_shuffle = bool(self.category_shuffle)
        msg.hop_shuffle = bool(self.hop_shuffle)
        msg.kind = request_pb2.BASE

        if isinstance(self, ClassificationInferenceConfig):
            msg.class_shuffle = bool(self.class_shuffle)
            msg.kind = request_pb2.CLASSIFICATION

        if isinstance(self, RegressionInferenceConfig):
            msg.target_transforms.extend([
                getattr(
                    request_pb2.TargetTransform,
                    'UNSPECIFIED' if transform is None else transform.upper(),
                ) for transform in self.target_transforms
            ])
            msg.output_type = getattr(
                request_pb2.RegressionOutput,
                self.output_type.upper(),
            )
            msg.kind = request_pb2.REGRESSION

    @classmethod
    def from_protobuf(cls, msg: Any, request_pb2: Any) -> 'InferenceConfig':
        if len(msg.ListFields()) == 0:
            return InferenceConfig()

        num_estimators = int(msg.num_estimators)
        column_shuffle = bool(msg.column_shuffle)
        category_shuffle = bool(msg.category_shuffle)
        hop_shuffle = bool(msg.hop_shuffle)
        kind = int(msg.kind)
        if kind == request_pb2.REGRESSION:
            target_transforms = [
                cls._target_transform_from_proto(
                    request_pb2.TargetTransform.Name(transform))
                for transform in list(msg.target_transforms)
            ]
            output_type = cast(
                RegressionOutput,
                request_pb2.RegressionOutput.Name(msg.output_type).lower(),
            )
            return RegressionInferenceConfig(
                num_estimators=num_estimators,
                column_shuffle=column_shuffle,
                category_shuffle=category_shuffle,
                hop_shuffle=hop_shuffle,
                target_transforms=target_transforms,
                output_type=output_type,
            )
        if kind == request_pb2.CLASSIFICATION:
            return ClassificationInferenceConfig(
                num_estimators=num_estimators,
                column_shuffle=column_shuffle,
                category_shuffle=category_shuffle,
                hop_shuffle=hop_shuffle,
                class_shuffle=bool(msg.class_shuffle),
            )
        return InferenceConfig(
            num_estimators=num_estimators,
            column_shuffle=column_shuffle,
            category_shuffle=category_shuffle,
            hop_shuffle=hop_shuffle,
        )


@dataclass
class ClassificationInferenceConfig(InferenceConfig):
    class_shuffle: bool = False


@dataclass
class RegressionInferenceConfig(InferenceConfig):
    target_transforms: list[TargetTransform | None] = field(
        default_factory=lambda: ['quantile'])
    output_type: RegressionOutput = 'median'

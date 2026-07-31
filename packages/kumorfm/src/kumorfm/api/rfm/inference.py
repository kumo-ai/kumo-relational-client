# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import Literal

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


@dataclass
class ClassificationInferenceConfig(InferenceConfig):
    class_shuffle: bool = False


@dataclass
class RegressionInferenceConfig(InferenceConfig):
    target_transforms: list[TargetTransform | None] = field(
        default_factory=lambda: ['quantile'])
    output_type: RegressionOutput = 'median'

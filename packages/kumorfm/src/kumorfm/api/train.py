# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import List, Optional, Tuple

from kumorfm.api.common import StrEnum
from kumorfm.api.table import Column


class TrainingStage(StrEnum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
    PRED = 'pred'

    @staticmethod
    def list_stages() -> List['TrainingStage']:
        return [TrainingStage.TRAIN, TrainingStage.VAL, TrainingStage.TEST]

    def add_stage_prefix(self, name: str) -> str:
        return f'{self.value}_{name}'

    @classmethod
    def split_staged_name(cls, name: str) -> Tuple['TrainingStage', str]:
        stage, suffix = name.split('_', maxsplit=1)
        return cls(stage), suffix


@dataclass
class TrainingTableSpec:
    """Specifies the modifications made
    to a training table generated from a predictive query.

    Args:
        weight_col: Weight column used for loss computation.
        feats: Features of each training instance.
           Currently used while training model used for online serving.
        candidate_col_name: For link prediction tasks this is the list of
            valid targets. These targets are used while training model
            used for online serving. The acutual `TARGET` set has
            to be a subset of this.
    """
    weight_col: Optional[str] = None
    feats: Optional[List[Column]] = None
    candidate_col_name: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.weight_col is None and self.feats is None
                and self.candidate_col_name is None):
            raise ValueError("At least one of weight_col, "
                             "feats, or candidate_col_name must be provided")

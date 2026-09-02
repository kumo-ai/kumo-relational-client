# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Sampling defaults and limits shared by the relational and tabular models.

Both models sample examples for the same query against the same sampler, so a
limit that differed between them would make the same query mean different
things depending on which model was asked.
"""

from collections import defaultdict

from kumo_relational_engine.api.task import TaskType
from kumo_relational_engine.core.backend import DataBackend

RANDOM_SEED = 42

MAX_PRED_SIZE: dict[TaskType, int] = defaultdict(lambda: 1_000)
MAX_PRED_SIZE[TaskType.TEMPORAL_LINK_PREDICTION] = 200

MAX_TEST_SIZE: dict[TaskType, int] = defaultdict(lambda: 2_000)
MAX_TEST_SIZE[TaskType.TEMPORAL_LINK_PREDICTION] = 400

# Backends whose querying can be optimized in place, e.g. by creating missing
# indices. Requires write access to the backend.
OPTIMIZABLE_BACKENDS = frozenset({DataBackend.SQLITE, DataBackend.DUCKDB})

__all__ = [
    'MAX_PRED_SIZE',
    'MAX_TEST_SIZE',
    'OPTIMIZABLE_BACKENDS',
    'RANDOM_SEED',
]

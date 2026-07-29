# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from pydantic.dataclasses import dataclass

from kumorfm.api.task import TaskType
from kumorfm.api.typing import Stype


@dataclass
class Cohorts:
    table_name: str
    column_name: str
    hop: int
    stype: Optional[Stype]
    cohorts: List[str]
    populations: List[float]
    targets: Optional[Union[List[float], Dict[str, List[float]]]]


@dataclass
class ContextExample:
    entity_id: Union[int, str, float]
    seed_time: Optional[datetime]
    score: float
    label: Optional[Any] = None


@dataclass
class Cell:
    value: Optional[Any]
    score: float


@dataclass
class Node:
    cells: Dict[str, Cell]
    links: Dict[str, Set[int]]


@dataclass
class Subgraph:
    seed_id: int
    seed_table: str
    seed_time: Optional[datetime]
    tables: Dict[str, Dict[int, Node]]
    context_examples: Optional[List[ContextExample]] = None


@dataclass
class Explanation:
    task_type: TaskType
    cohorts: List[Cohorts]
    subgraphs: List[Subgraph]

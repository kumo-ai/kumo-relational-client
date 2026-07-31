# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass

from kumorfm.api.task import TaskType
from kumorfm.api.typing import StrEnum, Stype

REV_REL = '__###REV###__'


class EdgeLayout(StrEnum):
    COO = 'COO'
    CSC = 'CSC'
    REV = 'REV'


@dataclass(config={'arbitrary_types_allowed': True})
class Table:
    df: pd.DataFrame
    row: Optional[np.ndarray]
    batch: np.ndarray
    num_sampled_nodes: List[int]
    stype_dict: Dict[str, Stype]
    primary_key: Optional[str]

    @property
    def num_rows(self) -> int:
        return sum(self.num_sampled_nodes)


@dataclass(config={'arbitrary_types_allowed': True})
class Link:
    layout: EdgeLayout
    row: Optional[np.ndarray]
    col: Optional[np.ndarray]
    num_sampled_edges: List[int]

    def __post_init__(self) -> None:
        if self.layout == EdgeLayout.REV:  # Look up edges from reverse link:
            assert self.row is None and self.col is None

    @property
    def num_edges(self) -> int:
        return sum(self.num_sampled_edges)


@dataclass(config={'arbitrary_types_allowed': True})
class Subgraph:
    anchor_time: np.ndarray
    table_dict: Dict[str, Table]
    link_dict: Dict[Tuple[str, str, str], Link]

    @property
    def batch_size(self) -> int:
        return len(self.anchor_time)

    @property
    def num_hops(self) -> int:
        return max(
            [len(link.num_sampled_edges)
             for link in self.link_dict.values()] + [0])

    @staticmethod
    def rev_edge_type(edge_type: Tuple[str, str, str]) -> Tuple[str, str, str]:
        src, rel, dst = edge_type
        if rel.startswith(REV_REL):
            return (dst, rel[len(REV_REL):], src)
        return (dst, f'{REV_REL}{rel}', src)


@dataclass(config={'arbitrary_types_allowed': True})
class Context:
    task_type: TaskType
    entity_table_names: Tuple[str, ...]
    subgraph: Subgraph
    y_train: pd.Series
    y_test: Optional[pd.Series]
    task_table: Optional[Table] = None
    top_k: Optional[int] = None
    step_size: Optional[int] = None
    num_forecasts: int = 1

    def __post_init__(self) -> None:
        if len(self.entity_table_names) == 0:
            raise ValueError("'entity_table_names' needs to at least contain "
                             "one entity table name")

        if self.task_table is not None:
            assert self.task_table.row is None
            assert self.task_table.num_sampled_nodes == []
            assert self.task_table.primary_key is None

    @property
    def num_train(self) -> int:
        return len(self.y_train)

    @property
    def num_test(self) -> int:
        return self.subgraph.batch_size - self.num_train


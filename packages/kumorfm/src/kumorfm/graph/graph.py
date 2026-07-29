# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple, TypeAlias, Union

import kumorfm.api.graph as api

from kumorfm.mixin import CastMixin


@dataclass(frozen=True, eq=True)
class Edge(CastMixin, api.Edge):
    r"""An edge represents a relationship between two RFM tables."""

    src_table: str
    fkey: str
    dst_table: str

    def __iter__(self) -> Iterator[str]:
        return iter((self.src_table, self.fkey, self.dst_table))

    def __hash__(self) -> int:
        return hash((self.src_table, self.fkey, self.dst_table))

    @property
    def _fully_qualified_name(self) -> str:
        return f"{self.src_table}.{self.fkey}.{self.dst_table}"


EdgeLike: TypeAlias = Union[Edge, Dict[str, str], Tuple[str, str, str]]

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic.dataclasses import dataclass

from kumorfm.api.table import TableDefinition

TableName = str


@dataclass(frozen=True)
class ColumnKey:
    r"""Reference to a column within a PQuery/Graph definition."""

    table_name: TableName
    col_name: str


@dataclass(frozen=True)
class ColumnKeyGroup:
    r"""Group of column keys to be linked together in a graph."""

    columns: tuple[ColumnKey, ...]  # Always sorted and deduped, immutable


@dataclass(frozen=True, eq=True)
class Edge:
    r"""A representation of an edge between tables in Kumo."""

    src_table: str
    fkey: str
    dst_table: str


@dataclass(frozen=True)
class EdgeMatches:
    r"""Stats (either absolute values or percentages) about an edge relating to
    how many rows in the source have edges to the destination and vice versa.
    """

    src_in_dst: float
    dst_in_src: float


@dataclass(frozen=True)
class EdgeHealthStatistics:
    r"""Information about the health of an edge."""

    absolute_match_stats: EdgeMatches
    percent_match_stats: EdgeMatches
    total_num_edges: int


@dataclass
class EdgeHealthResponse:
    r"""Statistics about the health of each edge in a graph."""

    is_ready: bool
    statistics: dict[str, EdgeHealthStatistics]


@dataclass
class GraphDefinition:
    tables: dict[TableName, TableDefinition]
    col_groups: list[ColumnKeyGroup]


@dataclass
class GraphResource:
    # Graph resource identifier in the format of "graph-<16digit hex id>",
    # this is generated on the backend.
    id: str

    # Immutable metadata definition for a graph.
    graph: GraphDefinition

    # Optional human-friendly name alias for the graph. Graphs can be created
    # with or without name alias. However, once a graph resource sets the name
    # alias, it cannot be renamed. Name aliases must be unique, in other words,
    # 1:1 mapping is required between graph name aliases to GraphResource.id
    # Effectively, name_alias is an optional, secondary identifier specified by
    # the client.  This means for graph with name alias, we can operate the
    # graph resource by either id or alias, e.g.:
    #   GET /graphs/<GraphResource.id>
    #   GET /graphs/<name_alias>
    name_alias: str | None = None


# Method: Validate ============================================================


@dataclass
class GraphValidationRequest:
    # NOTE Response is of type common.ValidationResponse
    graph_definition: GraphDefinition

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from pydantic import Field
from pydantic.dataclasses import dataclass

from nemotron_relational.api.source_table import SourceTableType
from nemotron_relational.api.typing import (
    Dtype,
    Stype,
    compatible_field_validator,
)


class TimestampUnit(Enum):
    r"""A timestamp unit for a column in a Nemotron Relational table."""

    #: Specified in seconds
    SECOND = 's'

    #: Specified in milliseconds
    MILLISECOND = 'ms'

    #: Specified in microseconds
    MICROSECOND = 'us'

    #: Specified in nanoseconds
    NANOSECOND = 'ns'


@dataclass(frozen=True)
class Column:
    r"""A column in a Nemotron Relational table."""

    name: str
    stype: Stype
    dtype: Dtype

    # If the column represents a timestamp, the format that the timestmap
    # is represented in:
    timestamp_format: TimestampUnit | str | None = None


@dataclass
class TableDefinition:
    r"""A definition of a Nemotron Relational table."""

    # List of ALL columns selected from source table
    cols: list[Column]

    source_table: SourceTableType = Field(discriminator='data_source_type')

    # Name of the primary key column.
    pkey: str | None = None

    # Name of the time column, required to have stype=SemanticType.timestamp
    time_col: str | None = None

    # Name of the end time column:
    end_time_col: str | None = None

    @compatible_field_validator('time_col', 'end_time_col', 'pkey')
    def empty_str_to_none(cls, v: str | None) -> str | None:
        if v == '':
            return None
        return v


@dataclass
class TableResource:
    # Table resource identifier in the format of "table-<16digit hex id>",
    # this is generated on the backend.
    id: str

    # Immutable metadata definition for a table.
    table: TableDefinition

    # Optional human-friendly name alias for the table. Tables can be created
    # with or without name alias. However, once a table resource sets the name
    # alias, it cannot be renamed. Name aliases must be unique, in other words,
    # 1:1 mapping is required between graph name aliases to TableResource.id
    # Effectively, name_alias is an optional, secondary identifier specified by
    # the client.  This means for graph with name alias, we can operate the
    # graph resource by either id or alias, e.g.:
    #   GET /tables/<TableResource.id>
    #   GET /tables/<name_alias>
    name_alias: str | None = None


# Method: Infer Metadata ======================================================


@dataclass(frozen=True)
class ColumnMetadataRequest:
    r"""A request to infer metadata for a column in a Nemotron Relational table. This request
    can be incomplete in its stype, dtype, or timestamp format.
    """

    name: str
    stype: Stype | None = None
    dtype: Dtype | None = None

    # If the column represents a timestamp, the format that the timestmap
    # is represented in:
    timestamp_format: TimestampUnit | str | None = None


@dataclass
class TableMetadataRequest:
    r"""A request to infer Nemotron Relational table metadata."""

    cols: list[ColumnMetadataRequest]
    source_table: SourceTableType = Field(discriminator='data_source_type')
    pkey: str | None = None
    time_col: str | None = None
    end_time_col: str | None = None


@dataclass
class TableMetadataResponse:
    r"""A response containing metadata for a Nemotron Relational table."""

    cols: list[Column]
    source_table: SourceTableType = Field(discriminator='data_source_type')
    pkey: str | None = None
    time_col: str | None = None
    end_time_col: str | None = None


# Method: Validate ============================================================


@dataclass
class TableValidationRequest:
    # NOTE Response is of type common.ValidationResponse
    table_definition: TableDefinition

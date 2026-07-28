from enum import Enum
from typing import List, Optional, Union

from pydantic import Field
from pydantic.dataclasses import dataclass

from kumorfm.api.source_table import SourceTableType
from kumorfm.api.typing import Dtype, Stype, compatible_field_validator


class TimestampUnit(Enum):
    r"""A timestamp unit for a column in a Kumo table."""

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
    r"""A column in a Kumo table."""
    name: str
    stype: Stype
    dtype: Dtype

    # If the column represents a timestamp, the format that the timestmap
    # is represented in:
    timestamp_format: Optional[Union[TimestampUnit, str]] = None


@dataclass
class TableDefinition:
    r"""A definition of a Kumo table."""
    # List of ALL columns selected from source table
    cols: List[Column]

    source_table: SourceTableType = Field(discriminator='data_source_type')

    # Name of the primary key column.
    pkey: Optional[str] = None

    # Name of the time column, required to have stype=SemanticType.timestamp
    time_col: Optional[str] = None

    # Name of the end time column:
    end_time_col: Optional[str] = None

    @compatible_field_validator("time_col", "end_time_col", "pkey")
    def empty_str_to_none(cls, v: Optional[str]) -> Optional[str]:
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
    name_alias: Optional[str] = None


# Method: Infer Metadata ======================================================


@dataclass(frozen=True)
class ColumnMetadataRequest:
    r"""A request to infer metadata for a column in a Kumo table. This request
    can be incomplete in its stype, dtype, or timestamp format."""
    name: str
    stype: Optional[Stype] = None
    dtype: Optional[Dtype] = None

    # If the column represents a timestamp, the format that the timestmap
    # is represented in:
    timestamp_format: Optional[Union[TimestampUnit, str]] = None


@dataclass
class TableMetadataRequest:
    r"""A request to infer Kumo table metadata."""
    cols: List[ColumnMetadataRequest]
    source_table: SourceTableType = Field(discriminator='data_source_type')
    pkey: Optional[str] = None
    time_col: Optional[str] = None
    end_time_col: Optional[str] = None


@dataclass
class TableMetadataResponse:
    r"""A response containing metadata for a Kumo table."""
    cols: List[Column]
    source_table: SourceTableType = Field(discriminator='data_source_type')
    pkey: Optional[str] = None
    time_col: Optional[str] = None
    end_time_col: Optional[str] = None


# Method: Validate ============================================================


@dataclass
class TableValidationRequest:
    # NOTE Response is of type common.ValidationResponse
    table_definition: TableDefinition

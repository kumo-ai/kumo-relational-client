r"""A snapshot resource represents a Table or a Graph associated with a fixed
data version. This resource is associated with a long-running stateful
workflow execution, which may succeed or fail to produce the data. A client
can use the resource identifier of the snapshot to poll and wait."""

from dataclasses import field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic.dataclasses import dataclass

from kumorfm.api.common import JobStatus, StrEnum
from kumorfm.api.graph import TableName
from kumorfm.api.table import TableDefinition

# For Pydantic v2, we need to import the CoreSchema class from pydantic_core
# since we need to define the __get_pydantic_core_schema__ method for the IDs
# to work.
if TYPE_CHECKING:
    try:
        from pydantic import GetCoreSchemaHandler
        from pydantic_core import CoreSchema
    except ImportError:
        pass


@dataclass
class JobState:
    r"""The status of a Kumo job, associated with a stateful resource."""
    status: JobStatus = JobStatus.NOT_STARTED

    # Start and end times:
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    warnings: List[str] = field(default_factory=list)
    # Present iff status == FAILED:
    error: Optional[str] = None


# Table Snapshot ##############################################################


class TableSnapshotID(str):
    r"""An identifier for a table at a particular data version. The identifier
    consists of two components joined with the '@' symbol. The first component
    is the table ID, and the second component is the data version.

    A string identifier was chosen to be consistent with REST resource
    identifier semantics."""
    pass

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: type,
        handler: 'GetCoreSchemaHandler',
    ) -> 'CoreSchema':
        from pydantic_core import core_schema
        return core_schema.no_info_after_validator_function(
            cls.validate, core_schema.str_schema())

    @classmethod
    def validate(cls, value: str) -> "TableSnapshotID":
        return cls(value)


@dataclass
class ColumnStats:
    r"""A dataclass representing a column and its assoicated statistics. These
    statistics are typically generated after ingestion."""
    # TODO(manan, siyang): refine this dataclass definition. Currently just
    # a placeholder:
    column_name: str
    stats: Dict[str, Any]


class TableSnapshotStage(StrEnum):
    r"""The stage of this table snapshot, indicating the long-running, stateful
    computations that have been performed on it."""

    #: Table has been ingested.
    INGEST = 'INGEST'

    #: Minimal (basic) column stats have been computed on this table.
    MIN_COL_STATS = 'MIN_COL_STATS'

    #: All column stats have been computed on this table.
    FULL_COL_STATS = 'FULL_COL_STATS'


@dataclass
class TableSnapshotResource:
    r"""TableDataSnapshotResource represents a full data snapshot of either:
    A) Ingested Kumo Table from an external source table, examples:

    POST /tablesnapshots  body := TableDefinition()
    POST /tablesnapshots?refresh=true|false  body := TableDefinition()

    GET /tablesnapshots/{snapshot_id}
    GET /tablesnapshots/{<pquery_id>.graph.<tablename>@__LATEST__}
    GET /tablesnapshots/{<graph_fpid>.<tablename>@__LATEST__}

    or,
    B) Kumo generated/produced table, e.g., training table, pred table, etc,
    (detailed design is not finalized yet). For example:

    POST /predictive_queries/{pquery_id}/generate_train_table => train_table_id
    GET /tablesnapshots/{train_table_id}?data_url=true&data_url_expiration=1d
    """
    # A combination of the table fingerprint ID and the data version of the
    # snapshot:
    id: TableSnapshotID

    # Table metadata:
    table_definition: Optional[TableDefinition] = None

    # Ingested data statistics:
    column_stats: Optional[List[ColumnStats]] = None

    # Table snapshot stage:
    stages: Dict[TableSnapshotStage, JobState] = field(default_factory=dict)


# Graph Snapshot ##############################################################


class GraphSnapshotID(str):
    r"""An identifier for a graph at a particular data version. The identifier
    consists of two components joined with the '@' symbol. The first component
    is the graph ID, and the second component is the data version.

    A string identifier was chosen to be consistent with REST resource
    identifier semantics."""
    pass

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: type,
        handler: 'GetCoreSchemaHandler',
    ) -> 'CoreSchema':
        from pydantic_core import core_schema
        return core_schema.no_info_after_validator_function(
            cls.validate, core_schema.str_schema())

    @classmethod
    def validate(cls, value: str) -> "GraphSnapshotID":
        return cls(value)


class GraphSnapshotStage(StrEnum):
    r"""The stage of this graph snapshot, indicating the long-running, stateful
    computations that have been performed on it."""
    # TODO(manan, siyang): expose graph table stats and edge stats?
    INGEST = 'INGEST'
    MATERIALIZE = 'MATERIALIZE'


@dataclass
class GraphSnapshotResource:
    r"""GraphSnapshotResource represents a full data snapshot of a graph,
    composed of multiple table snapshots."""
    # A combination of the graph fingerprint ID and the data version of the
    # snapshot:
    id: GraphSnapshotID

    # The table IDs that participated in this snapshot:
    table_ids: Dict[TableName, TableSnapshotID]

    # Graph snapshot stage:
    stages: Dict[GraphSnapshotStage, JobState] = field(default_factory=dict)

    # The name/alias of the graph at the time the snapshot was taken:
    graph_name: Optional[str] = None

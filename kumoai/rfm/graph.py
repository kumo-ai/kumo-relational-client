from __future__ import annotations

import contextlib
import copy
import io
import warnings
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import import_module
from importlib.util import find_spec
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union

import pandas as pd
from kumoapi.graph import ColumnKey, ColumnKeyGroup, GraphDefinition
from kumoapi.table import TableDefinition
from kumoapi.typing import Stype
from typing_extensions import Self

from kumoai import in_jupyter_notebook, in_streamlit_notebook, in_tmux
from kumoai.graph.graph import Edge, EdgeLike
from kumoai.mixin import CastMixin
from kumoai.rfm.base import ColumnSpec, DataBackend, Table
from kumoai.rfm.base.utils import Timedelta
from kumoai.rfm.infer import infer_time_column
from kumoai.utils import display

if TYPE_CHECKING:
    import graphviz
    from adbc_driver_duckdb.dbapi import Connection as AdbcDuckDBConnection
    from adbc_driver_sqlite.dbapi import AdbcSqliteConnection
    from snowflake.connector import SnowflakeConnection


@dataclass
class SqliteConnectionConfig(CastMixin):
    uri: str | Path
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DuckDBConnectionConfig(CastMixin):
    uri: str | Path | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


class Graph:
    r"""A graph of :class:`Table` objects, akin to relationships between
    tables in a relational database.

    Creating a graph is the final step of data definition; after a
    :class:`Graph` is created, you can use it to initialize the
    Kumo Relational Foundation Model (:class:`KumoRFM`).

    .. code-block:: python

        >>> # doctest: +SKIP
        >>> import pandas as pd
        >>> import kumoai.rfm as rfm

        >>> # Load data frames into memory:
        >>> df1 = pd.DataFrame(...)
        >>> df2 = pd.DataFrame(...)
        >>> df3 = pd.DataFrame(...)

        >>> # Define tables from data frames:
        >>> table1 = rfm.LocalTable(name="table1", data=df1)
        >>> table2 = rfm.LocalTable(name="table2", data=df2)
        >>> table3 = rfm.LocalTable(name="table3", data=df3)

        >>> # Create a graph from a dictionary of tables:
        >>> graph = rfm.Graph({
        ...     "table1": table1,
        ...     "table2": table2,
        ...     "table3": table3,
        ... })

        >>> # Infer table metadata:
        >>> graph.infer_metadata()

        >>> # Infer links/edges:
        >>> graph.infer_links()

        >>> # Inspect table metadata:
        >>> for table in graph.tables.values():
        ...     table.print_metadata()

        >>> # Visualize graph:
        >>> graph.visualize()

        >>> # Add/Remove edges between tables:
        >>> graph.link(src_table="table1", fkey="id1", dst_table="table2")
        >>> graph.unlink(src_table="table1", fkey="id1", dst_table="table2")

        >>> # Validate graph:
        >>> graph.validate()
    """

    # Constructors ############################################################

    def __init__(
        self,
        tables: Sequence[Table],
        edges: Sequence[EdgeLike] | None = None,
    ) -> None:

        self._tables: dict[str, Table] = {}
        self._edges: list[Edge] = []
        self._connection: (AdbcSqliteConnection | AdbcDuckDBConnection
                           | SnowflakeConnection | None) = None

        for table in tables:
            self.add_table(table)

        for table in tables:  # Use links from source metadata:
            if not any(column.is_source for column in table.columns):
                continue
            for fkey in table._source_foreign_key_dict.values():
                if fkey.name not in table:
                    continue
                if not table[fkey.name].is_source:
                    continue
                dst_table_names = [
                    table.name for table in self.tables.values()
                    if table.source_name == fkey.dst_table
                ]
                if len(dst_table_names) != 1:
                    continue
                dst_table = self[dst_table_names[0]]
                if dst_table._primary_key != fkey.primary_key:
                    continue
                if not dst_table[fkey.primary_key].is_source:
                    continue
                self.link(table.name, fkey.name, dst_table.name)

        for edge in (edges or []):
            _edge = Edge._cast(edge)
            assert _edge is not None
            if _edge not in self._edges:
                self.link(*_edge)

    @classmethod
    def from_data(
        cls,
        df_dict: dict[str, pd.DataFrame],
        edges: Sequence[EdgeLike] | None = None,
        infer_metadata: bool = True,
        verbose: bool = True,
    ) -> Self:
        r"""Creates a :class:`Graph` from a dictionary of
        :class:`pandas.DataFrame` objects.

        Automatically infers table metadata and links by default.

        .. code-block:: python

            >>> # doctest: +SKIP
            >>> import pandas as pd
            >>> import kumoai.rfm as rfm

            >>> # Load data frames into memory:
            >>> df1 = pd.DataFrame(...)
            >>> df2 = pd.DataFrame(...)
            >>> df3 = pd.DataFrame(...)

            >>> # Create a graph from a dictionary of data frames:
            >>> graph = rfm.Graph.from_data({
            ...     "table1": df1,
            ...     "table2": df2,
            ...     "table3": df3,
            ... })

        Args:
            df_dict: A dictionary of data frames, where the keys are the names
                of the tables and the values hold table data.
            edges: An optional list of :class:`~kumoai.graph.Edge` objects to
                add to the graph. If not provided, edges will be automatically
                inferred from the data in case ``infer_metadata=True``.
            infer_metadata: Whether to infer metadata for all tables in the
                graph.
            verbose: Whether to print verbose output.
        """
        from kumoai.rfm.backend.local import LocalTable

        graph = cls(
            tables=[LocalTable(df, name) for name, df in df_dict.items()],
            edges=edges or [],
        )

        if infer_metadata:
            graph.infer_metadata(verbose=False)

            if edges is None:
                graph.infer_links(verbose=False)

        if verbose:
            graph.print_metadata()
            graph.print_links()

        return graph

    @classmethod
    def from_sqlite(
        cls,
        connection: Union[
            'AdbcSqliteConnection',
            SqliteConnectionConfig,
            str,
            Path,
            dict[str, Any],
        ],
        tables: Sequence[str | dict[str, Any]] | None = None,
        edges: Sequence[EdgeLike] | None = None,
        infer_metadata: bool = True,
        verbose: bool = True,
    ) -> Self:
        r"""Creates a :class:`Graph` from a :class:`sqlite` database.

        Automatically infers table metadata and links by default.

        .. code-block:: python

            >>> # doctest: +SKIP
            >>> import kumoai.rfm as rfm

            >>> # Create a graph from all tables in a SQLite database:
            >>> graph = rfm.Graph.from_sqlite('data.db')

            >>> # Select specific tables and override table metadata:
            >>> graph = rfm.Graph.from_sqlite('data.db', tables=[
            ...     'USERS',
            ...     {'name': 'ORDERS', 'source_name': 'ORDERS_SNAPSHOT'},
            ...     {'name': 'ITEMS', 'primary_key': 'ITEM_ID'},
            ... ])

            >>> # Provide known edges while still inferring table metadata:
            >>> graph = rfm.Graph.from_sqlite(
            ...     'data.db',
            ...     tables=[
            ...         {'name': 'USERS', 'primary_key': 'USER_ID'},
            ...         {'name': 'ORDERS', 'primary_key': 'ORDER_ID'},
            ...         {'name': 'ITEMS', 'primary_key': 'ITEM_ID'},
            ...     ],
            ...     edges=[
            ...         ('ORDERS', 'USER_ID', 'USERS'),
            ...         ('ORDERS', 'ITEM_ID', 'ITEMS'),
            ...     ],
            ... )

        Args:
            connection: An open connection from
                :meth:`~kumoai.rfm.backend.sqlite.connect` or the
                path to the database file.
            tables: Set of table names or :class:`SQLiteTable` keyword
                arguments to include, such as ``name``, ``source_name``,
                ``primary_key``, or ``time_column``. If ``None``, will add all
                tables present in the database.
            edges: Optional edge-like objects to add to the graph. Each item
                may be a :class:`~kumoai.graph.Edge`, a dictionary, or a tuple
                such as ``(src_table, fkey, dst_table)``. If not provided,
                edges will be automatically inferred from the data in case
                ``infer_metadata=True``.
            infer_metadata: Whether to infer missing metadata for all tables in
                the graph.
            verbose: Whether to print verbose output.
        """
        from kumoai.rfm.backend.sqlite import Connection, SQLiteTable, connect

        internal_connection = False
        if not isinstance(connection, Connection):
            config = SqliteConnectionConfig._cast(connection)
            assert isinstance(config, SqliteConnectionConfig)
            connection = connect(config.uri, **config.kwargs)
            internal_connection = True
        assert isinstance(connection, Connection)

        if tables is None:
            with connection.cursor() as cursor:
                cursor.execute("SELECT name FROM sqlite_master "
                               "WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]

        table_kwargs: list[dict[str, Any]] = []
        for table in tables:
            kwargs = dict(name=table) if isinstance(table, str) else table
            table_kwargs.append(kwargs)

        graph = cls(
            tables=[
                SQLiteTable(connection=connection, **kwargs)
                for kwargs in table_kwargs
            ],
            edges=edges or [],
        )

        if internal_connection:
            graph._connection = connection

        if infer_metadata:
            graph.infer_metadata(verbose=False)

            if edges is None:
                graph.infer_links(verbose=False)

        if verbose:
            graph.print_metadata()
            graph.print_links()

        return graph

    @classmethod
    def from_duckdb(
        cls,
        connection: Union[
            'AdbcDuckDBConnection',
            DuckDBConnectionConfig,
            str,
            Path,
            dict[str, Any],
            None,
        ] = None,
        tables: Sequence[str | dict[str, Any]] | None = None,
        edges: Sequence[EdgeLike] | None = None,
        infer_metadata: bool = True,
        verbose: bool = True,
    ) -> Self:
        r"""Creates a :class:`Graph` from a :class:`duckdb` database.

        Automatically infers table metadata and links by default.

        Args:
            connection: An open connection from
                :meth:`~kumoai.rfm.backend.duckdb.connect`, the path to the
                database file, a connection config, or ``None`` for an
                in-memory database.
            tables: Set of table names or :class:`DuckDBTable` keyword
                arguments to include. If ``None``, will add all non-temporary
                tables present in the database.
            edges: An optional list of :class:`~kumoai.graph.Edge` objects to
                add to the graph. If not provided, edges will be automatically
                inferred from the data in case ``infer_metadata=True``.
            infer_metadata: Whether to infer missing metadata for all tables in
                the graph.
            verbose: Whether to print verbose output.
        """
        from kumoai.rfm.backend.duckdb import Connection, DuckDBTable, connect

        internal_connection = False
        if not isinstance(connection, Connection):
            if connection is None:
                config = DuckDBConnectionConfig()
            else:
                config = DuckDBConnectionConfig._cast(connection)
            assert isinstance(config, DuckDBConnectionConfig)
            connection = connect(config.uri, **config.kwargs)
            internal_connection = True
        assert isinstance(connection, Connection)

        if tables is None:
            with connection.cursor() as cursor:
                cursor.execute("SELECT table_name FROM duckdb_tables() "
                               "WHERE NOT temporary AND NOT internal")
                tables = [row[0] for row in cursor.fetchall()]

        table_kwargs: list[dict[str, Any]] = []
        for table in tables:
            kwargs = dict(name=table) if isinstance(table, str) else table
            table_kwargs.append(kwargs)

        graph = cls(
            tables=[
                DuckDBTable(connection=connection, **kwargs)
                for kwargs in table_kwargs
            ],
            edges=edges or [],
        )

        if internal_connection:
            graph._connection = connection

        if infer_metadata:
            graph.infer_metadata(verbose=False)

            if edges is None:
                graph.infer_links(verbose=False)

        if verbose:
            graph.print_metadata()
            graph.print_links()

        return graph

    @classmethod
    def from_snowflake(
        cls,
        connection: Union['SnowflakeConnection', dict[str, Any], None] = None,
        tables: Sequence[str | dict[str, Any]] | None = None,
        database: str | None = None,
        schema: str | None = None,
        edges: Sequence[EdgeLike] | None = None,
        infer_metadata: bool = True,
        verbose: bool = True,
    ) -> Self:
        r"""Creates a :class:`Graph` from a :class:`snowflake` database and
        schema.

        Automatically infers table metadata and links by default.

        .. code-block:: python

            >>> # doctest: +SKIP
            >>> import kumoai.rfm as rfm

            >>> # Create a graph directly in a Snowflake notebook:
            >>> graph = rfm.Graph.from_snowflake(
            ...     database='MY_DB',
            ...     schema='MY_SCHEMA',
            ... )

            >>> # Create a graph using explicit Snowflake credentials:
            >>> graph = rfm.Graph.from_snowflake(
            ...     connection={
            ...         'user': '<snowflake_user>',
            ...         'password': '<snowflake_password>',
            ...         'account': '<snowflake_account>',
            ...         'warehouse': '<snowflake_warehouse>',
            ...         'database': 'MY_DB',
            ...         'schema': 'MY_SCHEMA',
            ...     },
            ...     database='MY_DB',
            ...     schema='MY_SCHEMA',
            ...     tables=[
            ...         'USERS',
            ...         'ORDERS',
            ...         {'name': 'ITEMS', 'source_name': 'ITEMS_SNAPSHOT'},
            ...     ],
            ... )

            >>> # Fine-grained control over table specification:
            >>> graph = rfm.Graph.from_snowflake(tables=[
            ...     'USERS',
            ...     dict(name='ORDERS', source_name='ORDERS_SNAPSHOT'),
            ...     dict(name='ITEMS', schema='OTHER_SCHEMA'),
            ... ], database='DEFAULT_DB', schema='DEFAULT_SCHEMA')

        Args:
            connection: An open connection from
                :meth:`~kumoai.rfm.backend.snow.connect` or the
                :class:`snowflake` connector keyword arguments to open a new
                connection. If ``None``, will re-use an active session in case
                it exists, or create a new connection from credentials stored
                in environment variables.
            tables: Set of table names or :class:`SnowTable` keyword arguments
                to include. If ``None``, will add all tables present in the
                current database and schema.
            database: The database.
            schema: The schema.
            edges: An optional list of :class:`~kumoai.graph.Edge` objects to
                add to the graph. If not provided, edges will be automatically
                inferred from the data in case ``infer_metadata=True``.
            infer_metadata: Whether to infer metadata for all tables in the
                graph.
            verbose: Whether to print verbose output.
        """
        from kumoai.rfm.backend.snow import Connection, SnowTable, connect

        if not isinstance(connection, Connection):
            connection = connect(**(connection or {}))
        assert isinstance(connection, Connection)

        if database is None or schema is None:
            with connection.cursor() as cursor:
                cursor.execute("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()")
                result = cursor.fetchone()
                assert result is not None
                database = database or result[0]
                assert database is not None
                schema = schema or result[1]

        if tables is None:
            if schema is None:
                raise ValueError("No current 'schema' set. Please specify the "
                                 "Snowflake schema manually")

            with connection.cursor() as cursor:
                cursor.execute(f"""
                SELECT TABLE_NAME
                FROM {database}.INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = '{schema}'
                """)
                tables = [row[0] for row in cursor.fetchall()]

        table_kwargs: list[dict[str, Any]] = []
        for table in tables:
            if isinstance(table, str):
                kwargs = dict(name=table, database=database, schema=schema)
            else:
                kwargs = copy.copy(table)
                kwargs.setdefault('database', database)
                kwargs.setdefault('schema', schema)
            table_kwargs.append(kwargs)

        graph = cls(
            tables=[
                SnowTable(connection=connection, **kwargs)
                for kwargs in table_kwargs
            ],
            edges=edges or [],
        )

        if infer_metadata:
            graph.infer_metadata(verbose=False)

            if edges is None:
                graph.infer_links(verbose=False)

        if verbose:
            graph.print_metadata()
            graph.print_links()

        return graph

    @classmethod
    def from_snowflake_semantic_view(
        cls,
        semantic_view_name: str,
        connection: Union['SnowflakeConnection', dict[str, Any], None] = None,
        verbose: bool = True,
    ) -> Self:
        import yaml

        from kumoai.rfm.backend.snow import Connection, SnowTable, connect

        if not isinstance(connection, Connection):
            connection = connect(**(connection or {}))
        assert isinstance(connection, Connection)

        with connection.cursor() as cursor:
            sql = (f"SELECT SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW("
                   f"'{semantic_view_name}')")
            cursor.execute(sql)
            result = cursor.fetchone()
            assert result is not None
            cfg = yaml.safe_load(result[0])

        graph = cls(tables=[])

        msgs = []
        table_names = {table_cfg['name'] for table_cfg in cfg['tables']}
        for table_cfg in cfg['tables']:
            table_name = table_cfg['name']
            source_table_name = table_cfg['base_table']['table']
            database = table_cfg['base_table']['database']
            schema = table_cfg['base_table']['schema']

            primary_key: str | None = None
            if 'primary_key' in table_cfg:
                primary_key_cfg = table_cfg['primary_key']
                if len(primary_key_cfg['columns']) == 1:
                    primary_key = primary_key_cfg['columns'][0]
                elif len(primary_key_cfg['columns']) > 1:
                    msgs.append(f"Failed to add primary key for table "
                                f"'{table_name}' since composite primary keys "
                                f"are not yet supported")

            columns: list[ColumnSpec] = []
            unsupported_columns: list[str] = []
            for column_cfg in chain(
                    table_cfg.get('dimensions', []),
                    table_cfg.get('time_dimensions', []),
                    table_cfg.get('facts', []),
            ):
                column_name = column_cfg['name']
                column_expr = column_cfg.get('expr', None)
                column_data_type = column_cfg.get('data_type', None)

                if column_expr is None:
                    columns.append(ColumnSpec(name=column_name))
                    continue

                column_expr = column_expr.replace(f'{table_name}.', '')

                if column_expr == column_name:
                    columns.append(ColumnSpec(name=column_name))
                    continue

                # Drop expressions that reference other tables (for now):
                if any(f'{name.upper()}.' in column_expr.upper()
                       for name in table_names):
                    unsupported_columns.append(column_name)
                    continue

                column = ColumnSpec(
                    name=column_name,
                    expr=column_expr,
                    dtype=SnowTable._to_dtype(column_data_type),
                )
                columns.append(column)

            if len(unsupported_columns) == 1:
                msgs.append(f"Failed to add column '{unsupported_columns[0]}' "
                            f"of table '{table_name}' since its expression "
                            f"references other tables")
            elif len(unsupported_columns) > 1:
                msgs.append(f"Failed to add columns '{unsupported_columns}' "
                            f"of table '{table_name}' since their expressions "
                            f"reference other tables")

            table = SnowTable(
                connection,
                name=table_name,
                source_name=source_table_name,
                database=database,
                schema=schema,
                columns=columns,
                primary_key=primary_key,
            )

            # TODO Add a way to register time columns without heuristic usage.
            time_candidates = [  # Prioritize columns in `time_dimensions`:
                column_cfg['name']
                for column_cfg in table_cfg.get('time_dimensions', [])
                if table.has_column(column_cfg['name'])
                and table[column_cfg['name']].stype == Stype.timestamp
            ]
            if len(time_candidates) == 0:
                time_candidates = [
                    column.name for column in table.columns
                    if column.stype == Stype.timestamp
                ]
            if len(time_candidates) > 0:
                if time_column := infer_time_column(
                        df=table._get_sample_df(),
                        candidates=time_candidates,
                ):
                    table.time_column = time_column

            graph.add_table(table)

        for relation_cfg in cfg.get('relationships', []):
            name = relation_cfg['name']
            if len(relation_cfg['relationship_columns']) != 1:
                msgs.append(f"Failed to add relationship '{name}' since "
                            f"composite key references are not yet supported")
                continue

            left_table = relation_cfg['left_table']
            left_key = relation_cfg['relationship_columns'][0]['left_column']
            right_table = relation_cfg['right_table']
            right_key = relation_cfg['relationship_columns'][0]['right_column']

            if graph[right_table]._primary_key != right_key:
                # Semantic view error - this should never be triggered:
                msgs.append(f"Failed to add relationship '{name}' since the "
                            f"referenced key '{right_key}' of table "
                            f"'{right_table}' is not a primary key")
                continue

            if graph[left_table]._primary_key == left_key:
                msgs.append(f"Failed to add relationship '{name}' since the "
                            f"referencing key '{left_key}' of table "
                            f"'{left_table}' is a primary key")
                continue

            if left_key not in graph[left_table]:
                graph[left_table].add_column(left_key)

            graph.link(left_table, left_key, right_table)

        graph.validate()

        if verbose:
            graph.print_metadata()
            graph.print_links()

        if len(msgs) > 0:
            title = (f"Could not fully convert the semantic view definition "
                     f"'{semantic_view_name}' into a graph:\n")
            warnings.warn(title + '\n'.join(f'- {msg}' for msg in msgs))

        return graph

    @classmethod
    def from_relbench(
        cls,
        dataset: str,
        verbose: bool = True,
    ) -> Graph:
        r"""Loads a `RelBench <https://relbench.stanford.edu>`_ dataset into a
        :class:`Graph` instance.

        .. code-block:: python

            >>> # doctest: +SKIP
            >>> import kumoai.rfm as rfm

            >>> graph = rfm.Graph.from_relbench("f1")

        Args:
            dataset: The RelBench dataset name.
            verbose: Whether to print verbose output.
        """
        from kumoai.rfm.relbench import from_relbench
        graph = from_relbench(dataset, verbose=verbose)

        if verbose:
            graph.print_metadata()
            graph.print_links()

        return graph

    @classmethod
    def graph_and_pquery_from_timeseries(
        cls,
        df: pd.DataFrame,
        timeseries_col: str,
        timestamps_col: str | None = None,
        time_delta: pd.Timedelta | None = None,
        anchor_time: pd.Timestamp | None = None,
        entity_col: str | None = None,
        num_timeframes: int = 1,
    ) -> tuple[Self, str]:
        r"""Creates a :class:`Graph` and predictive query string from a
        time-series dataset stored as a single flat table.

        Many forecasting datasets arrive as a single table where each row
        represents one entity and a column holds an array of historical
        observations.  This method converts such a table into the two-table
        *(entity + target)* structure expected by KumoRFM and returns a
        ready-to-use predictive query.

        The input ``df`` is split into:

        * An entity table - one row per input row, containing all columns
          except ``timeseries_col`` (and ``timestamps_col`` when given).
        * A target table - one row per observation, with a foreign-key to
          the entity table, a timestamp column, and the observed value.

        .. code-block:: python

            >>> # doctest: +SKIP
            >>> import pandas as pd
            >>> import kumoai.rfm as rfm

            >>> df = pd.DataFrame({
            ...     'customer_id': [1, 2, 3],
            ...     'sales': [[10, 20, 15, 30], [5, 8, 12], [100, 95, 80]],
            ... })

            >>> anchor = pd.Timestamp('2024-01-10')
            >>> graph, pquery = rfm.Graph.graph_and_pquery_from_timeseries(
            ...     df,
            ...     timeseries_col='sales',
            ...     entity_col='customer_id',
            ...     time_delta=pd.Timedelta('1D'),
            ...     anchor_time=anchor,
            ...     num_timeframes=4,
            ... )
            >>> # pquery == ("PREDICT MAX(target.value, 0, 1, days) "
            >>> #            "FORECAST 4 TIMEFRAMES "
            >>> #            "FOR EACH entity.customer_id")

            >>> model = rfm.KumoRFM(graph)
            >>> result = model.predict(pquery, anchor_time=anchor)

        Args:
            df: Input DataFrame where each row represents one entity and
                ``timeseries_col`` holds a list or array of scalar
                observations for that entity.
            timeseries_col: Name of the column containing per-entity
                observation arrays.
            timestamps_col: Optional name of the column containing per-entity
                timestamp arrays (one timestamp per observation value).  Must
                be the same length as ``timeseries_col`` for every row.  When
                ``None``, synthetic timestamps are generated backwards from
                ``anchor_time`` using ``time_delta``.
            time_delta: Step size between consecutive observations.  Required
                when ``timestamps_col`` is ``None``.  Also controls the
                prediction-window size in the generated pquery.  When
                ``timestamps_col`` is provided and ``time_delta`` is ``None``,
                the typical step is inferred from the median inter-observation
                interval across all entities.
            anchor_time: The forecast cutoff timestamp.  Required when
                ``timestamps_col`` is ``None``; used to place synthetic
                timestamps so that the last observation falls at
                ``anchor_time - time_delta``.  Pass the same value to
                :meth:`KumoRFM.predict` as ``anchor_time``.
            entity_col: Name of an existing column to use as the entity
                primary key.  When ``None``, integer IDs ``[0, 1, ..., n-1]``
                are generated and stored in a new column named ``entity_id``.
            num_timeframes: Number of timeframes to forecast.

        Returns:
            A tuple ``(graph, pquery)`` where *graph* is a :class:`Graph`
            ready for use with :class:`KumoRFM` and *pquery* is a
            predictive-query string for single-step forecasting.  Pass
            *pquery* to :meth:`KumoRFM.predict` together with ``anchor_time``
            to generate forecasts.
        """
        from kumoai.rfm.backend.local import LocalTable

        target_table_name = "target"
        entity_table_name = "entity"
        value_col_name = "value"
        timestamp_col_name = "timestamp"

        if timeseries_col not in df.columns:
            raise ValueError(
                f"timeseries_col '{timeseries_col}' not found in the "
                f"DataFrame")

        if timestamps_col is not None and timestamps_col not in df.columns:
            raise ValueError(
                f"timestamps_col '{timestamps_col}' not found in the "
                f"DataFrame")

        if timestamps_col is None and time_delta is None:
            raise ValueError(
                "Either 'timestamps_col' or 'time_delta' must be provided. "
                "Supply a column of per-row timestamp lists via "
                "'timestamps_col', or a constant step size via 'time_delta' "
                "to generate synthetic timestamps.")

        if timestamps_col is None and anchor_time is None:
            raise ValueError(
                "'anchor_time' is required when 'timestamps_col' is not "
                "provided.  Pass the forecast cutoff timestamp so that "
                "synthetic timestamps can be placed correctly.")

        if entity_col is not None and entity_col not in df.columns:
            raise ValueError(
                f"entity_col '{entity_col}' not found in the DataFrame")

        pk_col = entity_col if entity_col is not None else 'entity_id'

        if entity_col is None and pk_col in df.columns:
            raise ValueError(
                f"entity_id '{pk_col}' conflicts with an existing "
                f"column in the DataFrame.  Rename the existing column or "
                f"pass a custom 'entity_col'.")

        if time_delta is not None:
            td = time_delta
        else:
            assert timestamps_col is not None
            td = _infer_timedelta_from_timestamps(df[timestamps_col])

        pquery_amount, pquery_unit = _timedelta_to_pquery(td)

        df = df.reset_index(drop=True)

        drop_cols = [timeseries_col]
        if timestamps_col is not None:
            drop_cols.append(timestamps_col)

        entity_df = df.drop(columns=drop_cols)
        if entity_col is None:
            entity_df.insert(0, pk_col, pd.RangeIndex(len(df)))

        series_list = df[timeseries_col].tolist()

        if timestamps_col is not None:
            timestamps_list = df[timestamps_col].tolist()
        else:
            assert anchor_time is not None
            timestamps_list = [[
                anchor_time - (len(s) - i) * td for i in range(len(s))
            ] for s in series_list]

        fk_vals: list = []
        ts_vals: list = []
        val_vals: list = []
        entity_ids = entity_df[pk_col].tolist()

        for eid, series, timestamps in zip(entity_ids, series_list,
                                           timestamps_list):
            n = len(series)
            fk_vals.extend([eid] * n)
            ts_vals.extend(timestamps)
            val_vals.extend(series)

        target_df = pd.DataFrame({
            pk_col:
            fk_vals,
            timestamp_col_name:
            pd.to_datetime(ts_vals),
            value_col_name:
            pd.array(val_vals, dtype=float),
        })

        entity_table = LocalTable(
            entity_df,
            entity_table_name,
            primary_key=pk_col,
        )
        target_table = LocalTable(
            target_df,
            target_table_name,
            time_column=timestamp_col_name,
        )
        target_table[value_col_name].stype = Stype.numerical

        graph = cls([entity_table, target_table])
        graph.link(target_table_name, pk_col, entity_table_name)

        pquery = (f"PREDICT MAX({target_table_name}.{value_col_name}, "
                  f"0, {pquery_amount}, {pquery_unit}) "
                  f"FORECAST {num_timeframes} TIMEFRAMES "
                  f"FOR EACH {entity_table_name}.{pk_col}")

        return graph, pquery

    # Backend #################################################################

    @property
    def backend(self) -> DataBackend | None:
        backends = [table.backend for table in self._tables.values()]
        return backends[0] if len(backends) > 0 else None

    # Tables ##################################################################

    def has_table(self, name: str) -> bool:
        r"""Returns ``True`` if the graph has a table with name ``name``;
        ``False`` otherwise.
        """
        return name in self.tables

    def table(self, name: str) -> Table:
        r"""Returns the table with name ``name`` in the graph.

        Raises:
            KeyError: If ``name`` is not present in the graph.
        """
        if not self.has_table(name):
            raise KeyError(f"Table '{name}' not found in graph")
        return self.tables[name]

    @property
    def tables(self) -> dict[str, Table]:
        r"""Returns the dictionary of table objects."""
        return self._tables

    def add_table(self, table: Table) -> Self:
        r"""Adds a table to the graph.

        Args:
            table: The table to add.

        Raises:
            KeyError: If a table with the same name already exists in the
                graph.
            ValueError: If the table belongs to a different backend than the
                rest of the tables in the graph.
        """
        if table.name in self._tables:
            raise KeyError(f"Cannot add table with name '{table.name}' to "
                           f"this graph; table names must be globally unique.")

        if self.backend is not None and table.backend != self.backend:
            raise ValueError(f"Cannot register a table with backend "
                             f"'{table.backend}' to this graph since other "
                             f"tables have backend '{self.backend}'.")

        self._tables[table.name] = table

        return self

    def remove_table(self, name: str) -> Self:
        r"""Removes a table with ``name`` from the graph.

        Args:
            name: The table to remove.

        Raises:
            KeyError: If no such table is present in the graph.
        """
        if not self.has_table(name):
            raise KeyError(f"Table '{name}' not found in the graph")

        del self._tables[name]

        self._edges = [
            edge for edge in self._edges
            if edge.src_table != name and edge.dst_table != name
        ]

        return self

    @property
    def metadata(self) -> pd.DataFrame:
        r"""Returns a :class:`pandas.DataFrame` object containing metadata
        information about the tables in this graph.

        The returned dataframe has columns ``"Name"``, ``"Primary Key"``,
        ``"Time Column"``, and ``"End Time Column"``, which provide an
        aggregated view of the properties of the tables of this graph.

        Example:
            >>> # doctest: +SKIP
            >>> import kumoai.rfm as rfm
            >>> graph = rfm.Graph(tables=...).infer_metadata()
            >>> graph.metadata  # doctest: +SKIP
                Name   Primary Key  Time Column  End Time Column
            0   users  user_id      -            -
        """
        tables = list(self.tables.values())

        return pd.DataFrame({
            'Name':
            pd.Series(dtype=str, data=[t.name for t in tables]),
            'Primary Key':
            pd.Series(dtype=str, data=[t._primary_key or '-' for t in tables]),
            'Time Column':
            pd.Series(dtype=str, data=[t._time_column or '-' for t in tables]),
            'End Time Column':
            pd.Series(
                dtype=str,
                data=[t._end_time_column or '-' for t in tables],
            ),
        })

    def print_metadata(self) -> None:
        r"""Prints the :meth:`~Graph.metadata` of the graph."""
        msg = "Graph Metadata"
        if not in_tmux():
            msg = f"🗂️ {msg}"
        display.title(msg)

        display.dataframe(self.metadata)

    def infer_metadata(self, verbose: bool = True) -> Self:
        r"""Infers metadata for all tables in the graph.

        Args:
            verbose: Whether to print verbose output.

        Note:
            For more information, please see
            :meth:`kumoai.rfm.Table.infer_metadata`.
        """
        for table in self.tables.values():
            table.infer_metadata(verbose=False)

        if verbose:
            self.print_metadata()

        return self

    # Edges ###################################################################

    @property
    def edges(self) -> list[Edge]:
        r"""Returns the edges of the graph."""
        return self._edges

    def print_links(self) -> None:
        r"""Prints the :meth:`~Graph.edges` of the graph."""
        edges = sorted([(
            edge.dst_table,
            self[edge.dst_table]._primary_key,
            edge.src_table,
            edge.fkey,
        ) for edge in self.edges])

        if in_tmux():
            display.title("Graph Links (FK <> PK)")
        else:
            display.title("🕸️ Graph Links (FK ↔️ PK)")
        if len(edges) > 0:
            items: list[str] = []
            for edge in edges:
                fkey = f"`{edge[2]}.{edge[3]}`"
                pkey = f"`{edge[0]}.{edge[1]}`"
                if in_tmux():
                    item = f"{fkey} <> {pkey}"
                else:
                    item = f"{fkey} ↔️ {pkey}"
                items.append(item)
            display.unordered_list(items)
        else:
            display.italic("No links registered")

    def link(
        self,
        src_table: str | Table,
        fkey: str,
        dst_table: str | Table,
    ) -> Self:
        r"""Links two tables (``src_table`` and ``dst_table``) from the foreign
        key ``fkey`` in the source table to the primary key in the destination
        table.

        The link is treated as bidirectional.

        Args:
            src_table: The name of the source table of the edge. This table
                must have a foreign key with name :obj:`fkey` that links to the
                primary key in the destination table.
            fkey: The name of the foreign key in the source table.
            dst_table: The name of the destination table of the edge. This
                table must have a primary key that links to the source table's
                foreign key.

        Raises:
            ValueError: if the edge is already present in the graph, if the
                source table does not exist in the graph, if the destination
                table does not exist in the graph, if the source key does not
                exist in the source table.
        """
        if isinstance(src_table, Table):
            src_table = src_table.name
        assert isinstance(src_table, str)

        if isinstance(dst_table, Table):
            dst_table = dst_table.name
        assert isinstance(dst_table, str)

        edge = Edge(src_table, fkey, dst_table)

        if edge in self.edges:
            raise ValueError(f"{edge} already exists in the graph")

        if not self.has_table(src_table):
            raise ValueError(f"Source table '{src_table}' does not exist in "
                             f"the graph")

        if not self.has_table(dst_table):
            raise ValueError(f"Destination table '{dst_table}' does not exist "
                             f"in the graph")

        if not self[src_table].has_column(fkey):
            raise ValueError(f"Source key '{fkey}' does not exist as a column "
                             f"in source table '{src_table}'")

        if not Stype.ID.supports_dtype(self[src_table][fkey].dtype):
            raise ValueError(f"Cannot use '{fkey}' in source table "
                             f"'{src_table}' as a foreign key due to its "
                             f"incompatible data type. Foreign keys must have "
                             f"data type 'int', 'float' or 'string' "
                             f"(got '{self[src_table][fkey].dtype}')")

        self[src_table][fkey].stype = Stype.ID
        self._edges.append(edge)

        return self

    def unlink(
        self,
        src_table: str | Table,
        fkey: str,
        dst_table: str | Table,
    ) -> Self:
        r"""Removes an :class:`~kumoai.graph.Edge` from the graph.

        Args:
            src_table: The name of the source table of the edge.
            fkey: The name of the foreign key in the source table.
            dst_table: The name of the destination table of the edge.

        Raises:
            ValueError: if the edge is not present in the graph.
        """
        if isinstance(src_table, Table):
            src_table = src_table.name
        assert isinstance(src_table, str)

        if isinstance(dst_table, Table):
            dst_table = dst_table.name
        assert isinstance(dst_table, str)

        edge = Edge(src_table, fkey, dst_table)

        if edge not in self.edges:
            raise ValueError(f"{edge} is not present in the graph")

        self._edges.remove(edge)

        return self

    def infer_links(self, verbose: bool = True) -> Self:
        r"""Infers missing links for the tables and adds them as edges to the
        graph.

        Args:
            verbose: Whether to print verbose output.
        """
        known_edges = {(edge.src_table, edge.fkey) for edge in self.edges}

        for table in self.tables.values():  # Use links from source metadata:
            if not any(column.is_source for column in table.columns):
                continue
            for fkey in table._source_foreign_key_dict.values():
                if fkey.name not in table:
                    continue
                if not table[fkey.name].is_source:
                    continue
                if (table.name, fkey.name) in known_edges:
                    continue
                dst_table_names = [
                    table.name for table in self.tables.values()
                    if table.source_name == fkey.dst_table
                ]
                if len(dst_table_names) != 1:
                    continue
                dst_table = self[dst_table_names[0]]
                if dst_table._primary_key != fkey.primary_key:
                    continue
                if not dst_table[fkey.primary_key].is_source:
                    continue
                self.link(table.name, fkey.name, dst_table.name)
                known_edges.add((table.name, fkey.name))

        # A list of primary key candidates (+score) for every column:
        candidate_dict: dict[
            tuple[str, str],
            list[tuple[str, float]],
        ] = defaultdict(list)

        for dst_table in self.tables.values():
            dst_key = dst_table.primary_key

            if dst_key is None:
                continue

            assert dst_key.dtype is not None
            dst_number = dst_key.dtype.is_int() or dst_key.dtype.is_float()
            dst_string = dst_key.dtype.is_string()

            dst_table_name = dst_table.name.lower()
            dst_key_name = dst_key.name.lower()

            for src_table in self.tables.values():
                src_table_name = src_table.name.lower()

                for src_key in src_table.columns:
                    if (src_table.name, src_key.name) in known_edges:
                        continue

                    if src_key in {
                            src_table.primary_key,
                            src_table.time_column,
                            src_table.end_time_column,
                    }:
                        continue  # Cannot link to special columns.

                    src_number = (src_key.dtype.is_int()
                                  or src_key.dtype.is_float())
                    src_string = src_key.dtype.is_string()

                    if src_number != dst_number or src_string != dst_string:
                        continue  # Non-compatible data types.

                    src_key_name = src_key.name.lower()

                    score = 0.0

                    # Name similarity:
                    if src_key_name == dst_key_name:
                        score += 7.0
                    elif (dst_key_name != 'id'
                          and src_key_name.endswith(dst_key_name)):
                        score += 4.0
                    elif src_key_name.endswith(  # e.g., user.id -> user_id
                            f'{dst_table_name}_{dst_key_name}'):
                        score += 4.0
                    elif src_key_name.endswith(  # e.g., user.id -> userid
                            f'{dst_table_name}{dst_key_name}'):
                        score += 4.0
                    elif (dst_table_name.endswith('s') and
                          src_key_name.endswith(  # e.g., users.id -> user_id
                              f'{dst_table_name[:-1]}_{dst_key_name}')):
                        score += 4.0
                    elif (dst_table_name.endswith('s') and
                          src_key_name.endswith(  # e.g., users.id -> userid
                              f'{dst_table_name[:-1]}{dst_key_name}')):
                        score += 4.0
                    elif src_key_name.endswith(dst_table_name):
                        score += 4.0  # e.g., users -> users
                    elif (dst_table_name.endswith('s')  # e.g., users -> user
                          and src_key_name.endswith(dst_table_name[:-1])):
                        score += 4.0
                    elif ((src_key_name == 'parentid'
                           or src_key_name == 'parent_id')
                          and src_table_name == dst_table_name):
                        score += 2.0

                    # `rel-bench` hard-coding :(
                    elif (src_table.name == 'posts'
                          and src_key.name == 'AcceptedAnswerId'
                          and dst_table.name == 'posts'):
                        score += 2.0
                    elif (src_table.name == 'user_friends'
                          and src_key.name == 'friend'
                          and dst_table.name == 'users'):
                        score += 3.0

                    # For non-exact matching, at least one additional
                    # requirement needs to be met.

                    # Exact data type compatibility:
                    if src_key.stype == Stype.ID:
                        score += 2.0

                    if src_key.dtype == dst_key.dtype:
                        score += 1.0

                    # Cardinality ratio:
                    if (src_table._num_rows is not None
                            and dst_table._num_rows is not None
                            and src_table._num_rows > dst_table._num_rows):
                        score += 1.0

                    if score < 5.0:
                        continue

                    candidate_dict[(src_table.name, src_key.name)].append(
                        (dst_table.name, score))

        for (src_table_name, src_key_name), scores in candidate_dict.items():
            scores.sort(key=lambda x: x[-1], reverse=True)

            if len(scores) > 1 and scores[0][1] == scores[1][1]:
                continue  # Cannot uniquely infer link.

            dst_table_name = scores[0][0]
            self.link(src_table_name, src_key_name, dst_table_name)

        if verbose:
            self.print_links()

        return self

    # Metadata ################################################################

    def validate(self) -> Self:
        r"""Validates the graph to ensure that all relevant metadata is
        specified for its tables and edges.

        Concretely, validation ensures that edges properly link foreign keys to
        primary keys between valid tables.
        It additionally ensures that primary and foreign keys between tables
        in an :class:`~kumoai.graph.Edge` are of the same data type.

        Raises:
            ValueError: if validation fails.
        """
        if len(self.tables) == 0:
            raise ValueError("At least one table needs to be added to the "
                             "graph")

        backends = {table.backend for table in self._tables.values()}
        if len(backends) != 1:
            raise ValueError("Found multiple table backends in the graph")

        for edge in self.edges:
            src_table, fkey, dst_table = edge

            src_key = self[src_table][fkey]
            dst_key = self[dst_table].primary_key

            # Check that the destination table defines a primary key:
            if dst_key is None:
                raise ValueError(f"Edge {edge} is invalid since table "
                                 f"'{dst_table}' does not have a primary key. "
                                 f"Add either a primary key or remove the "
                                 f"link before proceeding.")

            # Ensure that foreign key is not a primary key:
            src_pkey = self[src_table].primary_key
            if src_pkey is not None and src_pkey.name == fkey:
                raise ValueError(f"Cannot treat the primary key of table "
                                 f"'{src_table}' as a foreign key. Remove "
                                 f"either the primary key or the link before "
                                 f"before proceeding.")

            if self.backend == DataBackend.LOCAL:
                # Check that fkey/pkey have valid and consistent data types:
                assert src_key.dtype is not None
                src_number = src_key.dtype.is_int() or src_key.dtype.is_float()
                src_string = src_key.dtype.is_string()
                assert dst_key.dtype is not None
                dst_number = dst_key.dtype.is_int() or dst_key.dtype.is_float()
                dst_string = dst_key.dtype.is_string()

                if not src_number and not src_string:
                    raise ValueError(
                        f"{edge} is invalid as foreign key must be a number "
                        f"or string (got '{src_key.dtype}'")

                if src_number != dst_number or src_string != dst_string:
                    raise ValueError(
                        f"{edge} is invalid as foreign key '{fkey}' and "
                        f"primary key '{dst_key.name}' have incompatible data "
                        f"types (got foreign key data type '{src_key.dtype}' "
                        f"and primary key data type '{dst_key.dtype}')")

        return self

    # Visualization ###########################################################

    def _to_graphviz(
        self,
        format: str | None = None,
        show_columns: bool = True,
    ) -> 'graphviz.Graph':
        r"""Returns a ``graphviz.Graph`` representation of the graph.

        Args:
            format: The rendering output format.
            show_columns: Whether to show all columns of every table in the
                graph. If ``False``, will only show the primary key, foreign
                key(s), and time column of each table.

        Returns:
            The ``graphviz.Graph`` object.
        """
        graphviz = import_module('graphviz')
        graph = graphviz.Graph(format=format)

        def left_align(keys: list[str]) -> str:
            if len(keys) == 0:
                return ""
            return '\\l'.join(keys) + '\\l'

        fkeys_dict: dict[str, list[str]] = defaultdict(list)
        for src_table_name, fkey_name, _ in self.edges:
            fkeys_dict[src_table_name].append(fkey_name)

        for table_name, table in self.tables.items():
            keys = []
            if primary_key := table.primary_key:
                keys += [f'{primary_key.name}: PK ({primary_key.dtype})']
            keys += [
                f'{fkey_name}: FK ({self[table_name][fkey_name].dtype})'
                for fkey_name in fkeys_dict[table_name]
            ]
            if time_column := table.time_column:
                keys += [f'{time_column.name}: Time ({time_column.dtype})']
            if end_time_column := table.end_time_column:
                keys += [
                    f'{end_time_column.name}: '
                    f'End Time ({end_time_column.dtype})'
                ]
            key_repr = left_align(keys)

            columns = []
            if show_columns:
                columns += [
                    f'{column.name}: {column.stype} ({column.dtype})'
                    for column in table.columns
                    if column.name not in fkeys_dict[table_name] and
                    column.name != table._primary_key and column.name != table.
                    _time_column and column.name != table._end_time_column
                ]
            column_repr = left_align(columns)

            if len(keys) > 0 and len(columns) > 0:
                label = f'{{{table_name}|{key_repr}|{column_repr}}}'
            elif len(keys) > 0:
                label = f'{{{table_name}|{key_repr}}}'
            elif len(columns) > 0:
                label = f'{{{table_name}|{column_repr}}}'
            else:
                label = f'{{{table_name}}}'

            graph.node(table_name, shape='record', label=label)

        for src_table_name, fkey_name, dst_table_name in self.edges:
            if self[dst_table_name]._primary_key is None:
                continue  # Invalid edge.

            pkey_name = self[dst_table_name]._primary_key

            if fkey_name != pkey_name:
                label = f' {fkey_name}\n< >\n{pkey_name} '
            else:
                label = f' {fkey_name} '

            graph.edge(
                src_table_name,
                dst_table_name,
                label=label,
                headlabel='1',
                taillabel='*',
                minlen='2',
                fontsize='11pt',
                labeldistance='1.5',
            )

        return graph

    def _to_mermaid(self, show_columns: bool = True) -> str:
        r"""Returns a Mermaid ER diagram string representation of the graph.

        Args:
            show_columns: Whether to show all columns of every table in the
                graph. If ``False``, will only show the primary key, foreign
                key(s), and time column of each table.

        Returns:
            A string containing the Mermaid ER diagram.
        """
        fkeys_dict: dict[str, list[str]] = defaultdict(list)
        for src_table_name, fkey_name, _ in self.edges:
            fkeys_dict[src_table_name].append(fkey_name)

        lines = ["erDiagram"]

        for table_name, table in self.tables.items():
            lines.append(f"{' ' * 4}{table_name} {{")
            if pkey := table.primary_key:
                lines.append(f"{' ' * 8}{pkey.stype} {pkey.name} PK")
            for fkey_name in fkeys_dict[table_name]:
                fkey = table[fkey_name]
                lines.append(f"{' ' * 8}{fkey.stype} {fkey.name} FK")
            if time_col := table.time_column:
                lines.append(f"{' ' * 8}{time_col.stype} {time_col.name}")
            if time_col := table.end_time_column:
                lines.append(f"{' ' * 8}{time_col.dtype} {time_col.name}")

            if show_columns:
                for column in table.columns:
                    if column.name in fkeys_dict[table_name]:
                        continue
                    if column.name == table._primary_key:
                        continue
                    if column.name == table._time_column:
                        continue
                    if column.name == table._end_time_column:
                        continue
                    lines.append(f"{' ' * 8}{column.stype} {column.name}")

            lines.append(f"{' ' * 4}}}")

        if len(self.edges) > 0:
            lines.append("")

        for src_table, fkey, dst_table in self.edges:
            lines.append(f"{' ' * 4}{dst_table} o|--o{{ {src_table} : {fkey}")

        return '\n'.join(lines)

    def visualize(
        self,
        path: Path | str | io.BytesIO | None = None,
        show_columns: bool = True,
        backend: Literal['auto', 'graphviz', 'mermaid'] = 'auto',
    ) -> None:
        r"""Visualizes the tables and edges in this graph.

        Args:
            path: A path to write the produced image to. If ``None``, the image
                will not be written to disk.
            show_columns: Whether to show all columns of every table in the
                graph. If ``False``, will only show the primary key, foreign
                key(s), and time column of each table.
            backend: The visualization backend to use. ``auto`` chooses the
                backend based on environment and availability.
        """
        def has_graphviz_executables() -> bool:
            if find_spec('graphviz') is None:
                return False

            graphviz = import_module('graphviz')

            if in_streamlit_notebook() and path is None:
                return True

            try:
                graphviz.Digraph().pipe()
            except graphviz.backend.ExecutableNotFound:
                return False

            return True

        path = Path(path) if isinstance(path, str) else path

        suffix: str | None = None
        if isinstance(path, Path):
            suffix = path.suffix.removeprefix('.')
            if suffix == '':
                raise ValueError(f"Missing file extension in path '{path}'")
        elif isinstance(path, io.BytesIO):
            suffix = 'pdf'

        if backend == 'auto':
            if has_graphviz_executables():
                backend = 'graphviz'
            elif find_spec('mermaid') is not None:
                backend = 'mermaid'
            else:
                raise ValueError("Could not resolve visualization backend. "
                                 "Make sure that either 'graphviz' or "
                                 "'mermaid-py' are installed.")

        if backend == 'graphviz':
            if find_spec('graphviz') is None:
                raise ImportError("The 'graphviz' package is required for "
                                  "visualization")

            if not has_graphviz_executables():
                raise RuntimeError("Could not visualize graph as 'graphviz' "
                                   "executables are not installed. These "
                                   "dependencies are required in addition to "
                                   "the 'graphviz' Python package. Please "
                                   "install them as described at "
                                   "'https://graphviz.org/download'.")

            obj = self._to_graphviz(format=suffix, show_columns=show_columns)

            if isinstance(path, Path):
                obj.render(path.with_suffix(''), cleanup=True)
            elif isinstance(path, io.BytesIO):
                path.write(obj.pipe())
            elif in_streamlit_notebook():
                import streamlit as st
                st.graphviz_chart(obj)
            elif in_jupyter_notebook():
                from IPython.display import display
                display(obj)
            else:
                try:
                    stderr_buffer = io.StringIO()
                    with contextlib.redirect_stderr(stderr_buffer):
                        obj.view(cleanup=True)
                    if stderr_buffer.getvalue():
                        warnings.warn("Could not visualize graph for backend "
                                      f"'{backend}' since your system does "
                                      f"not know how to open or display PDF "
                                      f"files from the command line. Please "
                                      f"specify `visualize(path=...)` and "
                                      f"open the generated file yourself.")
                except Exception as e:
                    warnings.warn(
                        f"Could not visualize graph due to an "
                        f"unexpected error in 'graphviz'. Error: {e}")

        elif backend == 'mermaid':
            if find_spec('mermaid') is None:
                raise ImportError("The 'mermaid-py' package is required for "
                                  "visualization")

            md = import_module('mermaid')
            obj = md.Mermaid(md.Graph('graph', self._to_mermaid(show_columns)))

            if isinstance(path, Path):
                if suffix == 'png':
                    obj.to_png(path)
                elif suffix == 'svg':
                    obj.to_svg(path)
                else:
                    raise ValueError(f"File extension '{suffix}' not "
                                     f"supported for 'mermaid' visualization. "
                                     f"Expected either 'png' or 'svg'.")
            elif isinstance(path, io.BytesIO):
                path.write(obj.img_response.content)
            elif in_jupyter_notebook():
                from IPython.display import display
                display(obj)
            else:
                warnings.warn(f"Could not visualize graph for backend "
                              f"'{backend}'. Please specify "
                              f"`visualize(path=...)` and open the generated "
                              f"file yourself.")

        else:
            raise ValueError(f"Could not resolve visualization backend "
                             f"'{backend}'")

    # Helpers #################################################################

    def _to_api_graph_definition(self) -> GraphDefinition:
        tables: dict[str, TableDefinition] = {}
        col_groups: list[ColumnKeyGroup] = []
        for table_name, table in self.tables.items():
            tables[table_name] = table._to_api_table_definition()
            if table.primary_key is None:
                continue
            keys = [ColumnKey(table_name, table.primary_key.name)]
            for edge in self.edges:
                if edge.dst_table == table_name:
                    keys.append(ColumnKey(edge.src_table, edge.fkey))
            keys = sorted(
                list(set(keys)),
                key=lambda x: f'{x.table_name}.{x.col_name}',
            )
            if len(keys) > 1:
                col_groups.append(ColumnKeyGroup(tuple(keys)))
        return GraphDefinition(tables, col_groups)

    def update_connection(
        self,
        connection: (AdbcSqliteConnection | AdbcDuckDBConnection
                     | SnowflakeConnection),
    ) -> None:
        r"""Updates the connection to a database."""
        if self._connection is not None:
            self._connection.close()

        self._connection = None

        for table in self.tables.values():
            if table.backend == DataBackend.SQLITE:
                from adbc_driver_sqlite.dbapi import AdbcSqliteConnection

                from kumoai.rfm.backend.sqlite import SQLiteTable
                assert isinstance(table, SQLiteTable)
                assert isinstance(connection, AdbcSqliteConnection)
                table._connection = connection
            if table.backend == DataBackend.DUCKDB:
                from adbc_driver_duckdb.dbapi import Connection

                from kumoai.rfm.backend.duckdb import DuckDBTable
                assert isinstance(table, DuckDBTable)
                assert isinstance(connection, Connection)
                table._connection = connection
            if table.backend == DataBackend.SNOWFLAKE:
                from snowflake.connector import SnowflakeConnection

                from kumoai.rfm.backend.snow import SnowTable
                assert isinstance(table, SnowTable)
                assert isinstance(connection, SnowflakeConnection)
                table._connection = connection

    # Class properties ########################################################

    def __hash__(self) -> int:
        return hash((tuple(self.edges), tuple(sorted(self.tables.keys()))))

    def __contains__(self, name: str) -> bool:
        return self.has_table(name)

    def __getitem__(self, name: str) -> Table:
        return self.table(name)

    def __delitem__(self, name: str) -> None:
        self.remove_table(name)

    def __repr__(self) -> str:
        tables = '\n'.join(f'    {table},' for table in self.tables)
        tables = f'[\n{tables}\n  ]' if len(tables) > 0 else '[]'
        edges = '\n'.join(
            f'    {edge.src_table}.{edge.fkey}'
            f' ⇔ {edge.dst_table}.{self[edge.dst_table]._primary_key},'
            for edge in self.edges)
        edges = f'[\n{edges}\n  ]' if len(edges) > 0 else '[]'
        return (f'{self.__class__.__name__}(\n'
                f'  tables={tables},\n'
                f'  edges={edges},\n'
                f')')

    def __del__(self) -> None:
        if self._connection is not None:
            self._connection.close()


def _timedelta_to_pquery(td: pd.Timedelta) -> tuple[int, str]:
    """Convert a positive Timedelta to a ``(amount, unit)`` pair for a
    pquery time range.
    """
    total_seconds = int(td.total_seconds())
    if total_seconds <= 0:
        raise ValueError(f"time_delta must be positive, got {td}")
    if total_seconds % (24 * 3600) == 0:
        return total_seconds // (24 * 3600), 'days'
    if total_seconds % 3600 == 0:
        return total_seconds // 3600, 'hours'
    if total_seconds % 60 == 0:
        return total_seconds // 60, 'minutes'
    raise ValueError(
        f"time_delta {td} must be a whole number of minutes, hours, or days")


def _infer_timedelta_from_timestamps(
        timestamps_series: pd.Series) -> pd.Timedelta:
    """Infer the median step size from a Series of per-entity timestamp lists.

    Emits a warning when the observed step sizes are not all equal.
    """
    deltas: list[pd.Timedelta] = []
    for ts_list in timestamps_series:
        ts = pd.to_datetime(list(ts_list)).sort_values()
        if len(ts) >= 2:
            deltas.extend(ts[1:] - ts[:-1])
    if not deltas:
        raise ValueError(
            "Cannot infer 'time_delta' from 'timestamps_col': all series "
            "have fewer than 2 observations. Please provide 'time_delta' "
            "explicitly.")
    delta_series = pd.Series(deltas)
    median_td = Timedelta(delta_series.median())
    if delta_series.nunique() > 1:
        warnings.warn(
            f"Observed step sizes in 'timestamps_col' are not all equal "
            f"(found {delta_series.nunique()} distinct intervals). "
            f"Using the median ({median_td}) as 'time_delta' for the "
            f"predictive query. Pass 'time_delta' explicitly to override.",
            stacklevel=3,
        )
    return median_td

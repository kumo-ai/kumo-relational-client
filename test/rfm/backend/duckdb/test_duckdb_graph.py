from pathlib import Path

import pytest

from kumoai.graph import Edge
from kumoai.rfm import Graph

try:
    from kumoai.rfm.backend.duckdb import Connection, DuckDBTable, connect
except ImportError:
    pytest.skip("'duckdb' extension not installed", allow_module_level=True)


def test_graph(connection: Connection) -> None:
    graph = Graph(tables=[
        DuckDBTable(connection, name='USERS'),
        DuckDBTable(connection, name='ORDERS'),
        DuckDBTable(connection, name='ITEMS'),
    ])

    assert len(graph.tables) == 3

    assert graph['USERS'].primary_key is None
    assert graph['ORDERS'].primary_key is None
    assert graph['ITEMS'].primary_key is not None
    assert graph['ITEMS'].primary_key.name == 'ITEM_ID'

    assert graph['USERS'].time_column is None
    assert graph['ORDERS'].time_column is None
    assert graph['ITEMS'].time_column is None

    assert Edge('ORDERS', 'USER_ID', 'USERS') not in graph.edges
    assert Edge('ORDERS', 'ITEM_ID', 'ITEMS') in graph.edges


@pytest.mark.parametrize('use_open_connection', [False, True])
def test_from_duckdb(
    tmp_path: Path,
    connection: Connection,
    use_open_connection: bool,
) -> None:
    if use_open_connection:
        graph = Graph.from_duckdb(connection, verbose=False)
    else:
        # On Windows, DuckDB enforces an exclusive file lock, so we have to
        # release the fixture's connection before re-opening by path.
        connection.close()
        graph = Graph.from_duckdb(tmp_path / 'data.duckdb', verbose=False)

    assert len(graph.tables) == 3

    assert graph['USERS'].primary_key is not None
    assert graph['USERS'].primary_key.name == 'USER_ID'
    assert graph['ORDERS'].primary_key is None
    assert graph['ITEMS'].primary_key is not None
    assert graph['ITEMS'].primary_key.name == 'ITEM_ID'

    assert graph['USERS'].time_column is not None
    assert graph['USERS'].time_column.name == 'DOB'
    assert graph['ORDERS'].time_column is not None
    assert graph['ORDERS'].time_column.name == 'DATE'
    assert graph['ITEMS'].time_column is None

    assert Edge('ORDERS', 'USER_ID', 'USERS') in graph.edges
    assert Edge('ORDERS', 'ITEM_ID', 'ITEMS') in graph.edges


def test_stateless(connection: Connection) -> None:
    table = DuckDBTable(
        connection,
        name='USERS',
        columns=[
            dict(name='USER_ID', dtype='int', stype='ID'),
            dict(name='IS_FLAG', dtype='bool', stype='categorical'),
            dict(name='AGE', expr='2 * AGE', dtype='float', stype='numerical'),
            dict(name='GENDER', dtype='string', stype='categorical'),
            dict(name='DOB', dtype='date', stype='timestamp'),
        ],
        primary_key='USER_ID',
        time_column='DOB',
    )
    graph = Graph(tables=[table])

    assert '_source_column_dict' in graph['USERS'].__dict__
    assert '_source_primary_key' not in graph['USERS'].__dict__
    assert '_source_foreign_key_dict' in graph['USERS'].__dict__
    assert '_source_sample_df' not in graph['USERS'].__dict__
    assert '_num_rows' not in graph['USERS'].__dict__
    assert len(graph['USERS']._expr_sample_df.columns) == 0


def test_update_connection(tmp_path: Path) -> None:
    connection = connect(tmp_path / 'one.duckdb')
    with connection.cursor() as cursor:
        cursor.execute("CREATE TABLE USERS (USER_ID INTEGER UNIQUE)")
        cursor.execute("INSERT INTO USERS VALUES (1)")
        connection.commit()

    new_connection = connect(tmp_path / 'two.duckdb')
    with new_connection.cursor() as cursor:
        cursor.execute("CREATE TABLE USERS (USER_ID INTEGER UNIQUE)")
        cursor.execute("INSERT INTO USERS VALUES (1), (2)")
        new_connection.commit()

    graph = Graph(tables=[DuckDBTable(connection, name='USERS')])
    graph.update_connection(new_connection)

    assert graph['USERS']._connection is new_connection  # type: ignore
    assert graph['USERS']._get_num_rows() == 2

    connection.close()
    new_connection.close()

from pathlib import Path

import pytest

from kumoai.graph import Edge
from kumoai.rfm import Graph

try:
    from kumoai.rfm.backend.sqlite import Connection, SQLiteTable
except ImportError:
    pytest.skip("'sqlite' extension not installed", allow_module_level=True)


def test_graph(connection: Connection) -> None:
    graph = Graph(tables=[
        SQLiteTable(connection, name='USERS'),
        SQLiteTable(connection, name='ORDERS'),
        SQLiteTable(connection, name='ITEMS'),
    ])

    assert len(graph.tables) == 3

    assert graph['USERS'].primary_key is None
    assert graph['ORDERS'].primary_key is None
    assert graph['ITEMS'].primary_key is not None
    assert graph['ITEMS'].primary_key.name == 'ITEM_ID'

    assert graph['USERS'].time_column is None
    assert graph['ORDERS'].time_column is None
    assert graph['ITEMS'].time_column is None

    assert Edge('ORDERS', 'ITEM_ID', 'ITEMS') in graph.edges
    assert Edge('ORDERS', 'USER_ID', 'USERS') not in graph.edges


@pytest.mark.parametrize('use_open_connection', [False, True])
def test_from_sqlite(
    tmp_path: Path,
    connection: Connection,
    use_open_connection: bool,
) -> None:
    if use_open_connection:
        graph = Graph.from_sqlite(connection, verbose=False)
    else:
        graph = Graph.from_sqlite(tmp_path / 'data.db', verbose=False)

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
    assert graph['ITEMS'].time_column is None  # TODO

    assert Edge('ORDERS', 'USER_ID', 'USERS') in graph.edges
    assert Edge('ORDERS', 'ITEM_ID', 'ITEMS') in graph.edges


def test_stateless(connection: Connection) -> None:
    table = SQLiteTable(
        connection,
        name='USERS',
        columns=[
            dict(name='USER_ID', dtype='int', stype='ID'),
            dict(name='IS_FLAG', dtype='bool', stype='categorical'),
            dict(name='AGE', expr='2 * AGE', dtype='float', stype='numerical'),
            dict(name='GENDER', dtype='string', stype='categorical'),
            dict(name='DOB', dtype='string', stype='timestamp'),
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

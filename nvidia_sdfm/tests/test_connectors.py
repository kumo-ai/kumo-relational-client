from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from nvidia_sdfm.core.connectors import _resolve_sql, read
from nvidia_sdfm.errors import SdfmError


def test_read_unknown_connector_raises():
    with pytest.raises(SdfmError) as excinfo:
        read('databricks')
    assert excinfo.value.code == 'UNKNOWN_CONNECTOR'


def test_read_local_from_dataframe():
    frame = pd.DataFrame({'a': [1, 2]})
    assert read('local', data=frame) is frame


def test_read_local_from_csv(tmp_path):
    path = tmp_path / 'table.csv'
    pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']}).to_csv(path, index=False)
    frame = read('local', path=str(path))
    assert frame.shape == (2, 2)


def test_read_local_without_data_or_path_raises():
    with pytest.raises(SdfmError) as excinfo:
        read('local')
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


def test_read_sqlite_by_table_and_query(tmp_path):
    database = str(tmp_path / 'db.sqlite')
    with sqlite3.connect(database) as connection:
        pd.DataFrame({'a': [1, 2, 3]}).to_sql('items', connection, index=False)
    assert len(read('sqlite', database=database, table='items')) == 3
    assert len(read('sqlite', database=database,
                    query='SELECT * FROM items WHERE a > 1')) == 2


@pytest.mark.parametrize('kwargs', [
    {},
    {'table': 'items', 'query': 'SELECT 1'},
])
def test_resolve_sql_requires_exactly_one_source(kwargs):
    with pytest.raises(SdfmError) as excinfo:
        _resolve_sql(table=kwargs.get('table'), query=kwargs.get('query'))
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


@pytest.mark.parametrize('table', [
    'items',
    'main.items',
    'DB.SCHEMA.TABLE_1',
    '_private$tbl',
])
def test_resolve_sql_accepts_valid_identifiers(table):
    assert _resolve_sql(table=table, query=None) == f'SELECT * FROM {table}'


@pytest.mark.parametrize('table', [
    'items; DROP TABLE users',
    'items --',
    'items OR 1=1',
    '1items',
    'items..b',
    '',
])
def test_resolve_sql_rejects_invalid_identifiers(table):
    with pytest.raises(SdfmError) as excinfo:
        _resolve_sql(table=table, query=None)
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'

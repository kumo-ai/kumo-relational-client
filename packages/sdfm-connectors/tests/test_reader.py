# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sqlite3

import pandas as pd
import pyarrow as pa
import pytest

from sdfm_connectors import read, read_table
from sdfm_connectors.sql import ConnectorError

LARGE_ID = 9007199254740993


class _FakeCursor:
    def __init__(self, arrow_table, pandas_frame=None):
        self.arrow_table = arrow_table
        self.pandas_frame = pandas_frame
        self.pandas_calls = 0

    def execute(self, sql):
        pass

    def fetch_arrow_all(self):
        return self.arrow_table

    def fetch_pandas_all(self):
        self.pandas_calls += 1
        return self.pandas_frame

    def close(self):
        pass


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def test_read_unknown_connector_raises():
    with pytest.raises(ConnectorError) as excinfo:
        read('mysql')
    assert excinfo.value.code == 'UNKNOWN_CONNECTOR'


def test_read_local_from_dataframe():
    frame = pd.DataFrame({'a': [1, 2]})
    assert read('local', data=frame) is frame


def test_read_local_from_dict():
    assert read('local', data={'a': [1, 2, 3]}).shape == (3, 1)


def test_read_local_from_csv(tmp_path):
    path = tmp_path / 'table.csv'
    pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']}).to_csv(path, index=False)
    assert read('local', path=str(path)).shape == (2, 2)


def test_read_local_from_parquet(tmp_path):
    path = tmp_path / 'table.parquet'
    pd.DataFrame({'a': [1, 2, 3]}).to_parquet(path)
    assert read('local', path=str(path)).shape == (3, 1)


def test_read_local_without_data_or_path_raises():
    with pytest.raises(ConnectorError) as excinfo:
        read('local')
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


def test_read_local_rejects_both_data_and_path(tmp_path):
    path = tmp_path / 'table.csv'
    pd.DataFrame({'a': [1]}).to_csv(path, index=False)
    with pytest.raises(ConnectorError) as excinfo:
        read('local', data={'a': [1]}, path=str(path))
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


def test_read_sqlite_by_table_and_query(tmp_path):
    database = str(tmp_path / 'db.sqlite')
    with sqlite3.connect(database) as connection:
        pd.DataFrame({'a': [1, 2, 3]}).to_sql('items', connection, index=False)
    assert len(read('sqlite', database=database, table='items')) == 3
    assert len(read('sqlite', database=database,
                    query='SELECT * FROM items WHERE a > 1')) == 2


def test_read_duckdb_in_memory_query():
    frame = read('duckdb', query='SELECT 1 AS a, 2 AS b')
    assert frame.shape == (1, 2)
    assert int(frame['a'][0]) == 1


def test_read_duckdb_by_table(tmp_path):
    database = str(tmp_path / 'db.duckdb')
    import duckdb
    connection = duckdb.connect(database)
    connection.execute('CREATE TABLE items AS SELECT * FROM range(5) AS t(a)')
    connection.close()
    assert len(read('duckdb', database=database, table='items')) == 5


def test_read_sqlite_keeps_nullable_integers_exact(tmp_path):
    database = str(tmp_path / 'ids.sqlite')
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE users (user_id BIGINT, age INTEGER)')
        connection.execute('INSERT INTO users VALUES (?, 20)', (LARGE_ID, ))
        connection.execute('INSERT INTO users VALUES (NULL, 30)')

    frame = read('sqlite', database=database, table='users')

    assert frame['user_id'].dtype == 'Int64'
    assert frame['user_id'][0] == LARGE_ID
    assert frame['user_id'].isna().tolist() == [False, True]
    assert frame['age'].dtype == 'int64'
    assert frame['age'].tolist() == [20, 30]


def test_read_table_prefers_arrow_over_pandas_fetch():
    cursor = _FakeCursor(
        pa.table({'user_id': pa.array([LARGE_ID, None], type=pa.int64())}))

    frame = read_table(_FakeConnection(cursor), table='users')

    assert cursor.pandas_calls == 0
    assert frame['user_id'].dtype == 'Int64'
    assert frame['user_id'][0] == LARGE_ID


def test_read_table_falls_back_to_pandas_fetch_without_arrow_result():
    cursor = _FakeCursor(None, pd.DataFrame({'user_id': []}))

    frame = read_table(_FakeConnection(cursor), table='users')

    assert cursor.pandas_calls == 1
    assert frame.empty

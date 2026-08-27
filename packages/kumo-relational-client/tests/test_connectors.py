# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

import kumo_relational_client.core.connectors as connectors_module
from kumo_relational_client.core.connectors import read
from kumo_relational_client.errors import MissingExtraError, RelationalError


def test_read_unknown_connector_raises():
    with pytest.raises(RelationalError) as excinfo:
        read('mysql')
    assert excinfo.value.code == 'UNKNOWN_CONNECTOR'


@pytest.mark.parametrize(
    'extra, driver',
    [
        ('snowflake', 'snowflake-connector-python'),
        ('s3', 's3fs'),
    ],
)
def test_missing_backend_maps_to_missing_extra_error(
    monkeypatch,
    extra,
    driver,
):
    from kumo_connectors.sql import MissingBackendError

    def raise_missing(source, **kwargs):
        raise MissingBackendError(extra, driver)

    monkeypatch.setattr(connectors_module, '_read', raise_missing)
    with pytest.raises(MissingExtraError) as excinfo:
        read(extra, table='t')
    assert excinfo.value.details['extra'] == extra
    assert f'kumo-relational-client[{extra}]' in str(excinfo.value)


def test_broken_driver_is_not_reported_as_missing_extra(monkeypatch):
    def raise_broken(source, **kwargs):
        raise ImportError('native ABI mismatch in installed driver')

    monkeypatch.setattr(connectors_module, '_read', raise_broken)
    with pytest.raises(RelationalError) as excinfo:
        read('duckdb', table='t')
    assert excinfo.value.code == 'DRIVER_LOAD_FAILED'
    assert not isinstance(excinfo.value, MissingExtraError)


def test_read_local_from_dataframe():
    frame = pd.DataFrame({'a': [1, 2]})
    assert read('local', data=frame) is frame


def test_read_local_from_csv(tmp_path):
    path = tmp_path / 'table.csv'
    pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']}).to_csv(path, index=False)
    frame = read('local', path=str(path))
    assert frame.shape == (2, 2)


def test_read_local_without_data_or_path_raises():
    with pytest.raises(RelationalError) as excinfo:
        read('local')
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


def test_query_failure_maps_to_predict_error_with_code(tmp_path):
    database = str(tmp_path / 'db.sqlite')
    with sqlite3.connect(database) as connection:
        pd.DataFrame({'a': [1]}).to_sql('items', connection, index=False)
    with pytest.raises(RelationalError) as excinfo:
        read('sqlite', database=database, query='SELECT * FROM missing')
    assert excinfo.value.code == 'QUERY_FAILED'
    assert excinfo.value.details['sql'] == 'SELECT * FROM missing'


def test_connect_failure_maps_to_predict_error_with_code(tmp_path):
    database = tmp_path / 'not-a-database.duckdb'
    database.write_text('this is not a duckdb database')
    with pytest.raises(RelationalError) as excinfo:
        read('duckdb', database=str(database), table='items')
    assert excinfo.value.code == 'CONNECT_FAILED'


def test_local_missing_file_maps_to_not_found(tmp_path):
    with pytest.raises(RelationalError) as excinfo:
        read('local', path=str(tmp_path / 'absent.csv'))
    assert excinfo.value.code == 'NOT_FOUND'


def test_read_sqlite_by_table_and_query(tmp_path):
    database = str(tmp_path / 'db.sqlite')
    with sqlite3.connect(database) as connection:
        pd.DataFrame({'a': [1, 2, 3]}).to_sql('items', connection, index=False)
    assert len(read('sqlite', database=database, table='items')) == 3
    assert (
        len(
            read(
                'sqlite',
                database=database,
                query='SELECT * FROM items WHERE a > 1',
            )
        )
        == 2
    )

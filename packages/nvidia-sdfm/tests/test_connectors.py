from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

import nvidia_sdfm.core.connectors as connectors_module
from nvidia_sdfm.core.connectors import read
from nvidia_sdfm.errors import MissingExtraError, SdfmError


def test_read_unknown_connector_raises():
    with pytest.raises(SdfmError) as excinfo:
        read('mysql')
    assert excinfo.value.code == 'UNKNOWN_CONNECTOR'


def test_missing_backend_maps_to_missing_extra_error(monkeypatch):
    from sdfm_connectors.sql import MissingBackendError

    def raise_missing(source, **kwargs):
        raise MissingBackendError('snowflake', 'snowflake-connector-python')

    monkeypatch.setattr(connectors_module, '_read', raise_missing)
    with pytest.raises(MissingExtraError) as excinfo:
        read('snowflake', table='t')
    assert excinfo.value.details['extra'] == 'snowflake'


def test_broken_driver_is_not_reported_as_missing_extra(monkeypatch):
    def raise_broken(source, **kwargs):
        raise ImportError('native ABI mismatch in installed driver')

    monkeypatch.setattr(connectors_module, '_read', raise_broken)
    with pytest.raises(SdfmError) as excinfo:
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

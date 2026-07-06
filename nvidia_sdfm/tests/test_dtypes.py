from __future__ import annotations

import pandas as pd
import pytest

from nvidia_sdfm.core.dtypes import (
    JSON_SAFE_INT_MAX,
    infer_tfm_dtype,
    serialize_cell,
    serialize_column,
)


@pytest.mark.parametrize(('values', 'expected'), [
    ([1, 2, 3], 'int64'),
    ([1.0, 2.0], 'float64'),
    ([True, False], 'bool'),
    (['a', 'b'], 'string'),
])
def test_infer_tfm_dtype_basic(values, expected):
    assert infer_tfm_dtype(pd.Series(values)) == expected


def test_infer_tfm_dtype_int32():
    series = pd.Series([1, 2, 3], dtype='int32')
    assert infer_tfm_dtype(series) == 'int32'


def test_infer_tfm_dtype_timestamp():
    series = pd.to_datetime(pd.Series(['2025-01-01T00:00:00Z']))
    assert infer_tfm_dtype(series) == 'timestamp[us]'


def test_serialize_cell_none_and_nan_become_null():
    assert serialize_cell(None, 'string') is None
    assert serialize_cell(float('nan'), 'float64') is None


def test_serialize_cell_int_within_safe_range_stays_int():
    assert serialize_cell(42, 'int64') == 42


def test_serialize_cell_int_outside_safe_range_becomes_string():
    huge = JSON_SAFE_INT_MAX + 100
    assert serialize_cell(huge, 'int64') == str(huge)


def test_serialize_cell_timestamp_is_rfc3339_with_z_suffix():
    value = serialize_cell(pd.Timestamp('2025-01-01T00:00:00'), 'timestamp[us]')
    assert value == '2025-01-01T00:00:00.000000Z'


def test_serialize_cell_timestamp_preserves_microseconds():
    value = serialize_cell(
        pd.Timestamp('2025-01-01T00:00:00.123456'), 'timestamp[us]',
    )
    assert value == '2025-01-01T00:00:00.123456Z'


def test_serialize_cell_timestamp_converts_offset_to_utc():
    value = serialize_cell(
        pd.Timestamp('2025-01-01T05:30:00+05:30'), 'timestamp[us]',
    )
    assert value == '2025-01-01T00:00:00.000000Z'


def test_serialize_column_preserves_order():
    series = pd.Series([3, 1, 2])
    assert serialize_column(series, 'int64') == [3, 1, 2]

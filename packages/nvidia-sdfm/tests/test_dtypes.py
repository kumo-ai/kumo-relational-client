# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nvidia_sdfm.core.dtypes import (
    JSON_SAFE_INT_MAX,
    infer_tfm_dtype,
    serialize_cell,
    serialize_column,
    widen_tfm_dtype,
)
from nvidia_sdfm.errors import SdfmError


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


def test_serialize_cell_treats_ndarray_as_a_value_not_a_null():
    # Regression: bugs/tabicl-request-builder-crashes-on-ndarray-and-duplicate-
    # columns.md -- ``bool(pd.isna(array))`` used to raise ValueError here.
    assert serialize_cell(np.array([1, 2]), 'string') == '[1 2]'
    assert serialize_cell(np.array([]), 'string') == '[]'
    assert serialize_cell(np.array([float('nan')]), 'string') == '[nan]'


def test_serialize_cell_treats_list_like_values_as_values():
    assert serialize_cell([1, 2], 'string') == '[1, 2]'
    assert serialize_cell((1, 2), 'string') == '(1, 2)'


@pytest.mark.parametrize(('candidates', 'expected'), [
    (['int64'], 'int64'),
    (['int64', 'float64'], 'float64'),
    (['int64', 'float32'], 'float64'),
    (['int32', 'float32'], 'float32'),
    (['bool', 'int32'], 'int32'),
    (['int64', 'string'], 'string'),
    (['timestamp[us]', 'int64'], 'string'),
])
def test_widen_tfm_dtype(candidates, expected):
    # Regression: bugs/tabicl-predict-frame-int-truncation.md
    assert widen_tfm_dtype(candidates) == expected


def test_serialize_cell_rejects_fractional_value_under_an_int_dtype():
    # Regression: bugs/tabicl-predict-frame-int-truncation.md -- ``int(3.7)``
    # used to silently truncate to ``3``.
    with pytest.raises(SdfmError) as err:
        serialize_cell(3.7, 'int64')
    assert err.value.code == 'INVALID_REQUEST'
    assert serialize_cell(3.0, 'int64') == 3


def test_serialize_cell_rejects_non_finite_numbers():
    # Regression: bugs/rfm-nonfinite-and-decimal-cells-raise-bare-json-errors.md
    for dtype in ('float64', 'int64'):
        with pytest.raises(SdfmError) as err:
            serialize_cell(float('inf'), dtype)
        assert err.value.code == 'INVALID_REQUEST'
    with pytest.raises(SdfmError):
        serialize_cell(float('-inf'), 'float64')

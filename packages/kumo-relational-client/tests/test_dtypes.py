# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from kumo_relational_client.errors import RelationalError
from kumo_relational_client.wire.dtypes import (
    JSON_SAFE_INT_MAX,
    UNTYPED,
    infer_tfm_dtype,
    serialize_cell,
    serialize_column,
    widen_tfm_dtype,
)


@pytest.mark.parametrize(
    ('values', 'expected'),
    [
        ([1, 2, 3], 'int64'),
        ([1.0, 2.0], 'float64'),
        ([True, False], 'bool'),
        (['a', 'b'], 'string'),
    ],
)
def test_infer_tfm_dtype_basic(values, expected):
    assert infer_tfm_dtype(pd.Series(values)) == expected


@pytest.mark.parametrize('width', ['int8', 'int16', 'int32', 'int64'])
def test_every_integer_width_is_declared_int64(width: str) -> None:
    r"""One rule for both model paths.

    They used to disagree: one reported the true width, the other always
    ``int64``, so identical data was described two ways. The server decodes
    both through ``int()`` and cannot tell them apart, but it does require a
    task's target dtype to equal the schema dtype for that column exactly, so
    a second rule anywhere is a 422 rather than a cosmetic difference.
    """
    assert infer_tfm_dtype(pd.Series([1, 2, 3], dtype=width)) == 'int64'


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
        pd.Timestamp('2025-01-01T00:00:00.123456'),
        'timestamp[us]',
    )
    assert value == '2025-01-01T00:00:00.123456Z'


def test_serialize_cell_timestamp_converts_offset_to_utc():
    value = serialize_cell(
        pd.Timestamp('2025-01-01T05:30:00+05:30'),
        'timestamp[us]',
    )
    assert value == '2025-01-01T00:00:00.000000Z'


def test_serialize_column_preserves_order():
    series = pd.Series([3, 1, 2])
    assert serialize_column(series, 'int64') == [3, 1, 2]


def test_serialize_cell_treats_ndarray_as_a_value_not_a_null():
    # Regression: ``bool(pd.isna(array))`` used to raise ValueError here.
    assert serialize_cell(np.array([1, 2]), 'string') == '[1 2]'
    assert serialize_cell(np.array([]), 'string') == '[]'
    assert serialize_cell(np.array([float('nan')]), 'string') == '[nan]'


def test_serialize_cell_treats_list_like_values_as_values():
    assert serialize_cell([1, 2], 'string') == '[1, 2]'
    assert serialize_cell((1, 2), 'string') == '(1, 2)'


@pytest.mark.parametrize(
    ('candidates', 'expected'),
    [
        (['int64'], 'int64'),
        (['int64', 'float64'], 'float64'),
        (['int64', 'float32'], 'float64'),
        (['int32', 'float32'], 'float32'),
        (['bool', 'int32'], 'int32'),
        (['int64', 'string'], 'string'),
        (['timestamp[us]', 'int64'], 'string'),
    ],
)
def test_widen_tfm_dtype(candidates, expected):
    # Regression.
    assert widen_tfm_dtype(candidates) == expected


def test_serialize_cell_rejects_fractional_value_under_an_int_dtype():
    # Regression: ``int(3.7)`` used to silently truncate to ``3``.
    with pytest.raises(RelationalError) as err:
        serialize_cell(3.7, 'int64')
    assert err.value.code == 'INVALID_REQUEST'
    assert serialize_cell(3.0, 'int64') == 3


def test_serialize_cell_rejects_non_finite_numbers():
    # Regression.
    for dtype in ('float64', 'int64'):
        with pytest.raises(RelationalError) as err:
            serialize_cell(float('inf'), dtype)
        assert err.value.code == 'INVALID_REQUEST'
    with pytest.raises(RelationalError):
        serialize_cell(float('-inf'), 'float64')


@pytest.mark.parametrize(
    ('values', 'expected'),
    [
        (pd.Series([1, 2, 3], dtype=object), 'int64'),
        (pd.Series([1, None, 3], dtype=object), 'int64'),
        (pd.Series([1.5, 2.5], dtype=object), 'float64'),
        (pd.Series([True, False], dtype=object), 'bool'),
        (pd.Series([Decimal('1.5')], dtype=object), 'float64'),
        (pd.Series([datetime.date(2025, 1, 1)], dtype=object), 'timestamp[us]'),
        (pd.Series(['a', 'b'], dtype=object), 'string'),
        (pd.Series([1, 'a'], dtype=object), 'string'),
    ],
)
def test_infer_tfm_dtype_classifies_object_columns_by_content(values, expected):
    # Regression: every ``object`` column was reported as ``'string'``
    # regardless of what it held, so a numeric column in the other frame was
    # re-typed as strings.
    assert infer_tfm_dtype(values) == expected


@pytest.mark.parametrize(
    'values',
    [
        pd.Series([None, None], dtype=object),
        pd.Series([np.nan, np.nan], dtype=object),
        pd.Series([], dtype=object),
    ],
)
def test_infer_tfm_dtype_reports_an_all_null_object_column_as_untyped(values):
    # Regression: ``[None, None]`` carries no type information, and used to
    # widen the other frame's real values to strings.
    assert infer_tfm_dtype(values) == UNTYPED


@pytest.mark.parametrize(
    ('candidates', 'expected'),
    [
        ([UNTYPED, 'int64'], 'int64'),
        (['int64', UNTYPED], 'int64'),
        ([UNTYPED, 'float64'], 'float64'),
        ([UNTYPED, 'string'], 'string'),
        ([UNTYPED, UNTYPED], 'string'),
    ],
)
def test_widen_tfm_dtype_ignores_untyped_candidates(candidates, expected):
    # Regression.
    assert widen_tfm_dtype(candidates) == expected


def test_serialize_cell_refuses_an_integer_a_float_cannot_hold_exactly():
    # Regression: ``widen(['int64', 'float64']) == 'float64'`` had no
    # equivalent of the JS-safe-int guard its ``int64`` branch applies, so past
    # 2**53 distinct ids merged into one value with nothing logged.
    with pytest.raises(RelationalError) as err:
        serialize_cell(9007199254740993, 'float64')
    assert err.value.code == 'INVALID_REQUEST'
    assert serialize_cell(9007199254740992, 'float64') == 9007199254740992.0
    assert serialize_cell(2**53 - 1, 'float64') == float(2**53 - 1)
    assert serialize_cell(1.5, 'float64') == 1.5
    assert serialize_cell(1e300, 'float64') == 1e300

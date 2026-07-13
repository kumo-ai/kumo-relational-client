from __future__ import annotations

from typing import Any

import pandas as pd

JSON_SAFE_INT_MAX = 9007199254740991
JSON_SAFE_INT_MIN = -9007199254740991


def infer_tfm_dtype(series: pd.Series) -> str:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return 'bool'
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return 'timestamp[us]'
    if pd.api.types.is_integer_dtype(dtype):
        return 'int64' if dtype.itemsize > 4 else 'int32'
    if pd.api.types.is_float_dtype(dtype):
        return 'float64' if dtype.itemsize > 4 else 'float32'
    return 'string'


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, dict)):
        return False
    return bool(pd.isna(value))


def serialize_cell(value: Any, dtype: str) -> Any:
    if _is_missing(value):
        return None
    if dtype == 'timestamp[us]':
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize('UTC')
        else:
            timestamp = timestamp.tz_convert('UTC')
        return timestamp.isoformat(timespec='microseconds').replace('+00:00', 'Z')
    if dtype in ('int64', 'int32'):
        integer_value = int(value)
        if integer_value > JSON_SAFE_INT_MAX or integer_value < JSON_SAFE_INT_MIN:
            return str(integer_value)
        return integer_value
    if dtype in ('float64', 'float32'):
        return float(value)
    if dtype == 'bool':
        return bool(value)
    return str(value)


def serialize_column(series: pd.Series, dtype: str) -> list[Any]:
    return [serialize_cell(value, dtype) for value in series.tolist()]

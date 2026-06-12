import warnings
from typing import Any, cast

import pandas as pd
import pyarrow as pa


def Timestamp(*args: Any, **kwargs: Any) -> pd.Timestamp:
    r"""Constructs a :class:`pandas.Timestamp`, raising on ``NaT``."""
    result = pd.Timestamp(*args, **kwargs)
    if result is pd.NaT:
        raise TypeError("Expected pd.Timestamp, got NaT")
    return cast(pd.Timestamp, result)


def Timedelta(*args: Any, **kwargs: Any) -> pd.Timedelta:
    r"""Constructs a :class:`pandas.Timedelta`, raising on ``NaT``."""
    result = pd.Timedelta(*args, **kwargs)
    if result is pd.NaT:
        raise TypeError("Expected pd.Timedelta, got NaT")
    return cast(pd.Timedelta, result)


def is_datetime(ser: pd.Series) -> bool:
    r"""Check whether a :class:`pandas.Series` holds datetime values."""
    if isinstance(ser.dtype, pd.ArrowDtype):
        dtype = ser.dtype.pyarrow_dtype
        return (pa.types.is_timestamp(dtype) or pa.types.is_date(dtype)
                or pa.types.is_time(dtype))

    return pd.api.types.is_datetime64_any_dtype(ser)


def to_datetime(ser: pd.Series) -> pd.Series:
    """Converts a :class:`pandas.Series` to ``datetime64[ns]`` format."""
    if isinstance(ser.dtype, pd.ArrowDtype):
        ser = pd.Series(ser.to_numpy(), index=ser.index, name=ser.name)

    if not pd.api.types.is_datetime64_any_dtype(ser):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='Could not infer format',
            )
            ser = pd.to_datetime(ser, unit='ns', errors='coerce')

    if isinstance(ser.dtype, pd.DatetimeTZDtype):
        ser = ser.dt.tz_localize(None)

    if ser.dtype != 'datetime64[ns]':
        ser = ser.astype('datetime64[ns]')

    return ser

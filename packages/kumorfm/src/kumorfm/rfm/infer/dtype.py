import numpy as np
import pandas as pd
import pyarrow as pa
from kumoapi.typing import Dtype

from kumorfm.rfm.base.utils import is_datetime

PANDAS_TO_DTYPE: dict[str, Dtype] = {
    'bool': Dtype.bool,
    'boolean': Dtype.bool,
    'uint8': Dtype.int,
    'int8': Dtype.int,
    'int16': Dtype.int,
    'int32': Dtype.int,
    'int64': Dtype.int,
    'float': Dtype.float,
    'double': Dtype.float,
    'float16': Dtype.float,
    'float32': Dtype.float,
    'float64': Dtype.float,
    'object': Dtype.string,
    'str': Dtype.string,
    'string': Dtype.string,
    'string[python]': Dtype.string,
    'string[pyarrow]': Dtype.string,
    'binary': Dtype.binary,
    'binary[python]': Dtype.binary,
    'binary[pyarrow]': Dtype.binary,
}


def infer_dtype(ser: pd.Series) -> Dtype:
    """Extracts the :class:`Dtype` from a :class:`pandas.Series`.

    Args:
        ser: A :class:`pandas.Series` to analyze.

    Returns:
        The data type.
    """
    if is_datetime(ser):
        return Dtype.date
    if pd.api.types.is_timedelta64_dtype(ser.dtype):
        return Dtype.timedelta
    if isinstance(ser.dtype, pd.CategoricalDtype):
        return Dtype.string

    if (pd.api.types.is_object_dtype(ser.dtype)
            and not isinstance(ser.dtype, pd.ArrowDtype)):
        index = ser.iloc[:1000].first_valid_index()
        if index is not None and pd.api.types.is_list_like(ser[index]):
            pos = ser.index.get_loc(index)
            assert isinstance(pos, int)
            ser = ser.iloc[pos:pos + 1000].dropna()
            arr = pa.array(ser.tolist())
            ser = pd.Series(arr, dtype=pd.ArrowDtype(arr.type))

    if isinstance(ser.dtype, pd.ArrowDtype):
        if (pa.types.is_list(ser.dtype.pyarrow_dtype)
                or pa.types.is_fixed_size_list(ser.dtype.pyarrow_dtype)):
            elem_dtype = ser.dtype.pyarrow_dtype.value_type
            if pa.types.is_integer(elem_dtype):
                return Dtype.intlist
            if pa.types.is_floating(elem_dtype):
                return Dtype.floatlist
            if pa.types.is_decimal(elem_dtype):
                return Dtype.floatlist
            if pa.types.is_string(elem_dtype):
                return Dtype.stringlist
            if pa.types.is_null(elem_dtype):
                return Dtype.floatlist

    if isinstance(ser.dtype, np.dtype):
        dtype_str = str(ser.dtype).lower()
    elif isinstance(ser.dtype, pd.api.extensions.ExtensionDtype):
        dtype_str = ser.dtype.name.lower()
        dtype_str = dtype_str.split('[')[0]  # Remove backend metadata
    elif isinstance(ser.dtype, pa.DataType):
        dtype_str = str(ser.dtype).lower()
    else:
        dtype_str = 'object'

    if dtype_str not in PANDAS_TO_DTYPE:
        raise ValueError(f"Unsupported data type '{ser.dtype}'")

    return PANDAS_TO_DTYPE[dtype_str]

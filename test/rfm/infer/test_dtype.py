import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from kumoapi.typing import Dtype

from kumoai.rfm.infer import infer_dtype


def test_infer_dtype_all_nan() -> None:
    ser = pd.Series([None, None, None, None])
    assert infer_dtype(ser) == Dtype.string


def test_infer_dtype_stringlist() -> None:
    arr = pa.array([[1, 2, 3], [4, 5], None])
    ser = pd.Series(arr)
    assert infer_dtype(ser) == Dtype.intlist
    ser = pd.Series(arr, dtype=pd.ArrowDtype(arr.type))
    assert infer_dtype(ser) == Dtype.intlist

    arr = pa.array([[1.0, 2, 3], [4, 5], None])
    ser = pd.Series(arr)
    assert infer_dtype(ser) == Dtype.floatlist
    ser = pd.Series(arr, dtype=pd.ArrowDtype(arr.type))
    assert infer_dtype(ser) == Dtype.floatlist

    arr = pa.array([['A', 'B', 'C'], ['D', 'E'], None])
    ser = pd.Series(arr)
    assert infer_dtype(ser) == Dtype.stringlist
    ser = pd.Series(arr, dtype=pd.ArrowDtype(arr.type))
    assert infer_dtype(ser) == Dtype.stringlist

    ser = pd.Series([None, [1, 2], [2, 1]])
    assert infer_dtype(ser) == Dtype.intlist

    ser = pd.Series([None, np.array([1, 2]), np.array([2, 1])])
    assert infer_dtype(ser) == Dtype.intlist

    ser = pd.Series([None, [1.0, 2.0], [2, 1]])
    assert infer_dtype(ser) == Dtype.floatlist

    ser = pd.Series([None, np.array([1.0, 2.0]), np.array([2, 1])])
    assert infer_dtype(ser) == Dtype.floatlist

    ser = pd.Series([None, ['A', 'B'], ['B', 'A']])
    assert infer_dtype(ser) == Dtype.stringlist

    ser = pd.Series([None, np.array(['A', 'B']), np.array(['B', 'A'])])
    assert infer_dtype(ser) == Dtype.stringlist

    ser = pd.Series([None, [], []])
    assert infer_dtype(ser) == Dtype.floatlist

    ser = pd.Series([None, np.array([]), np.array([])])
    assert infer_dtype(ser) == Dtype.floatlist

    ser = pd.Series([None, [], [1, 2]])
    assert infer_dtype(ser) == Dtype.intlist

    ser = pd.Series([None, np.array([]), np.array([1, 2])])
    assert infer_dtype(ser) == Dtype.floatlist

    ser = pd.Series([None, [1, None]])
    assert infer_dtype(ser) == Dtype.intlist

    ser = pd.Series([None, np.array([1, None])])
    assert infer_dtype(ser) == Dtype.intlist

    ser = pd.Series([None, [1.0, None]])
    assert infer_dtype(ser) == Dtype.floatlist

    ser = pd.Series([None, np.array([1.0, None])])
    assert infer_dtype(ser) == Dtype.floatlist

    with pytest.raises(pa.lib.ArrowTypeError, match="Expected bytes"):
        ser = pd.Series([None, ['A', 1]])
        infer_dtype(ser)

    ser = pd.Series([None, np.array(['A', 1])])
    assert infer_dtype(ser) == Dtype.stringlist

    with pytest.raises(ValueError, match=r"Unsupported data type"):
        ser = pd.Series([None, [True, False], [False, True]])
        infer_dtype(ser)

    with pytest.raises(ValueError, match=r"Unsupported data type"):
        ser = pd.Series([None, np.array([True, False])])
        infer_dtype(ser)

    with pytest.raises(ValueError, match="cannot mix list and non-list"):
        infer_dtype(pd.Series([None, ['A'], 1]))

import numpy as np
import pandas as pd
import pytest
from kumoapi.typing import Dtype

from kumoai.rfm.base.mapper import Mapper


@pytest.mark.parametrize('dtype', [Dtype.int, Dtype.float, Dtype.string])
def test_mapper(dtype: Dtype) -> None:
    if dtype == Dtype.int:
        A = 2
        B = 1
        C = 0
    elif dtype == Dtype.float:
        A = 1.5
        B = 3.7
        C = 0.3
    else:
        assert dtype == Dtype.string
        A = 'A'
        B = 'B'
        C = 'C'

    mapper = Mapper(num_examples=2)
    out = mapper.get(
        pkey=pd.Series([A, B, C, A, B, C]),
        batch=np.array([1, 1, 1, 0, 0, 0]),
    )
    assert np.array_equal(out, np.array([-1, -1, -1, -1, -1, -1]))

    mapper.add(
        pkey=pd.Series([A, A, B, B]),
        batch=np.array([0, 0, 0, 1]),
    )
    out = mapper.get(
        pkey=pd.Series([A, B, C, A, B, C]),
        batch=np.array([1, 1, 1, 0, 0, 0]),
    )
    assert np.array_equal(out, np.array([-1, 2, -1, 0, 1, -1]))

    mapper.add(
        pkey=pd.Series([A, B]),
        batch=np.array([1, 1]),
    )
    out = mapper.get(
        pkey=pd.Series([A, B, C, A, B, C]),
        batch=np.array([1, 1, 1, 0, 0, 0]),
    )
    assert np.array_equal(out, np.array([3, 2, -1, 0, 1, -1]))

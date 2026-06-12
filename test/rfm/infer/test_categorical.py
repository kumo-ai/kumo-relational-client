import pandas as pd
from kumoapi.typing import Dtype

from kumoai.rfm.infer import contains_categorical


def test_contains_categorical() -> None:
    assert not contains_categorical(pd.Series([1]), 'test', Dtype.date)
    assert contains_categorical(pd.Series([1]), 'test', Dtype.bool)
    assert contains_categorical(
        ser=pd.Series(['A', 'B', 'C']),
        column_name='test',
        dtype=Dtype.string,
    )
    assert contains_categorical(
        ser=pd.Series(list(range(20)) * 20),
        column_name='test',
        dtype=Dtype.int,
    )
    assert not contains_categorical(
        ser=pd.Series(list(range(20)) * 19),
        column_name='test',
        dtype=Dtype.int,
    )

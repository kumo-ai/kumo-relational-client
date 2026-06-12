import pandas as pd
from kumoapi.typing import Dtype

from kumoai.rfm.infer import contains_timestamp


def test_contains_timestamp() -> None:
    invalid_ser = pd.Series(['A', 'B', 'C'])
    valid_ser = pd.Series(['1990', '1991', '1992'])

    assert contains_timestamp(invalid_ser, 'test', dtype=Dtype.date)
    assert contains_timestamp(invalid_ser, 'test', dtype=Dtype.time)
    assert not contains_timestamp(invalid_ser, 'datetime', dtype=Dtype.bool)
    assert not contains_timestamp(invalid_ser, 'datetime', dtype=Dtype.string)
    assert contains_timestamp(valid_ser, 'test', dtype=Dtype.string)
    assert not contains_timestamp(invalid_ser, 'test', dtype=Dtype.string)

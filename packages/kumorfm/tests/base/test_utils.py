import pandas as pd
import pytest

from kumorfm.rfm.base.utils import Timestamp


def test_timestamp() -> None:
    assert Timestamp('2026-01-01') == pd.Timestamp('2026-01-01')

    with pytest.raises(TypeError, match='got NaT'):
        Timestamp(None)


def test_timestamp_normalizes_timezone() -> None:
    out = Timestamp('2026-01-01 12:00:00+00:00')
    assert out.tzinfo is None
    assert out == pd.Timestamp('2026-01-01 12:00:00')

    out = Timestamp(pd.Timestamp('2026-01-01 12:00:00+05:00'))
    assert out.tzinfo is None
    assert out == pd.Timestamp('2026-01-01 07:00:00')

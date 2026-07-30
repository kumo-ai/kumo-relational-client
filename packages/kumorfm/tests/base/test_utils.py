# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta, timezone

import pandas as pd
import pytest

from kumorfm.rfm.base.utils import Timestamp, to_datetime


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


def test_to_datetime_normalizes_timezone() -> None:
    ser = pd.Series(pd.to_datetime(['2026-01-01 12:00:00']))
    assert to_datetime(ser).iloc[0] == pd.Timestamp('2026-01-01 12:00:00')

    ser = pd.Series(pd.to_datetime(['2026-01-01 12:00:00+05:00']))
    out = to_datetime(ser)
    assert out.dtype == 'datetime64[ns]'
    assert out.iloc[0] == pd.Timestamp('2026-01-01 07:00:00')
    assert out.iloc[0] == Timestamp(ser.iloc[0])

    ser = pd.Series(pd.to_datetime(['2026-01-01 17:00:00'
                                    ])).dt.tz_localize(timezone(timedelta(
                                        hours=-8)))
    assert to_datetime(ser).iloc[0] == pd.Timestamp('2026-01-02 01:00:00')

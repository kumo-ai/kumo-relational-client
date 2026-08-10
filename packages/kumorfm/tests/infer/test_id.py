# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from kumorfm.api.typing import Dtype
from kumorfm.rfm.infer import contains_id


def test_contains_id() -> None:
    ser = pd.Series([1, 2, 3])

    assert not contains_id(ser, column_name='user_id', dtype=Dtype.bool)
    assert not contains_id(ser, column_name='test', dtype=Dtype.int)
    assert not contains_id(ser, column_name='liquid', dtype=Dtype.int)
    assert contains_id(ser, column_name='__id__', dtype=Dtype.int)
    assert contains_id(ser, column_name='color_code', dtype=Dtype.int)
    assert contains_id(ser, column_name='userid', dtype=Dtype.int)

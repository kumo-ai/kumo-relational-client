# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The shared ``arrays`` table encoder.

``encode_table`` came out of the TabICL adapter to be shared with the
relational path, so what matters is that it still writes exactly what the
adapter wrote. The cases here are the ones where a rewrite could plausibly
differ and the server would still accept the result: the empty-column frame,
the row/column transpose, and the JS-safe-integer boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nemotron_predict.wire import (
    JSON_SAFE_INT_MAX,
    encode_table,
    infer_tfm_dtype,
)


def encode(frame: pd.DataFrame) -> dict:
    dtypes = {c: infer_tfm_dtype(frame[c]) for c in frame.columns}
    return encode_table(frame, dtypes)


def test_shape_is_columns_plus_row_arrays() -> None:
    frame = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    assert encode(frame) == {
        'format': 'arrays',
        'columns': ['a', 'b'],
        'rows': [[1, 'x'], [2, 'y']],
    }


def test_column_order_follows_the_frame_not_sorting() -> None:
    r"""Rows are positional against ``columns``, so a reordering would
    silently move every value into the wrong field.
    """
    frame = pd.DataFrame({'z': [1], 'a': [2], 'm': [3]})
    encoded = encode(frame)
    assert encoded['columns'] == ['z', 'a', 'm']
    assert encoded['rows'] == [[1, 2, 3]]


def test_a_frame_with_no_columns_keeps_its_row_count() -> None:
    r"""An empty-column table and an empty table are different requests."""
    frame = pd.DataFrame(index=range(3))
    assert encode(frame) == {
        'format': 'arrays',
        'columns': [],
        'rows': [[], [], []],
    }


def test_no_rows_is_not_the_same_as_no_columns() -> None:
    frame = pd.DataFrame({'a': pd.Series([], dtype='int64')})
    encoded = encode(frame)
    assert encoded['columns'] == ['a']
    assert encoded['rows'] == []


def test_integers_past_the_js_safe_range_go_out_as_strings() -> None:
    r"""The recurring interoperability rule of this contract: a JSON number
    above 2**53-1 loses precision in a JS reader, so it is sent base-10.
    """
    frame = pd.DataFrame({'big': [JSON_SAFE_INT_MAX + 2]})
    value = encode(frame)['rows'][0][0]
    assert value == str(JSON_SAFE_INT_MAX + 2)
    assert isinstance(value, str)


def test_safe_integers_stay_numbers() -> None:
    frame = pd.DataFrame({'small': [JSON_SAFE_INT_MAX]})
    assert encode(frame)['rows'][0][0] == JSON_SAFE_INT_MAX


def test_missing_values_become_null() -> None:
    frame = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
    assert encode(frame)['rows'] == [[1.0], [None], [3.0]]


def test_the_adapter_and_the_shared_encoder_agree() -> None:
    r"""The move this module came from: the TabICL payload builder must be
    emitting exactly what ``encode_table`` produces.
    """
    from nemotron_predict.adapters.tabicl import build_request

    context = pd.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [0, 1, 0]})
    predict = pd.DataFrame({'x': [4.0]})
    body = build_request(
        context=context,
        predict=predict,
        task='binary_classification',
        target='y',
        outputs=['prediction'],
        request_id='fixed',
    )

    assert body['context']['instance_table'] == encode(context)
    assert body['predict']['instance_table'] == encode(predict)

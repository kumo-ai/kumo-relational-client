# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from kumorfm.rfm.infer.pkey import infer_primary_key


def test_ambiguous_primary_key_warning_names_the_tied_candidates() -> None:
    r"""graph-ambiguous-primary-key-warning-lists-nothing.md

    ``max_score`` used to be bound to a ``(name, score)`` tuple and compared
    against a float, so the list of tied candidates was always empty and the
    message told the user to choose without saying from what.
    """
    df = pd.DataFrame({
        'user_id': [1, 2, 3, 4],
        'user_key': [9, 8, 7, 6],
        'age': [10, 20, 30, 40],
    })

    with pytest.warns(UserWarning,
                      match='multiple potential primary keys') as record:
        assert infer_primary_key('users', df, ['user_id', 'user_key']) is None

    message = str(record[0].message)
    assert "'user_id'" in message
    assert "'user_key'" in message


def test_unambiguous_primary_key_is_returned_without_warning() -> None:
    df = pd.DataFrame({'user_id': [1, 2, 3], 'age': [10, 20, 30]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert infer_primary_key('users', df, ['user_id']) == 'user_id'

    assert caught == []

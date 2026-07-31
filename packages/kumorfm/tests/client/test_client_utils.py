# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest

from kumorfm.client.utils import parse_response


class _Response:
    def __init__(self, payload, text=''):
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


@pytest.mark.parametrize('value, expected', [
    (True, True),
    (False, False),
    ('true', True),
    ('True', True),
    ('false', False),
    ('False', False),
])
def test_bool_list_parsing_is_not_shadowed_by_the_int_cast(value, expected):
    # `bool` is a subclass of `int`, so an `issubclass(data_class, (str, int))`
    # check placed first claims it and `bool('false')` is True.
    response = _Response([value])
    assert parse_response(List[bool], response) == [expected]


@pytest.mark.parametrize('value, expected', [
    ('true', True),
    ('false', False),
])
def test_scalar_bool_response_parsing(value, expected):
    response = _Response(None, text=value)
    assert parse_response(bool, response) is expected


def test_int_and_str_lists_still_cast():
    assert parse_response(List[int], _Response(['3', 4])) == [3, 4]
    assert parse_response(List[str], _Response([5, 'x'])) == ['5', 'x']


def test_parse_response_passes_text_through_for_str():
    assert parse_response(str, _Response(None, text='hello')) == 'hello'

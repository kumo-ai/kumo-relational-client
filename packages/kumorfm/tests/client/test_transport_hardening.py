# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Guards on what the RFM transport accepts back from a NIM.

Both cases here are about a response the SDK does not control: an error body
large enough to be unreadable, and a session id chosen by the server.
"""
from __future__ import annotations

from typing import Any

import pytest

from kumorfm.client.client import _MAX_BODY_SNIPPET, capped_body
from kumorfm.client.endpoints import Endpoint
from kumorfm.client.rfm import RFMAPI, _path_segment
from kumorfm.client.utils import raise_on_error
from kumorfm.exceptions import HTTPException


class _Response:
    r"""The four members ``raise_on_error`` reads."""

    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text
        self.ok = status_code < 400

    def json(self) -> Any:
        raise ValueError('not json')


class _RecordingClient:
    r"""Captures the path RFMAPI builds, without sending anything."""

    def __init__(self) -> None:
        self.paths: list[str] = []

    def _request(self, endpoint: Endpoint, **kwargs: Any) -> _Response:
        self.paths.append(endpoint.get_path())
        return _Response(200, '{}')


def test_error_body_is_capped() -> None:
    r"""A proxy can answer with a large HTML page; it must not become the
    whole exception message.
    """
    with pytest.raises(HTTPException) as info:
        raise_on_error(_Response(502, '<html>' + 'x' * 200_000 + '</html>'))

    detail = info.value.detail
    assert len(detail) < _MAX_BODY_SNIPPET + 100
    assert 'truncated' in detail
    assert '200013 chars total' in detail


def test_short_error_body_is_untouched() -> None:
    with pytest.raises(HTTPException) as info:
        raise_on_error(_Response(422, 'categorical cardinality 15001'))
    assert info.value.detail == 'categorical cardinality 15001'


def test_capped_body_collapses_whitespace() -> None:
    assert capped_body('a\n\n  b\tc') == 'a b c'
    assert capped_body(None) == ''
    assert capped_body('') == ''


@pytest.mark.parametrize('session_id,expected', [
    ('sess-123', 'sess-123'),
    ('../v1/predictions', '..%2Fv1%2Fpredictions'),
    ('a/b', 'a%2Fb'),
    ('a?b=c', 'a%3Fb%3Dc'),
    ('a b', 'a%20b'),
])
def test_path_segment_escapes_server_chosen_ids(session_id: str,
                                                expected: str) -> None:
    assert _path_segment(session_id) == expected


def test_session_predict_escapes_the_session_id() -> None:
    r"""The id comes back from the NIM, so a slash in it would otherwise
    redirect the call to a different route.
    """
    client = _RecordingClient()
    api = RFMAPI(client)  # type: ignore[arg-type]
    try:
        api.session_predict('../predictions', {'predict': {}},
                            entity_ids=[1], instance_ids=[0])
    except Exception:  # the fake returns no usable body; the path is the point
        pass
    assert client.paths == ['/v1/sessions/..%2Fpredictions/predictions']


def test_delete_session_escapes_the_session_id() -> None:
    client = _RecordingClient()
    api = RFMAPI(client)  # type: ignore[arg-type]
    api.delete_session('../../etc')
    assert client.paths == ['/v1/sessions/..%2F..%2Fetc']

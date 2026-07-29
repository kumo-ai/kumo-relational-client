# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sdfm_connectors.backends import mark_owned, owns_connection


class _Conn:
    pass


class _SlottedConn:
    r"""Weakref-able but rejects arbitrary attribute writes (like a borrowed
    Snowpark session that only exposes a fixed API)."""
    __slots__ = ('__weakref__',)


def test_unmarked_connection_is_treated_as_owned():
    assert owns_connection(_Conn()) is True


def test_marked_owned_connection_is_owned():
    connection = _Conn()
    mark_owned(connection, True)
    assert owns_connection(connection) is True


def test_borrowed_connection_is_not_owned():
    connection = _Conn()
    mark_owned(connection, False)
    assert owns_connection(connection) is False


def test_borrowed_connection_rejecting_attribute_writes_is_not_owned():
    connection = _SlottedConn()
    mark_owned(connection, False)
    assert owns_connection(connection) is False

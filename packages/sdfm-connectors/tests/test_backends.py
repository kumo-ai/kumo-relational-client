# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sdfm_connectors import connect
from sdfm_connectors.backends import mark_owned, owns_connection
from sdfm_connectors.sql import ConnectorError, check_connect_args


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


def test_check_connect_args_suggests_the_intended_name():
    r"""connectors-unknown-connect-kwargs-silently-ignored.md"""
    with pytest.raises(ConnectorError) as excinfo:
        check_connect_args('snowflake', {'shema': 'X'}, {'schema', 'database'})
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert "did you mean 'schema'" in excinfo.value.message
    assert excinfo.value.details['arguments'] == ['shema']


def test_check_connect_args_accepts_known_arguments():
    check_connect_args('snowflake', {'schema': 'X'}, {'schema', 'database'})


@pytest.mark.parametrize(
    'backend,module,typo,intended',
    [
        ('snowflake', 'snowflake.connector', 'shema', 'schema'),
        ('databricks', 'databricks.sql', 'catalg', 'catalog'),
    ],
)
def test_connect_rejects_unknown_kwargs(backend, module, typo, intended):
    r"""connectors-unknown-connect-kwargs-silently-ignored.md

    Both drivers accept ``**kwargs`` and silently drop what they do not know,
    so the typo must be rejected before a connection is attempted.
    """
    pytest.importorskip(module)
    with pytest.raises(ConnectorError) as excinfo:
        connect(backend, **{typo: 'X'})
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert f"did you mean {intended!r}" in excinfo.value.message


def test_databricks_driver_options_bypass_the_allow_list(monkeypatch):
    pytest.importorskip('databricks.sql')
    from sdfm_connectors.backends import databricks as backend

    seen = {}
    monkeypatch.setattr(
        backend.databricks_sql, 'connect',
        lambda **kwargs: seen.update(kwargs) or _Conn())
    monkeypatch.setattr(backend.os, 'getenv', lambda name: None)

    backend.connect(catalog='c', driver_options={'use_cloud_fetch': True})

    assert seen == {'catalog': 'c', 'use_cloud_fetch': True}


@pytest.mark.parametrize(
    'backend,module,argument',
    [
        ('snowflake', 'snowflake.connector', 'password'),
        ('databricks', 'databricks.sql', 'catalog'),
    ],
)
def test_driver_options_cannot_restate_a_validated_argument(
        backend, module, argument):
    r"""MR !69 review: the escape hatch must not override a validated argument.

    ``driver_options`` bypasses the allow-list by design, so silently letting
    either side win would let it replace a credential or redirect a read to
    another catalog -- the failure this backend validation exists to prevent.
    """
    pytest.importorskip(module)
    with pytest.raises(ConnectorError) as excinfo:
        connect(backend, driver_options={argument: 'from-options'},
                **{argument: 'from-caller'})
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert excinfo.value.details['arguments'] == [argument]
    assert 'driver_options' in excinfo.value.message


def test_snowflake_driver_options_credentials_suppress_borrowing(monkeypatch):
    r"""MR !69 review: credentials given through the escape hatch still count.

    Anchoring the borrow decision to the top-level arguments alone would
    silently ignore them, which is what the guard exists to prevent.
    """
    pytest.importorskip('snowflake.connector')
    from sdfm_connectors.backends import snowflake as backend

    opened, seen = _Conn(), {}
    monkeypatch.setattr(
        backend, '_active_snowpark_connection', lambda: _Conn())
    monkeypatch.setattr(
        backend.snowflake_connector, 'connect',
        lambda **kwargs: seen.update(kwargs) or opened)

    result = backend.connect(
        driver_options={'account': 'a', 'user': 'u', 'password': 'p'})

    assert result is opened
    assert seen == {'account': 'a', 'user': 'u', 'password': 'p'}


def test_snowflake_borrows_session_when_no_auth_arguments(monkeypatch):
    r"""connectors-snowflake-session-borrowing-all-or-nothing.md"""
    pytest.importorskip('snowflake.connector')
    from sdfm_connectors.backends import snowflake as backend

    borrowed = _Conn()
    monkeypatch.setattr(
        backend, '_active_snowpark_connection', lambda: borrowed)

    assert backend.connect() is borrowed
    assert owns_connection(borrowed) is False


def test_snowflake_rejects_session_arguments_on_a_borrowed_session(monkeypatch):
    r"""connectors-snowflake-session-borrowing-all-or-nothing.md

    Previously this fell through to a credential-less connect that failed with
    the driver's unrelated "User is empty" message.
    """
    pytest.importorskip('snowflake.connector')
    from sdfm_connectors.backends import snowflake as backend

    monkeypatch.setattr(
        backend, '_active_snowpark_connection', lambda: _Conn())

    with pytest.raises(ConnectorError) as excinfo:
        backend.connect(schema='OTHER')
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert 'Snowpark session' in excinfo.value.message
    assert excinfo.value.details['arguments'] == ['schema']


def test_snowflake_credentials_still_open_their_own_connection(monkeypatch):
    pytest.importorskip('snowflake.connector')
    from sdfm_connectors.backends import snowflake as backend

    opened = _Conn()
    monkeypatch.setattr(
        backend, '_active_snowpark_connection', lambda: _Conn())
    monkeypatch.setattr(
        backend.snowflake_connector, 'connect', lambda **kwargs: opened)

    assert backend.connect(account='a', user='u', password='p') is opened
    assert owns_connection(opened) is True


def test_snowflake_no_arguments_and_no_session_explains_requirements(
        monkeypatch):
    r"""connectors-snowflake-session-borrowing-all-or-nothing.md"""
    pytest.importorskip('snowflake.connector')
    from sdfm_connectors.backends import snowflake as backend

    def _fail(**kwargs):
        raise RuntimeError(
            "Default connection with name 'default' cannot be found")

    monkeypatch.setattr(backend, '_active_snowpark_connection', lambda: None)
    monkeypatch.setattr(backend.snowflake_connector, 'connect', _fail)

    with pytest.raises(ConnectorError) as excinfo:
        connect('snowflake')
    assert excinfo.value.code == 'CONNECT_FAILED'
    assert 'Snowpark session' in excinfo.value.message
    assert "'account'" in excinfo.value.message

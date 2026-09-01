# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kumo_connectors import connect
from kumo_connectors._databricks_telemetry import databricks_user_agent
from kumo_connectors._version import __version__ as sdk_version
from kumo_connectors.backends import mark_owned, owns_connection
from kumo_connectors.sql import (
    ConnectorError,
    check_connect_args,
)


class _Conn:
    pass


class _SlottedConn:
    r"""Weakref-able but rejects arbitrary attribute writes (like a borrowed
    Snowpark session that only exposes a fixed API)."""

    __slots__ = ('__weakref__',)


def _clear_databricks_env(monkeypatch, backend):
    for name in backend._ENV_BY_ARG.values():
        monkeypatch.delenv(name, raising=False)


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
    """Both drivers accept ``**kwargs`` and silently drop what they do not know,
    so the typo must be rejected before a connection is attempted.
    """
    pytest.importorskip(module)
    with pytest.raises(ConnectorError) as excinfo:
        connect(backend, **{typo: 'X'})
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert f'did you mean {intended!r}' in excinfo.value.message


def test_databricks_driver_options_bypass_the_allow_list(monkeypatch):
    pytest.importorskip('databricks.sql')
    from kumo_connectors.backends import databricks as backend

    seen = {}
    monkeypatch.setattr(
        backend.databricks_sql,
        'connect',
        lambda **kwargs: seen.update(kwargs) or _Conn(),
    )
    _clear_databricks_env(monkeypatch, backend)

    backend.connect(catalog='c', driver_options={'use_cloud_fetch': True})

    assert seen == {
        'catalog': 'c',
        'use_cloud_fetch': True,
        'user_agent_entry': databricks_user_agent(sdk_version),
    }


def test_databricks_sdk_attribution_preserves_a_caller_entry(monkeypatch):
    pytest.importorskip('databricks.sql')
    from kumo_connectors.backends import databricks as backend

    seen = {}
    monkeypatch.setattr(
        backend.databricks_sql,
        'connect',
        lambda **kwargs: seen.update(kwargs) or _Conn(),
    )
    _clear_databricks_env(monkeypatch, backend)

    backend.connect(user_agent_entry='customer_product/9.9.9')

    assert seen['user_agent_entry'] == databricks_user_agent(
        sdk_version, 'customer_product/9.9.9'
    )


def test_real_databricks_driver_renders_sdk_attribution(monkeypatch):
    pytest.importorskip('databricks.sql')
    from kumo_connectors.backends import databricks as backend

    seen = {}

    class _FakeThriftBackend:
        def __init__(
            self,
            host,
            port,
            http_path,
            http_headers,
            auth_provider,
            **kwargs,
        ):
            seen['headers'] = dict(http_headers)

        def open_session(self, session_configuration, catalog, schema):
            handle = SimpleNamespace(serverProtocolVersion=None)
            return SimpleNamespace(
                sessionHandle=handle, serverProtocolVersion=None
            )

        def handle_to_hex_id(self, handle):
            return '00'

    _clear_databricks_env(monkeypatch, backend)

    kwargs = {
        'server_hostname': 'example.invalid',
        'http_path': '/sql/1.0/warehouses/test',
        'access_token': 'unused-test-token',
        'user_agent_entry': 'customer_product/9.9.9',
    }
    if hasattr(backend._client, 'ThriftBackend'):
        monkeypatch.setattr(
            backend._client, 'ThriftBackend', _FakeThriftBackend
        )
        connection = backend.connect(**kwargs)
        connection.open = False
    else:

        class _HeaderCapturedError(Exception):
            pass

        def _capture_header(session):
            seen['headers'] = {'User-Agent': session.useragent_header}
            raise _HeaderCapturedError

        monkeypatch.setattr(backend._client.Session, 'open', _capture_header)
        monkeypatch.setattr(
            backend._client.TelemetryClientFactory,
            'connection_failure_log',
            lambda **kwargs: None,
        )
        with pytest.raises(_HeaderCapturedError):
            backend.connect(**kwargs)

    header = seen['headers']['User-Agent']
    assert f'nvidia_kumo-relational-client/{sdk_version}' in header
    assert 'customer_product/9.9.9' in header


@pytest.mark.parametrize(
    'entry', ['', '   ', 'product/1\r\nInjected: yes', object()]
)
def test_databricks_rejects_an_invalid_caller_entry(monkeypatch, entry):
    pytest.importorskip('databricks.sql')
    from kumo_connectors.backends import databricks as backend

    _clear_databricks_env(monkeypatch, backend)
    with pytest.raises(ConnectorError) as excinfo:
        backend.connect(user_agent_entry=entry)
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert excinfo.value.details['arguments'] == ['user_agent_entry']


def test_an_invalid_sdk_version_is_not_reported_as_a_connection_failure(
    monkeypatch,
):
    pytest.importorskip('databricks.sql')
    from kumo_connectors.backends import databricks as backend

    monkeypatch.setattr(backend, '__version__', '1.0.0.dev')
    _clear_databricks_env(monkeypatch, backend)
    with pytest.raises(ConnectorError) as excinfo:
        backend.connect()
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert excinfo.value.details['package_version'] == '1.0.0.dev'


@pytest.mark.parametrize(
    'backend,module,argument',
    [
        ('snowflake', 'snowflake.connector', 'password'),
        ('databricks', 'databricks.sql', 'catalog'),
    ],
)
def test_driver_options_cannot_restate_a_validated_argument(
    backend, module, argument
):
    r"""The escape hatch must not override a validated argument.

    ``driver_options`` bypasses the allow-list by design, so silently letting
    either side win would let it replace a credential or redirect a read to
    another catalog -- the failure this backend validation exists to prevent.
    """
    pytest.importorskip(module)
    with pytest.raises(ConnectorError) as excinfo:
        connect(
            backend,
            driver_options={argument: 'from-options'},
            **{argument: 'from-caller'},
        )
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert excinfo.value.details['arguments'] == [argument]
    assert 'driver_options' in excinfo.value.message


def test_snowflake_driver_options_credentials_suppress_borrowing(monkeypatch):
    r"""Credentials given through the escape hatch still count.

    Anchoring the borrow decision to the top-level arguments alone would
    silently ignore them, which is what the guard exists to prevent.
    """
    pytest.importorskip('snowflake.connector')
    from kumo_connectors.backends import snowflake as backend

    opened, seen = _Conn(), {}
    monkeypatch.setattr(backend, '_active_snowpark_connection', lambda: _Conn())
    monkeypatch.setattr(
        backend.snowflake_connector,
        'connect',
        lambda **kwargs: seen.update(kwargs) or opened,
    )

    result = backend.connect(
        driver_options={'account': 'a', 'user': 'u', 'password': 'p'}
    )

    assert result is opened
    assert seen == {'account': 'a', 'user': 'u', 'password': 'p'}


def test_snowflake_borrows_session_when_no_auth_arguments(monkeypatch):
    pytest.importorskip('snowflake.connector')
    from kumo_connectors.backends import snowflake as backend

    borrowed = _Conn()
    monkeypatch.setattr(
        backend, '_active_snowpark_connection', lambda: borrowed
    )

    assert backend.connect() is borrowed
    assert owns_connection(borrowed) is False


def test_snowflake_rejects_session_arguments_on_a_borrowed_session(monkeypatch):
    """Previously this fell through to a credential-less connect that failed with
    the driver's unrelated "User is empty" message.
    """
    pytest.importorskip('snowflake.connector')
    from kumo_connectors.backends import snowflake as backend

    monkeypatch.setattr(backend, '_active_snowpark_connection', lambda: _Conn())

    with pytest.raises(ConnectorError) as excinfo:
        backend.connect(schema='OTHER')
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert 'Snowpark session' in excinfo.value.message
    assert excinfo.value.details['arguments'] == ['schema']


def test_snowflake_credentials_still_open_their_own_connection(monkeypatch):
    pytest.importorskip('snowflake.connector')
    from kumo_connectors.backends import snowflake as backend

    opened = _Conn()
    monkeypatch.setattr(backend, '_active_snowpark_connection', lambda: _Conn())
    monkeypatch.setattr(
        backend.snowflake_connector, 'connect', lambda **kwargs: opened
    )

    assert backend.connect(account='a', user='u', password='p') is opened
    assert owns_connection(opened) is True


def test_snowflake_no_arguments_and_no_session_explains_requirements(
    monkeypatch,
):
    pytest.importorskip('snowflake.connector')
    from kumo_connectors.backends import snowflake as backend

    def _fail(**kwargs):
        raise RuntimeError(
            "Default connection with name 'default' cannot be found"
        )

    monkeypatch.setattr(backend, '_active_snowpark_connection', lambda: None)
    monkeypatch.setattr(backend.snowflake_connector, 'connect', _fail)

    with pytest.raises(ConnectorError) as excinfo:
        connect('snowflake')
    assert excinfo.value.code == 'CONNECT_FAILED'
    assert 'Snowpark session' in excinfo.value.message
    assert "'account'" in excinfo.value.message


@pytest.mark.parametrize(
    'argument', ['connection_name', 'connections_file_path']
)
def test_snowflake_allows_named_connection_arguments(argument):
    """``DEFAULT_CONFIGURATION`` is the driver's config-parameter table and omits
    the two constructor-only parameters that select a named connection from a
    TOML file -- the secrets-file-free auth path. The guard rejected them as
    typos, and ``connection_name`` was already listed in the connector's own
    ``_AUTH_ARGS``, so the allow-list contradicted the auth logic behind it.
    """
    snowflake = pytest.importorskip('kumo_connectors.backends.snowflake')
    check_connect_args('snowflake', {argument: 'x'}, snowflake._CONNECT_ARGS)


def test_snowflake_allow_list_covers_every_auth_argument():
    snowflake = pytest.importorskip('kumo_connectors.backends.snowflake')
    assert snowflake._AUTH_ARGS <= snowflake._CONNECT_ARGS


def test_snowflake_still_rejects_a_misspelled_argument():
    snowflake = pytest.importorskip('kumo_connectors.backends.snowflake')
    with pytest.raises(ConnectorError) as excinfo:
        check_connect_args(
            'snowflake', {'accont': 'x'}, snowflake._CONNECT_ARGS
        )
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert 'account' in str(excinfo.value)


@pytest.mark.parametrize(
    'argument',
    [
        'auth_type',
        'credentials_provider',
        'use_cloud_fetch',
        'user_agent_entry',
    ],
)
def test_databricks_allows_kwargs_routed_arguments(argument):
    """The driver routes its documented OAuth parameters through ``**kwargs``,
    where ``inspect.signature`` cannot see them, so a signature-only
    allow-list rejected the whole non-PAT auth path.
    """
    databricks = pytest.importorskip('kumo_connectors.backends.databricks')
    check_connect_args('databricks', {argument: 'x'}, databricks._CONNECT_ARGS)


def test_databricks_still_rejects_a_misspelled_argument():
    databricks = pytest.importorskip('kumo_connectors.backends.databricks')
    with pytest.raises(ConnectorError) as excinfo:
        check_connect_args(
            'databricks', {'cattalog': 'x'}, databricks._CONNECT_ARGS
        )
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert 'catalog' in str(excinfo.value)

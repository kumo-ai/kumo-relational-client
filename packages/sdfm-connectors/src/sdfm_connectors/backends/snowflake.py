from __future__ import annotations

from typing import Any, TypeAlias

from sdfm_connectors.sql import MissingBackendError

try:
    import snowflake.connector
except ModuleNotFoundError as error:
    if error.name not in ('snowflake', 'snowflake.connector'):
        raise
    raise MissingBackendError('snowflake',
                              'snowflake-connector-python') from error

from sdfm_connectors.backends import mark_owned

Connection: TypeAlias = snowflake.connector.SnowflakeConnection


def _active_snowpark_connection() -> Connection | None:
    try:
        from snowflake.snowpark.context import get_active_session
    except ModuleNotFoundError:
        return None
    try:
        session = get_active_session()
    except Exception:
        return None
    return session.connection


def connect(**kwargs: Any) -> Connection:
    if not kwargs:
        borrowed = _active_snowpark_connection()
        if borrowed is not None:
            mark_owned(borrowed, False)
            return borrowed
    connection = snowflake.connector.connect(**kwargs)
    mark_owned(connection, True)
    return connection

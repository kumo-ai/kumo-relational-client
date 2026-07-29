# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, TypeAlias

from sdfm_connectors.backends import mark_owned
from sdfm_connectors.sql import require_driver

snowflake_connector = require_driver(
    'snowflake',
    'snowflake-connector-python',
    'snowflake.connector',
    absent=('snowflake', 'snowflake.connector'),
)

Connection: TypeAlias = snowflake_connector.SnowflakeConnection


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
    connection = snowflake_connector.connect(**kwargs)
    mark_owned(connection, True)
    return connection

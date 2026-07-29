# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any, TypeAlias

from sdfm_connectors.sql import require_driver

_ABSENT = ('databricks', 'databricks.sql', 'databricks.sql.client')

databricks_sql = require_driver(
    'databricks', 'databricks-sql-connector', 'databricks.sql',
    absent=_ABSENT,
)
_client = require_driver(
    'databricks', 'databricks-sql-connector', 'databricks.sql.client',
    absent=_ABSENT,
)

if not hasattr(_client, 'Connection'):
    raise ImportError(
        "cannot import name 'Connection' from 'databricks.sql.client'")

Connection: TypeAlias = _client.Connection

_ENV_BY_ARG = {
    'server_hostname': 'DATABRICKS_SERVER_HOSTNAME',
    'http_path': 'DATABRICKS_HTTP_PATH',
    'access_token': 'DATABRICKS_TOKEN',
    'catalog': 'DATABRICKS_CATALOG',
    'schema': 'DATABRICKS_SCHEMA',
}


def connect(**kwargs: Any) -> Connection:
    for arg, env in _ENV_BY_ARG.items():
        if kwargs.get(arg) is None:
            kwargs[arg] = os.getenv(env)
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return databricks_sql.connect(**kwargs)

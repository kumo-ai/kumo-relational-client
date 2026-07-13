from __future__ import annotations

import os
from typing import Any, TypeAlias

from sdfm_connectors.sql import MissingBackendError

try:
    from databricks import sql as databricks_sql
    from databricks.sql.client import Connection as _Connection
except ModuleNotFoundError as error:
    if error.name not in ('databricks', 'databricks.sql', 'databricks.sql.client'):
        raise
    raise MissingBackendError('databricks',
                              'databricks-sql-connector') from error

Connection: TypeAlias = _Connection

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

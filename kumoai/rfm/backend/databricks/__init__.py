import os
from typing import Any, TypeAlias

try:
    from databricks import sql as databricks_sql
    from databricks.sql.client import Connection as _Connection
except ImportError as error:
    raise ImportError("No module named 'databricks'. Please install Kumo SDK "
                      "with the 'databricks' extension via "
                      "`pip install kumoai[databricks]`.") from error

Connection: TypeAlias = _Connection

# Connection arguments that fall back to the corresponding environment
# variable when not passed explicitly:
_ENV_BY_ARG = {
    'server_hostname': 'DATABRICKS_SERVER_HOSTNAME',
    'http_path': 'DATABRICKS_HTTP_PATH',
    'access_token': 'DATABRICKS_TOKEN',
    'catalog': 'DATABRICKS_CATALOG',
    'schema': 'DATABRICKS_SCHEMA',
}


def connect(**kwargs: Any) -> Connection:
    r"""Opens a connection to a :class:`databricks` SQL warehouse.

    kwargs: Connection arguments, following the :class:`databricks.sql`
        protocol, *e.g.*, ``server_hostname``, ``http_path`` and
        ``access_token`` (or ``credentials_provider`` for OAuth). Any of
        ``server_hostname``, ``http_path``, ``access_token``, ``catalog`` and
        ``schema`` that are not passed fall back to the corresponding
        ``DATABRICKS_*`` environment variable, so a connection can be opened
        entirely from the environment.
    """
    for arg, env in _ENV_BY_ARG.items():
        if kwargs.get(arg) is None:
            kwargs[arg] = os.getenv(env)
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return databricks_sql.connect(**kwargs)


from .table import DatabricksTable  # noqa: E402
from .sampler import DatabricksSampler  # noqa: E402

__all__ = [
    'connect',
    'Connection',
    'DatabricksTable',
    'DatabricksSampler',
]

from typing import Any, TypeAlias

try:
    import snowflake.connector
except ImportError:
    raise ImportError("No module named 'snowflake'. Please install Kumo SDK "
                      "with the 'snowflake' extension via "
                      "`pip install kumoai[snowflake]`.")

Connection: TypeAlias = snowflake.connector.SnowflakeConnection


def connect(**kwargs: Any) -> Connection:
    r"""Opens a connection to a :class:`snowflake` database.

    If available, will return a connection to the active session.

    kwargs: Connection arguments, following the :class:`snowflake` protocol.
    """
    try:
        from snowflake.snowpark.context import get_active_session
        return get_active_session().connection
    except Exception:
        pass

    return snowflake.connector.connect(**kwargs)


from .table import SnowTable  # noqa: E402
from .sampler import SnowSampler  # noqa: E402

__all__ = [
    'connect',
    'Connection',
    'SnowTable',
    'SnowSampler',
]

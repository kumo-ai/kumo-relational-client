from pathlib import Path
from typing import Any, TypeAlias

try:
    import adbc_driver_duckdb.dbapi as adbc
except ImportError:
    raise ImportError("No module named 'adbc_driver_duckdb'. Please install "
                      "KumoRFM with the 'duckdb' extension via "
                      "`pip install kumorfm[duckdb]`.")

Connection: TypeAlias = adbc.Connection


def connect(uri: str | Path | None = None, **kwargs: Any) -> Connection:
    r"""Opens a connection to a :class:`duckdb` database.

    uri: The path to the database file to be opened, or ``None`` for an
        in-memory database.
    kwargs: Additional connection arguments, following the
        :class:`adbc_driver_duckdb` protocol.
    """
    return adbc.connect(str(uri) if uri is not None else None, **kwargs)


from .table import DuckDBTable  # noqa: E402
from .sampler import DuckDBSampler  # noqa: E402

__all__ = [
    'connect',
    'Connection',
    'DuckDBTable',
    'DuckDBSampler',
]

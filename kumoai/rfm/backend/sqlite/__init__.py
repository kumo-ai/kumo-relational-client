from pathlib import Path
from typing import Any, TypeAlias

try:
    import adbc_driver_sqlite.dbapi as adbc
except ImportError:
    raise ImportError("No module named 'adbc_driver_sqlite'. Please install "
                      "Kumo SDK with the 'sqlite' extension via "
                      "`pip install kumoai[sqlite]`.")

Connection: TypeAlias = adbc.AdbcSqliteConnection


def connect(uri: str | Path | None = None, **kwargs: Any) -> Connection:
    r"""Opens a connection to a :class:`sqlite` database.

    uri: The path to the database file to be opened.
    kwargs: Additional connection arguments, following the
        :class:`adbc_driver_sqlite` protocol.
    """
    return adbc.connect(str(uri) if uri is not None else None, **kwargs)


from .table import SQLiteTable  # noqa: E402
from .sampler import SQLiteSampler  # noqa: E402

__all__ = [
    'connect',
    'Connection',
    'SQLiteTable',
    'SQLiteSampler',
]

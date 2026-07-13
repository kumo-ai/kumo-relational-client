from __future__ import annotations

import importlib
import weakref
from typing import Any

_SQL_BACKENDS = ('sqlite', 'duckdb', 'snowflake', 'databricks')

OWNED_ATTR = '_sdfm_connectors_owned'

_ownership: weakref.WeakKeyDictionary[Any, bool] = weakref.WeakKeyDictionary()


def connect(backend: str, *args: Any, **kwargs: Any) -> Any:
    if backend not in _SQL_BACKENDS:
        raise ValueError(
            f'unknown backend {backend!r}; supported: {list(_SQL_BACKENDS)}',
        )
    module = importlib.import_module(f'{__name__}.{backend}')
    return module.connect(*args, **kwargs)


def mark_owned(connection: Any, owned: bool) -> None:
    try:
        _ownership[connection] = owned
        return
    except TypeError:
        pass
    try:
        setattr(connection, OWNED_ATTR, owned)
    except (AttributeError, TypeError):
        pass


def owns_connection(connection: Any) -> bool:
    r"""Whether we opened this connection, vs. borrowed an active session.

    Borrowed connections (e.g. an active Snowpark session) must not be closed
    by the reader, since the caller still owns them. Ownership is tracked
    out-of-band so a borrowed connection that rejects attribute writes is never
    misread as owned.
    """
    try:
        if connection in _ownership:
            return _ownership[connection]
    except TypeError:
        pass
    return getattr(connection, OWNED_ATTR, True)

from __future__ import annotations

import re

_TABLE_IDENTIFIER_RE = re.compile(
    r'^[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*)*$',
)


class MissingBackendError(ModuleNotFoundError):
    r"""Raised when a backend's driver package is not installed.

    Distinct from a plain ``ImportError``, which indicates an installed but
    broken driver (e.g. a native ABI mismatch) and must not be reported as an
    absent optional dependency.
    """
    def __init__(self, backend: str, driver: str) -> None:
        super().__init__(
            f"The {backend!r} backend requires {driver!r}. Install it via the "
            f"{backend!r} extra, e.g. `pip install "
            f"'sdfm-connectors[{backend}]'`.")
        self.backend = backend
        self.driver = driver


class ConnectorError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


def quote_ident(ident: str, char: str = '"') -> str:
    return char + ident.replace(char, char + char) + char


def resolve_sql(*, table: str | None, query: str | None) -> str:
    if (table is None) == (query is None):
        raise ConnectorError(
            "exactly one of 'table' or 'query' must be provided",
            code='INVALID_CONNECTOR_ARGS',
        )
    if query is not None:
        return query
    if not _TABLE_IDENTIFIER_RE.fullmatch(table):
        raise ConnectorError(
            f'invalid table identifier {table!r}',
            code='INVALID_CONNECTOR_ARGS',
            details={'table': table},
        )
    return f'SELECT * FROM {table}'

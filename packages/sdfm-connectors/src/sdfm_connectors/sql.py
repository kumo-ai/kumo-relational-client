# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import re
from contextlib import contextmanager
from typing import Any, Iterator

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


def require_driver(
    backend: str,
    driver: str,
    module: str,
    absent: tuple[str, ...] | None = None,
) -> Any:
    r"""Import a backend's driver module, or raise MissingBackendError.

    ``absent`` lists the module names whose ModuleNotFoundError means the
    optional driver is not installed; any other missing module signals a
    broken installation and propagates unchanged.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        if error.name not in (absent or (module.partition('.')[0],)):
            raise
        raise MissingBackendError(backend, driver) from error


@contextmanager
def driver_guard(
    code: str,
    context: str,
    *,
    missing_as_not_found: bool = False,
    **details: Any,
) -> Iterator[None]:
    r"""Translate driver exceptions into ConnectorError with a stable code.

    ConnectorError and MissingBackendError pass through untouched so guarded
    regions never double-wrap, and ImportError propagates unchanged because a
    broken driver installation must not be reported as a connect/query/read
    failure. The original driver exception stays chained as ``__cause__``.
    TypeError maps to INVALID_CONNECTOR_ARGS (a deterministic caller error,
    never worth retrying); FileNotFoundError maps to NOT_FOUND only where the
    guarded region reads data by path (``missing_as_not_found=True``).
    """
    try:
        yield
    except (ConnectorError, ImportError):
        raise
    except Exception as error:
        if isinstance(error, TypeError):
            mapped = 'INVALID_CONNECTOR_ARGS'
        elif missing_as_not_found and isinstance(error, FileNotFoundError):
            mapped = 'NOT_FOUND'
        else:
            mapped = code
        raise ConnectorError(
            f'{context}: {str(error) or type(error).__name__}',
            code=mapped,
            details={'driver_error': type(error).__name__, **details},
        ) from error


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

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import difflib
import importlib
import re
from contextlib import contextmanager
from typing import Any, Collection, Iterator

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


def check_connect_args(
    backend: str,
    kwargs: dict[str, Any],
    allowed: Collection[str],
) -> None:
    r"""Reject connection keywords the driver would silently ignore.

    The Snowflake and Databricks drivers accept ``**kwargs`` without complaint,
    so a misspelled parameter is otherwise dropped and the read runs against
    whatever catalog/schema the connection defaults to — a wrong answer rather
    than an error. ``allowed`` is read off the driver itself so it stays
    accurate across driver upgrades, and ``driver_options`` is the escape hatch
    for anything genuinely outside it.
    """
    unknown = sorted(set(kwargs) - set(allowed))
    if not unknown:
        return
    named = []
    for name in unknown:
        close = difflib.get_close_matches(name, allowed, n=1)
        named.append(f'{name!r} (did you mean {close[0]!r}?)' if close
                     else repr(name))
    raise ConnectorError(
        f'{backend} connector got unexpected arguments {", ".join(named)}; '
        "pass driver-specific options through 'driver_options'",
        code='INVALID_CONNECTOR_ARGS',
        details={'arguments': unknown},
    )


def merge_driver_options(
    backend: str,
    kwargs: dict[str, Any],
    driver_options: dict[str, Any] | None,
) -> dict[str, Any]:
    r"""Fold the ``driver_options`` escape hatch into the connection arguments.

    ``driver_options`` deliberately bypasses the allow-list, so it must not be
    able to quietly restate an argument that was already validated: naming a
    key both ways is rejected rather than letting one silently win, matching
    every other mutually-exclusive pair in this package. Merging before the
    backend inspects the result also means an option supplied through the
    escape hatch is never silently dropped from the connection decision.
    """
    options = dict(driver_options or {})
    if not options:
        return kwargs
    overlap = sorted(set(options) & set(kwargs))
    if overlap:
        raise ConnectorError(
            f'{backend} connector got {overlap} both as a connection argument '
            f"and in 'driver_options'; pass each one once",
            code='INVALID_CONNECTOR_ARGS',
            details={'arguments': overlap},
        )
    return {**kwargs, **options}


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

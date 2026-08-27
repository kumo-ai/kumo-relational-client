# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The exceptions this client raises.

Every call on :class:`~kumo_relational_client.RelationalClient` and its model handles raises
:class:`RelationalError` or one of its subclasses, so ``except RelationalError`` is enough
to catch anything the client reports. Building a graph is the exception: the
Kumo Relational engine validates a graph in its own constructor, before any client call,
and reports problems as :class:`ValueError`.

Every error carries a ``code``, a short stable string meant to be branched on.
Messages are written for people and will change; codes will not.
"""

from __future__ import annotations

from typing import Any


class RelationalError(Exception):
    r"""Base class for every error this client raises.

    Attributes:
        message: The human-readable description, without the code prefix.
        code: A short stable identifier, e.g. ``'TRANSPORT_ERROR'`` or
            ``'INVALID_REQUEST'``. Branch on this rather than on the message.
        details: Whatever structured context the failure carried, such as the
            ``invalid_params`` entries from a NIM validation error. Empty when
            there is none.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def __str__(self) -> str:
        if self.code:
            return f'[{self.code}] {self.message}'
        return self.message


class UnknownModelError(RelationalError):
    r"""A request named a model this client does not serve.

    Raised by :meth:`~kumo_relational_client.RelationalClient.capabilities` and
    when dispatching a request whose ``model`` is not the one this client
    serves. Code: ``UNKNOWN_MODEL``.
    """

    def __init__(self, model: str, known: list[str]) -> None:
        super().__init__(
            f'Unknown model {model!r}; this client serves: {sorted(known)}',
            code='UNKNOWN_MODEL',
            details={'model': model, 'known_models': sorted(known)},
        )
        self.model = model


class MissingExtraError(RelationalError):
    r"""An optional dependency this call needs is not installed.

    The Kumo Relational engine and the connector drivers ship as extras, so the client can
    be installed without them. The message names the ``pip install`` that fixes
    it. Code: ``MISSING_EXTRA``.
    """

    def __init__(self, extra: str, package: str) -> None:
        super().__init__(
            f'{package!r} is required for this adapter; install it with '
            f'`pip install kumo-relational-client[{extra}]`',
            code='MISSING_EXTRA',
            details={'extra': extra, 'package': package},
        )


class NimRequestError(RelationalError):
    r"""The NIM answered, and the answer was an error.

    Distinguished from :class:`RelationalError` by carrying the HTTP
    :attr:`status_code`, which is what tells a caller whether retrying can
    help: 5xx and 429 are worth retrying, 4xx means the request has to change.

    Attributes:
        status_code: The HTTP status the NIM returned.
    """

    def __init__(
        self,
        status_code: int,
        *,
        code: str | None,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)
        self.status_code = status_code

    def __str__(self) -> str:
        if self.code:
            return f'[{self.status_code} {self.code}] {self.message}'
        return f'[{self.status_code}] {self.message}'

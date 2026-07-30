# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


class SdfmError(Exception):
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


class UnknownModelError(SdfmError):
    def __init__(self, model: str, known: list[str]) -> None:
        super().__init__(
            f'Unknown model {model!r}; registered adapters: {sorted(known)}',
            code='UNKNOWN_MODEL',
            details={'model': model, 'known_models': sorted(known)},
        )
        self.model = model


class MissingExtraError(SdfmError):
    def __init__(self, extra: str, package: str) -> None:
        super().__init__(
            f'{package!r} is required for this adapter; install it with '
            f'`pip install nvidia-sdfm[{extra}]`',
            code='MISSING_EXTRA',
            details={'extra': extra, 'package': package},
        )


class NimRequestError(SdfmError):
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

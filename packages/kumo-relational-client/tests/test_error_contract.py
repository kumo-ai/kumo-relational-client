# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The error surface a caller can rely on.

What is pinned here: ``except RelationalError`` catches everything this package
raises, and each subclass is importable so a caller can branch on it.
"""

from __future__ import annotations

import pytest

import kumo_relational_client
from kumo_relational_client import (
    MissingExtraError,
    NimRequestError,
    RelationalError,
    UnknownModelError,
)

# --- the public surface -----------------------------------------------------


@pytest.mark.parametrize(
    'name',
    [
        'RelationalError',
        'NimRequestError',
        'MissingExtraError',
        'UnknownModelError',
    ],
)
def test_error_types_are_exported(name: str) -> None:
    r"""A caller cannot branch on an error they cannot import."""
    assert name in kumo_relational_client.__all__
    assert hasattr(kumo_relational_client, name)


@pytest.mark.parametrize(
    'error',
    [
        NimRequestError(503, code='X', message='m'),
        MissingExtraError('kumo_relational_engine', 'kumo_relational_engine'),
        UnknownModelError('nope', ['kumo-relational']),
    ],
)
def test_every_error_is_catchable_as_predict_error(
    error: RelationalError,
) -> None:
    with pytest.raises(RelationalError):
        raise error


@pytest.mark.parametrize(
    'cls',
    [RelationalError, NimRequestError, MissingExtraError, UnknownModelError],
)
def test_error_types_are_documented(cls: type) -> None:
    r"""These are the first thing a user reads when something breaks."""
    assert cls.__doc__ and len(cls.__doc__.strip()) > 40


def test_status_code_is_reachable_for_branching() -> None:
    error = NimRequestError(
        429, code='RATE_LIMIT_EXCEEDED', message='slow down'
    )
    assert error.status_code == 429
    assert str(error) == '[429 RATE_LIMIT_EXCEEDED] slow down'

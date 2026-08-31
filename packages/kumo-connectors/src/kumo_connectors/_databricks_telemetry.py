# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

DATABRICKS_PARTNER = 'nvidia'
DATABRICKS_PRODUCT = 'kumo-relational-client'

_PACKAGE_VERSION_RE = re.compile(
    r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)'
    r'(?:(a|b|rc)(0|[1-9]\d*))?$'
)
_PRERELEASE_NAMES = {'a': 'alpha', 'b': 'beta', 'rc': 'rc'}


def databricks_product_version(package_version: str) -> str:
    r"""Return the SDK release in the SemVer form Databricks accepts.

    Stable SDK releases already use SemVer. Python pre-releases use PEP 440's
    compact spelling, so translate the ``a`` / ``b`` / ``rc`` forms supported
    by this repository. Other PEP 440 forms are rejected deliberately: an
    invented fallback would make attribution claim the wrong SDK release.
    """
    match = _PACKAGE_VERSION_RE.fullmatch(package_version)
    if match is None:
        raise ValueError(
            f'Package version {package_version!r} cannot be represented in the '
            'Databricks User-Agent; expected X.Y.Z or an aN/bN/rcN '
            'pre-release'
        )
    major, minor, patch, prerelease, number = match.groups()
    version = f'{major}.{minor}.{patch}'
    if prerelease is not None:
        version = f'{version}-{_PRERELEASE_NAMES[prerelease]}.{number}'
    return version


def databricks_user_agent(
    package_version: str,
    additional_entry: str | None = None,
) -> str:
    r"""The partner identifier plus an optional caller-provided entry."""
    version = databricks_product_version(package_version)
    required = f'{DATABRICKS_PARTNER}_{DATABRICKS_PRODUCT}/{version}'
    if additional_entry:
        return f'{required} {additional_entry}'
    return required

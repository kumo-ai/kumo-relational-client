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
    r"""Return the engine release in the SemVer form Databricks accepts.

    Python pre-releases use PEP 440's compact spelling, so translate the
    ``a`` / ``b`` / ``rc`` forms without depending on another distribution's
    telemetry implementation.
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

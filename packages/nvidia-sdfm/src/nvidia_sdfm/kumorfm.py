# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nvidia_sdfm.errors import MissingExtraError

__all__ = [
    'Dtype',
    'ExplainConfig',
    'Explanation',
    'Graph',
    'LocalTable',
    'MaterializedPredictionRequest',
    'Stype',
    'Table',
    'TaskTable',
]

# Names exported from the driver package root rather than from ``kumorfm.rfm``.
_ROOT_NAMES = frozenset({'Dtype', 'Stype'})

if TYPE_CHECKING:
    from kumorfm import Dtype, Stype
    from kumorfm.rfm import (
        ExplainConfig,
        Explanation,
        Graph,
        LocalTable,
        MaterializedPredictionRequest,
        Table,
        TaskTable,
    )


def _import(module_name: str) -> Any:
    try:
        module = __import__(module_name, fromlist=['__name__'])
    except ModuleNotFoundError as error:
        if error.name != 'kumorfm':
            raise
        raise MissingExtraError('kumorfm', 'kumorfm') from error
    return module


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = _import('kumorfm' if name in _ROOT_NAMES else 'kumorfm.rfm')
        return getattr(module, name)
    raise AttributeError(
        f"module 'nvidia_sdfm.kumorfm' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

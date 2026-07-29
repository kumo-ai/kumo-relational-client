# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nvidia_sdfm.errors import MissingExtraError

__all__ = [
    'Graph',
    'Table',
    'LocalTable',
    'TaskTable',
    'ExplainConfig',
    'Explanation',
    'MaterializedPredictionRequest',
]

if TYPE_CHECKING:
    from kumorfm.rfm import (
        ExplainConfig,
        Explanation,
        Graph,
        LocalTable,
        MaterializedPredictionRequest,
        Table,
        TaskTable,
    )


def _engine() -> Any:
    try:
        import kumorfm.rfm as engine
    except ModuleNotFoundError as error:
        if error.name != 'kumorfm':
            raise
        raise MissingExtraError('kumorfm', 'kumorfm') from error
    return engine


def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(_engine(), name)
    raise AttributeError(
        f"module 'nvidia_sdfm.kumorfm' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

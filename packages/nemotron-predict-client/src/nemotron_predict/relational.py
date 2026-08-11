# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nemotron_predict.errors import MissingExtraError

# The one surface onto the driver package, so "you never import the driver
# directly" stays true: a name here is one the supported API hands back or takes.
#
# `ViewConversionWarning` earns its place because `Graph.from_snowflake_semantic_
# view` / `from_databricks_metric_view` are reached through this shim and
# routinely emit partial-conversion diagnostics; the driver gives them a
# dedicated category precisely so a caller can filter or escalate just those.
# `MaterializedPredictionRequest` and `TaskTable` do not: the first is produced
# only by `NemotronRelational.materialize_*` and the second built internally by the adapter,
# and `NemotronRelational` is deliberately absent, so nothing on the supported surface
# returns or accepts either.
__all__ = [
    'Dtype',
    'ExplainConfig',
    'Explanation',
    'Graph',
    'LocalTable',
    'Stype',
    'Table',
    'ViewConversionWarning',
]

_ROOT_NAMES = frozenset({'Dtype', 'Stype'})

if TYPE_CHECKING:
    from nemotron_relational import Dtype, Stype
    from nemotron_relational.rfm import (
        ExplainConfig,
        Explanation,
        Graph,
        LocalTable,
        Table,
        ViewConversionWarning,
    )


def _import(module_name: str) -> Any:
    try:
        module = __import__(module_name, fromlist=['__name__'])
    except ModuleNotFoundError as error:
        if error.name != 'nemotron_relational':
            raise
        raise MissingExtraError(
            'nemotron_relational', 'nemotron_relational'
        ) from error
    return module


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = _import(
            'nemotron_relational'
            if name in _ROOT_NAMES
            else 'nemotron_relational.rfm'
        )
        return getattr(module, name)
    raise AttributeError(
        f"module 'nemotron_predict.relational' has no attribute {name!r}"
    )


def __dir__() -> list[str]:
    return sorted(__all__)

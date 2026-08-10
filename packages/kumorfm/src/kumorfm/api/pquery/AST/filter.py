# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Union

import pydantic
from pydantic.dataclasses import dataclass

from kumorfm.api.pquery.AST.ast_node import ASTNode
from kumorfm.api.pquery.AST.column import Column
from kumorfm.api.pquery.utils import maybe_bold


@dataclass(repr=False)
class Filter(ASTNode):
    r"""Creates an atomic description of a filter on :obj:`target`
    corresponding to statement "target WHERE condition".

    Args:
        target: :class:`Column` defining the column to filter.
        condition: :class:`ASTNode` used to
            determine which rows to filter out.
    """

    target: Column | None = None
    condition: Union['Condition', 'LogicalOperation', None] = None

    def __post_init__(self) -> None:
        if self.target is None:
            raise ValueError(
                f"Class '{self.__class__.__name__}' is missing a target."
            )
        if self.condition is None:
            raise ValueError(
                f"Class '{self.__class__.__name__}' is missing a condition."
            )
        super().__post_init__()

    @property
    def children(self) -> list['ASTNode']:
        assert self.target is not None
        assert self.condition is not None
        return [self.target, self.condition]

    def to_string(self, rich: bool = False) -> str:
        r"""Creates a predictive query statement from the filter."""
        assert self.target is not None
        assert self.condition is not None
        return (
            f'{self.target.to_string(rich=rich)} '
            f'{maybe_bold("WHERE", rich)} '
            f'{self.condition.to_string(rich=rich)}'
        )


from kumorfm.api.pquery.AST.condition import Condition  # noqa: E402
from kumorfm.api.pquery.AST.logical_operation import (  # noqa: E402
    LogicalOperation,
)

if pydantic.__version__.startswith('1.'):
    Filter.__pydantic_model__.update_forward_refs()

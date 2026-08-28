# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

import pandas as pd

from kumo_relational_client.requests import ModelRequest

if TYPE_CHECKING:
    from kumo_relational_engine.rfm.rfm import Explanation

# Quoted whole: `Explanation` is a TYPE_CHECKING-only name, and a runtime
# `X | 'forward ref'` is a TypeError. The string is never evaluated unless a
# caller asks for the hints.
PredictResult: TypeAlias = 'pd.DataFrame | Explanation'

RequestTypes: TypeAlias = type[ModelRequest] | tuple[type[ModelRequest], ...]


def request_type_names(request_type: RequestTypes) -> str:
    r"""Render the accepted request type(s) for messages and capabilities."""
    types = request_type if isinstance(request_type, tuple) else (request_type,)
    return ' | '.join(rt.__name__ for rt in types)


@dataclass(frozen=True)
class ModelCapabilities:
    r"""What the model supports, via ``RelationalClient.capabilities(model)``.

    Describes the client side, not the connected NIM: nothing here is read from
    the endpoint, and none of it is enforced before a request is sent.
    Attributes:

    - ``model``: The model id, ``'kumo-relational'``.
    - ``request_type``: The name of the request class, for diagnostics. These
      classes are an implementation detail of the model handles and are not
      importable from ``kumo_relational_client``.
    - ``tasks``: The task kinds accepted.
    - ``outputs``: The output fields the model can produce.
    """

    model: str
    request_type: str
    tasks: tuple[str, ...] = field(default_factory=tuple)
    outputs: tuple[str, ...] = field(default_factory=tuple)

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Union

import pandas as pd

from nvidia_sdfm.core.transport import Transport
from nvidia_sdfm.errors import UnknownModelError
from nvidia_sdfm.requests import ModelRequest

if TYPE_CHECKING:
    from kumorfm.rfm.rfm import Explanation

PredictResult = Union[pd.DataFrame, "Explanation"]


@dataclass(frozen=True)
class ModelCapabilities:
    r"""What a model supports, discoverable via ``SDFMClient.capabilities(model)``."""
    model: str
    request_type: str
    tasks: tuple[str, ...] = field(default_factory=tuple)
    outputs: tuple[str, ...] = field(default_factory=tuple)


class ModelAdapter(ABC):
    name: str
    request_type: ClassVar[type[ModelRequest]]

    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        transport: Transport,
        request: ModelRequest,
    ) -> PredictResult:
        raise NotImplementedError


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, ModelAdapter] = {}

    def register(self, adapter: ModelAdapter) -> None:
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> ModelAdapter:
        adapter = self._adapters.get(name)
        if adapter is None:
            raise UnknownModelError(name, list(self._adapters))
        return adapter

    def names(self) -> list[str]:
        return sorted(self._adapters)

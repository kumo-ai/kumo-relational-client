from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from nvidia_sdfm.core.transport import TFMClient
from nvidia_sdfm.errors import UnknownModelError


class ModelAdapter(ABC):
    name: str

    @abstractmethod
    def predict(self, client: TFMClient, **kwargs: Any) -> pd.DataFrame:
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

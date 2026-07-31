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

RequestTypes = Union[type[ModelRequest], tuple[type[ModelRequest], ...]]


def request_type_names(request_type: RequestTypes) -> str:
    r"""Render an adapter's accepted request type(s) for messages and capabilities."""
    types = (request_type
             if isinstance(request_type, tuple) else (request_type, ))
    return ' | '.join(rt.__name__ for rt in types)


@dataclass(frozen=True)
class ModelCapabilities:
    r"""What a model supports, discoverable via ``SDFMClient.capabilities(model)``.

    Describes the client-side adapter, not the connected NIM: nothing here is
    read from the endpoint, and none of it is enforced before a request is
    sent. Attributes:

    - ``model``: The model id, e.g. ``'kumo-rfm'``.
    - ``request_type``: The name of the adapter's request class, for
      diagnostics. These classes are an implementation detail of the model
      handles and are not importable from ``nvidia_sdfm``.
    - ``tasks``: The task kinds the adapter accepts.
    - ``outputs``: The output fields the model can produce.
    """
    model: str
    request_type: str
    tasks: tuple[str, ...] = field(default_factory=tuple)
    outputs: tuple[str, ...] = field(default_factory=tuple)


class ModelAdapter(ABC):
    r"""The extension point for serving another model through ``SDFMClient``.

    A subclass sets ``name`` (the model id in the wire request) and
    ``request_type`` (the request class, or tuple of classes, it accepts), and
    is added to a client with ``SDFMClient._register``.
    """
    name: str
    request_type: ClassVar[RequestTypes]

    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        r"""What this adapter supports; see :class:`ModelCapabilities`."""
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        transport: Transport,
        request: ModelRequest,
    ) -> PredictResult:
        r"""Serializes ``request``, posts it over ``transport`` and parses the
        response.

        Args:
            transport: The owning client's HTTP layer.
            request: An instance of this adapter's ``request_type``; the client
                checks that before dispatching.
        """
        raise NotImplementedError


class AdapterRegistry:
    r"""The per-client mapping from model id to :class:`ModelAdapter`."""

    def __init__(self) -> None:
        self._adapters: dict[str, ModelAdapter] = {}

    def register(self, adapter: ModelAdapter) -> None:
        r"""Adds ``adapter``, replacing any adapter with the same ``name``."""
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> ModelAdapter:
        r"""The adapter registered for model id ``name``.

        Raises:
            UnknownModelError: If no adapter is registered under ``name``.
        """
        adapter = self._adapters.get(name)
        if adapter is None:
            raise UnknownModelError(name, list(self._adapters))
        return adapter

    def names(self) -> list[str]:
        r"""The registered model ids, sorted."""
        return sorted(self._adapters)

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, TypeAlias

import pandas as pd

from nvidia_sdfm.core.serving import ServingTarget
from nvidia_sdfm.core.transport import Transport
from nvidia_sdfm.errors import UnknownModelError
from nvidia_sdfm.requests import ModelRequest

if TYPE_CHECKING:
    from kumorfm.rfm.rfm import Explanation

# Quoted whole: `Explanation` is a TYPE_CHECKING-only name, and a runtime
# `X | 'forward ref'` is a TypeError. The string is never evaluated unless a
# caller asks for the hints.
PredictResult: TypeAlias = 'pd.DataFrame | Explanation'

RequestTypes: TypeAlias = type[ModelRequest] | tuple[type[ModelRequest], ...]


def request_type_names(request_type: RequestTypes) -> str:
    r"""Render an adapter's accepted request type(s) for messages and capabilities."""
    types = request_type if isinstance(request_type, tuple) else (request_type,)
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
        transport: Transport | ServingTarget,
        request: ModelRequest,
    ) -> PredictResult:
        r"""Serializes ``request``, posts it over ``transport`` and parses the
        response.

        ``transport`` is a :class:`Transport` for a NIM addressed by URL, or a
        :class:`ServingTarget` for a model served by name on a managed
        platform. An adapter that supports only one need not check: a
        ``ServingTarget`` raises an ``SdfmError`` naming the reason from
        ``url``, ``health_ready`` and ``predict`` alike, so reaching for the
        wrong one fails with an error a caller can read. An adapter that
        supports both must branch on the type, as
        :class:`~nvidia_sdfm.adapters.kumorfm.KumoRFMAdapter` does.

        Args:
            transport: The owning client's HTTP layer, or its serving target.
            request: An instance of this adapter's ``request_type``; the client
                checks that before dispatching.
        """
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027 - optional hook, see below
        r"""Releases anything this adapter opened that the client cannot see.

        Called by :meth:`~nvidia_sdfm.SDFMClient.close` for every registered
        adapter, before the client closes its own transport. Most adapters
        send through that transport and so have nothing of their own to
        release, which is why this does nothing by default; an adapter backed
        by a driver that opens its own connections overrides it.

        Releasing must leave the adapter usable: a client is closed once, but
        an adapter may share what it releases with another client, and that
        client has to be able to carry on. Reconnecting is an acceptable cost
        here, failing is not.
        """


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

    def close(self) -> None:
        r"""Closes every registered adapter.

        One adapter failing to release its resources must not strand the
        rest, so each is closed independently and the first error is raised
        only once they all have been.
        """
        error: BaseException | None = None
        for adapter in self._adapters.values():
            try:
                adapter.close()
            except BaseException as adapter_error:
                error = error or adapter_error
        if error is not None:
            raise error

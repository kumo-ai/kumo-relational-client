# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Addressing a model served by a managed platform rather than by URL.

:class:`~kumo_relational_client.core.transport.Transport` speaks HTTP to a NIM at a base
URL. A Databricks Model Serving endpoint is addressed by *name* through a
workspace client, so there is no URL, no API key, and no readiness route.

This carries that target instead. It exposes the same ``url`` and
``health_ready`` members a :class:`Transport` does, but both refuse: each raises
an :class:`RelationalError` naming the endpoint and the reason it does not
apply. An adapter reaching for one is a bug, and it
surfaces as that error rather than as a plausible-looking value or a request
to nowhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from kumo_relational_client.errors import RelationalError


@dataclass(frozen=True)
class ServingTarget:
    r"""A model served by name on a managed platform.

    Attributes:
        endpoint: The serving endpoint name.
        workspace_client: An injected platform client, or ``None`` to build one
            from ambient configuration. Kept out of ``repr``: a real
            ``WorkspaceClient`` renders its workspace host, and an injected one
            renders whatever it likes, including credentials.

    Frozen, because the endpoint and workspace client are this target's
    identity. Whether it is still usable is not, so ``_closed`` is set through
    ``object.__setattr__``.
    """

    endpoint: str
    kind: ClassVar[str] = ''

    platform_client: Any | None = field(default=None, repr=False)
    _closed: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, str) or not self.endpoint.strip():
            raise RelationalError(
                'endpoint must be a non-empty serving endpoint name',
                code='INVALID_CONFIGURATION',
            )
        if any(token in self.endpoint for token in ('://', '/', '?', '#', '@')):
            raise RelationalError(
                'endpoint must be a serving endpoint name, not a URL',
                code='INVALID_CONFIGURATION',
            )
        if self.endpoint != self.endpoint.strip():
            raise RelationalError(
                'endpoint name has surrounding whitespace',
                code='INVALID_CONFIGURATION',
            )

    def close(self) -> None:
        r"""Mark the target closed.

        There is nothing to release -- the workspace client owns its own
        transport -- but a closed client must refuse work here exactly as it
        does over HTTP. Otherwise ``close()`` followed by ``predict()`` fails
        on one transport and silently succeeds on the other.
        """
        object.__setattr__(self, '_closed', True)

    def _require_open(self) -> None:
        r"""Part of the transport contract ``RelationalClient._predict`` relies on.

        Absent it, the serving path raises ``AttributeError`` from inside the
        client rather than the documented error.
        """
        if self._closed:
            raise RelationalError(
                'this client is closed; construct a new RelationalClient',
                code='INVALID_CONFIGURATION',
            )

    @property
    def url(self) -> str:
        raise RelationalError(
            f'the serving endpoint {self.endpoint!r} has no URL; it is '
            'addressed by name through a workspace client',
            code='INVALID_CONFIGURATION',
        )

    def health_ready(self) -> bool:
        raise RelationalError(
            f'the serving endpoint {self.endpoint!r} exposes no readiness '
            "route; use the platform's own endpoint status instead",
            code='UNSUPPORTED_FEATURE',
        )


@dataclass(frozen=True)
class DatabricksServingTarget(ServingTarget):
    """A model served by Databricks Model Serving, addressed by endpoint name."""

    kind: ClassVar[str] = 'databricks'


@dataclass(frozen=True)
class SnowflakeServingTarget(ServingTarget):
    """A model served on Snowpark Container Services, addressed by service name.

    ``platform_client`` is a Snowpark ``Session`` or a ``snowflake.connector``
    connection rather than a Databricks ``WorkspaceClient``.
    """

    kind: ClassVar[str] = 'snowflake'

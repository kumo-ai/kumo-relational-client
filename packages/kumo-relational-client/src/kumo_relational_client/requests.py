# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import pandas as pd

if TYPE_CHECKING:
    from kumo_relational_client.relational import ExplainConfig


@dataclass
class ModelRequest:
    r"""Base class for typed per-model prediction requests.

    Every request declares the ``model`` id it targets, so the model handles
    (``client.relational(...)``) can dispatch to the right
    adapter without a loose ``**kwargs`` bag.
    """

    model: ClassVar[str] = ''


@dataclass
class KumoRelationalRequest(ModelRequest):
    r"""A relational Kumo Relational prediction request.

    ``graph`` is a ``kumo_relational_client.relational`` graph, ``query`` is a PQL string, and
    ``options`` forwards any additional keyword arguments to the driver's
    ``predict`` (e.g. ``num_neighbors``, ``anchor_time``).

    ``num_retries`` applies whether or not ``batch_size`` is set: each
    transiently failed request is retried that many times with an exponential
    backoff. Set it to ``0`` to fail on the first error. It sits above
    ``RelationalClient(max_retries=...)``, which retries at the transport.

    This is the internal request built by the public handle
    ``client.relational(graph).predict(query, ...)``. With ``explain`` set (a
    ``bool``, an ``ExplainConfig``, or an ``ExplainConfig`` dict) the prediction
    is limited to a single entity and ``client.relational(...).predict(...)``
    returns a ``nemotron_relational.rfm.rfm.Explanation`` (its ``prediction`` attribute
    holds the plain prediction DataFrame) instead of a bare DataFrame.

    Filling in ``Explanation.summary`` calls an external LLM endpoint (OpenAI
    by default) with the query, the predictions and the raw subgraph cell
    values. Pass ``explain=dict(skip_summary=True)`` to keep that data on the
    machine; see ``nemotron_relational.rfm.ExplainConfig`` for the full disclosure.
    """

    model: ClassVar[str] = 'kumo-relational'

    graph: Any
    query: str
    indices: Sequence[Any] | None = None
    run_mode: str = 'fast'
    explain: bool | ExplainConfig | dict[str, Any] = False
    batch_size: int | Literal['max'] | None = None
    num_retries: int = 1
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class KumoRelationalTaskRequest(ModelRequest):
    r"""A Kumo Relational prediction request with a caller-supplied context table.

    Where :class:`KumoRelationalRequest` derives its in-context (train) examples from a
    PQL ``query``, this request carries them directly: ``context`` holds the
    labelled rows and ``predict`` the rows to score, both referencing entities of
    ``entity_table`` in the ``graph`` (a ``(source, target)`` pair for temporal
    link prediction). Their columns follow the same convention as the prediction
    output -- ``ENTITY``, ``TARGET`` and an optional ``ANCHOR_TIMESTAMP`` --
    overridable via ``entity_column`` / ``target_column`` / ``time_column``.
    ``time_column`` defaults to ``ANCHOR_TIMESTAMP`` when present, otherwise the
    entity table's own time column.

    ``task_type='forecasting'`` additionally **requires** ``step_size`` -- the
    spacing between forecast steps, as an integer number of **nanoseconds**
    (e.g. ``int(pd.Timedelta(days=30).value)``) -- and uses ``num_forecasts``
    for how many steps to produce. Omitting ``step_size`` is rejected by the
    NIM, not client-side. Both are ignored for every other task type.

    This is the internal request built by the public handle
    ``client.relational(graph).predict_task(...)``. ``run_mode``, ``explain``,
    ``batch_size`` and ``options`` behave as in :class:`KumoRelationalRequest`.
    """

    model: ClassVar[str] = 'kumo-relational'

    graph: Any
    context: pd.DataFrame
    predict: pd.DataFrame
    task_type: str
    entity_table: str | Sequence[str]
    entity_column: str = 'ENTITY'
    target_column: str = 'TARGET'
    time_column: str | None = None
    num_forecasts: int = 1
    step_size: int | None = None
    run_mode: str = 'fast'
    explain: bool | ExplainConfig | dict[str, Any] = False
    batch_size: int | Literal['max'] | None = None
    num_retries: int = 1
    options: dict[str, Any] = field(default_factory=dict)

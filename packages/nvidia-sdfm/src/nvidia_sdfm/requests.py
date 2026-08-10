# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import pandas as pd

if TYPE_CHECKING:
    from nvidia_sdfm.kumorfm import ExplainConfig


@dataclass
class ModelRequest:
    r"""Base class for typed per-model prediction requests.

    Every request declares the ``model`` id it targets, so the model handles
    (``client.kumorfm(...)`` / ``client.tabicl(...)``) can dispatch to the right
    adapter without a loose ``**kwargs`` bag.
    """

    model: ClassVar[str] = ''


@dataclass
class TabICLSession:
    r"""The server-side context a ``TabICLModel`` handle reuses across calls.

    One of these lives on each handle so repeated ``predict`` calls against the
    same bound context upload it once instead of once per call. ``pinned`` is a
    digest of the context half of the payload the session was opened for -- a
    digest rather than the sections themselves so a handle does not hold a
    second copy of the context; a request whose context half differs cannot use
    the session. ``supported`` latches to ``False`` against a NIM without
    session routes, so the handle stops asking.

    ``lock`` serializes the decision of whether to open a session, so a handle
    shared across a thread pool -- which the handle's own docstring recommends
    -- opens one session rather than one per thread. Without it every thread
    reads ``id is None`` at once, they all create, the last writer wins and the
    rest are pinned on the NIM with no ``session_id`` left to release them by.
    Scoring against an established session runs outside the lock and so stays
    concurrent.
    """

    id: str | None = None
    pinned: str | None = None
    supported: bool = True
    lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )


@dataclass
class TabICLRequest(ModelRequest):
    r"""A single-table TabICL prediction request.

    ``context`` holds labelled rows (including the ``target`` column) and
    ``predict`` holds the unlabelled rows to score.
    """

    model: ClassVar[str] = 'tabicl'

    context: pd.DataFrame
    predict: pd.DataFrame
    task: str
    target: str
    outputs: list[str] = field(default_factory=lambda: ['prediction'])
    positive_class: str | None = None
    prediction_statistic: str | None = None
    quantile_levels: list[float] | None = None
    score_format: str | None = None
    embedding_dtype: str | None = None
    max_results: int | None = None
    request_id: str | None = None
    session: TabICLSession | None = field(
        default=None, repr=False, compare=False
    )


@dataclass
class KumoRFMRequest(ModelRequest):
    r"""A relational KumoRFM prediction request.

    ``graph`` is a ``nvidia_sdfm.kumorfm`` graph, ``query`` is a PQL string, and
    ``options`` forwards any additional keyword arguments to the driver's
    ``predict`` (e.g. ``num_neighbors``, ``anchor_time``).

    ``num_retries`` applies whether or not ``batch_size`` is set: each
    transiently failed request is retried that many times with an exponential
    backoff. Set it to ``0`` to fail on the first error. It sits above
    ``SDFMClient(max_retries=...)``, which retries at the transport.

    This is the internal request built by the public handle
    ``client.kumorfm(graph).predict(query, ...)``. With ``explain`` set (a
    ``bool``, an ``ExplainConfig``, or an ``ExplainConfig`` dict) the prediction
    is limited to a single entity and ``client.kumorfm(...).predict(...)``
    returns a ``kumorfm.rfm.rfm.Explanation`` (its ``prediction`` attribute
    holds the plain prediction DataFrame) instead of a bare DataFrame.

    Filling in ``Explanation.summary`` calls an external LLM endpoint (OpenAI
    by default) with the query, the predictions and the raw subgraph cell
    values. Pass ``explain=dict(skip_summary=True)`` to keep that data on the
    machine; see ``kumorfm.rfm.ExplainConfig`` for the full disclosure.
    """

    model: ClassVar[str] = 'kumo-rfm'

    graph: Any
    query: str
    indices: Sequence[Any] | None = None
    run_mode: str = 'fast'
    explain: bool | ExplainConfig | dict[str, Any] = False
    batch_size: int | Literal['max'] | None = None
    num_retries: int = 1
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class KumoRFMTaskRequest(ModelRequest):
    r"""A KumoRFM prediction request with a caller-supplied context table.

    Where :class:`KumoRFMRequest` derives its in-context (train) examples from a
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
    ``client.kumorfm(graph).predict_task(...)``. ``run_mode``, ``explain``,
    ``batch_size`` and ``options`` behave as in :class:`KumoRFMRequest`.
    """

    model: ClassVar[str] = 'kumo-rfm'

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

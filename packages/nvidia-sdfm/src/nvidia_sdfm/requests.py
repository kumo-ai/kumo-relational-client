from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Sequence

import pandas as pd


@dataclass
class ModelRequest:
    r"""Base class for typed per-model prediction requests.

    Every request declares the ``model`` id it targets, so ``SDFMClient.predict``
    can dispatch and validate against the model's advertised capabilities
    without a loose ``**kwargs`` bag.
    """
    model: ClassVar[str] = ''


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


@dataclass
class KumoRFMRequest(ModelRequest):
    r"""A relational KumoRFM prediction request.

    ``graph`` is a ``nvidia_sdfm.kumorfm`` graph, ``query`` is a PQL string, and
    ``options`` forwards any additional keyword arguments to the driver's
    ``predict`` (e.g. ``num_neighbors``, ``anchor_time``).
    """
    model: ClassVar[str] = 'kumo-rfm'

    graph: Any
    query: str
    indices: Sequence[Any] | None = None
    run_mode: str = 'fast'
    options: dict[str, Any] = field(default_factory=dict)

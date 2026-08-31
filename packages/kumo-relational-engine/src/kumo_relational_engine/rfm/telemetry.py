# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What a prediction has to say about itself afterwards.

A prediction that comes back wrong days later can only be explained if the run
that produced it can be reconstructed: which graph, under which engine, for which
query, at which anchor. Predictions are not deterministic and the data underneath
them moves, so re-running the question does not reveal what happened the first
time.

One record is emitted per prediction, whether it returned, refused, or raised. It
names the run rather than describing it: a graph fingerprint, versions, counts and
timings. No entity identifiers and no cell values, since both are the caller's
data. It goes to the ``kumo_relational_engine`` logger rather than to the progress
display, which belongs to whoever is watching the call rather than to whoever has
to explain it later.
"""

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass

logger = logging.getLogger('kumo_relational_engine')


@dataclass
class PredictionRecord:
    r"""One prediction, as it will be read back when something looks wrong."""

    graph_fingerprint: str = ''
    engine_version: str = ''
    query_type: str = ''
    entities: int = 0
    run_mode: str = ''
    batch_size: str = ''
    num_hops: int = 0
    explain: bool = False
    outcome: str = 'raised'
    seconds: float = 0.0
    error: str = ''


def emit(record: PredictionRecord) -> None:
    r"""Write one prediction to the log, never failing the call it describes."""
    try:
        logger.info('rfm.predict %s', json.dumps(asdict(record), default=str))
    except Exception:
        logger.debug('Prediction record could not be emitted', exc_info=True)


@contextmanager
def recorded(record: PredictionRecord) -> Iterator[PredictionRecord]:
    r"""Time a prediction and emit its record however it ends.

    A run that raised is the one most worth having a record of, so the record is
    written on the way out rather than on success.
    """
    started = time.perf_counter()
    try:
        yield record
        if record.outcome == 'raised':
            record.outcome = 'returned'
    except BaseException as error:
        record.error = f'{type(error).__name__}: {error}'
        raise
    finally:
        record.seconds = round(time.perf_counter() - started, 3)
        emit(record)

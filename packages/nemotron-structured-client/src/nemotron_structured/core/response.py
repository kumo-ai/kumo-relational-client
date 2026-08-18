# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from nemotron_structured.errors import StructuredError

_ITEM_FIELDS = (
    'id',
    'row_index',
    'prediction',
    'probabilities',
    'scores',
    'rankings',
    'embeddings',
    'quantiles',
    'explanation',
    'metadata',
)

# Columns a caller is entitled to index into even when every value is null.
# Everything else is an optional output the NIM omits unless asked for, and is
# dropped when empty so it does not clutter the frame.
_KEPT_FIELDS = ('row_index', 'prediction')


def parse_prediction_response(
    response: dict[str, Any],
    *,
    requested_fields: Iterable[str] | None = None,
) -> pd.DataFrame:
    predictions = response.get('predictions')
    if predictions is None:
        raise StructuredError(
            "response is missing required field 'predictions'",
            code='INVALID_RESPONSE',
        )
    if not isinstance(predictions, list):
        raise StructuredError(
            "response field 'predictions' must be a list, "
            f'got {type(predictions).__name__}',
            code='INVALID_RESPONSE',
        )
    rows = []
    for item in predictions:
        if not isinstance(item, dict):
            raise StructuredError(
                f'prediction item must be an object, got {type(item).__name__}',
                code='INVALID_RESPONSE',
            )
        if item.get('row_index') is None:
            raise StructuredError(
                "prediction item is missing required field 'row_index'",
                code='INVALID_RESPONSE',
            )
        rows.append({field: item.get(field) for field in _ITEM_FIELDS})
    frame = pd.DataFrame(rows, columns=_ITEM_FIELDS)
    kept = set(_KEPT_FIELDS) | set(requested_fields or ())
    frame = frame.drop(
        columns=[
            column
            for column in frame.columns
            if column not in kept and frame[column].isna().all()
        ]
    )
    if len(frame) > 0:
        try:
            frame = frame.sort_values('row_index', kind='stable')
        except TypeError as error:
            raise StructuredError(
                "response field 'row_index' has values that cannot be ordered",
                code='INVALID_RESPONSE',
            ) from error
        frame = frame.reset_index(drop=True)
    return frame

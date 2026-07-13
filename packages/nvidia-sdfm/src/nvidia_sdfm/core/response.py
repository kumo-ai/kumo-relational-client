from __future__ import annotations

from typing import Any

import pandas as pd

from nvidia_sdfm.errors import SdfmError

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


def parse_prediction_response(response: dict[str, Any]) -> pd.DataFrame:
    predictions = response.get('predictions')
    if predictions is None:
        raise SdfmError(
            "response is missing required field 'predictions'",
            code='INVALID_RESPONSE',
        )
    if not isinstance(predictions, list):
        raise SdfmError(
            "response field 'predictions' must be a list, "
            f'got {type(predictions).__name__}',
            code='INVALID_RESPONSE',
        )
    rows = []
    for item in predictions:
        if not isinstance(item, dict):
            raise SdfmError(
                f'prediction item must be an object, got {type(item).__name__}',
                code='INVALID_RESPONSE',
            )
        if 'row_index' not in item:
            raise SdfmError(
                "prediction item is missing required field 'row_index'",
                code='INVALID_RESPONSE',
            )
        rows.append({field: item.get(field) for field in _ITEM_FIELDS})
    frame = pd.DataFrame(rows, columns=_ITEM_FIELDS)
    frame = frame.dropna(axis='columns', how='all')
    if 'row_index' in frame.columns and len(frame) > 0:
        frame = frame.sort_values('row_index', kind='stable')
        frame = frame.reset_index(drop=True)
    return frame

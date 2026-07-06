from __future__ import annotations

import uuid
from typing import Any

import pandas as pd

from nvidia_sdfm.base import ModelAdapter
from nvidia_sdfm.core.dtypes import infer_tfm_dtype, serialize_column
from nvidia_sdfm.core.response import parse_prediction_response
from nvidia_sdfm.core.transport import TFMClient
from nvidia_sdfm.errors import SdfmError

_CLASSIFICATION_KINDS = frozenset({
    'classification',
    'binary_classification',
    'multiclass_classification',
})


def _new_request_id() -> str:
    return f'sdfm_{uuid.uuid4().hex[:12]}'


def _table_payload(frame: pd.DataFrame, dtypes: dict[str, str]) -> dict[str, Any]:
    columns = list(frame.columns)
    if not columns:
        return {
            'format': 'arrays',
            'columns': [],
            'rows': [[] for _ in range(len(frame))],
        }
    serialized_columns = [
        serialize_column(frame[column], dtypes[column]) for column in columns
    ]
    rows = [list(row) for row in zip(*serialized_columns)]
    return {'format': 'arrays', 'columns': columns, 'rows': rows}


def build_request(
    *,
    context: pd.DataFrame,
    predict: pd.DataFrame,
    task: str,
    target: str,
    outputs: list[str],
    positive_class: str | None = None,
    prediction_statistic: str | None = None,
    quantile_levels: list[float] | None = None,
    score_format: str | None = None,
    embedding_dtype: str | None = None,
    max_results: int | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    if target not in context.columns:
        raise SdfmError(
            f'target column {target!r} not found in context',
            code='INVALID_REQUEST',
        )

    dtypes = {column: infer_tfm_dtype(context[column]) for column in context.columns}
    for column in predict.columns:
        dtypes.setdefault(column, infer_tfm_dtype(predict[column]))

    task_spec: dict[str, Any] = {
        'kind': task,
        'target': {
            'column_name': target,
            'dtype': dtypes[target],
        },
    }
    if task in _CLASSIFICATION_KINDS:
        classes = sorted(str(value) for value in context[target].dropna().unique())
        task_spec['target']['classes'] = classes
        if positive_class is not None:
            task_spec['target']['positive_class'] = positive_class

    output_spec: dict[str, Any] = {'fields': outputs}
    if prediction_statistic is not None:
        output_spec['prediction_statistic'] = prediction_statistic
    if quantile_levels is not None:
        output_spec['quantile_levels'] = quantile_levels
    if score_format is not None:
        output_spec['score_format'] = score_format
    if embedding_dtype is not None:
        output_spec['embedding_dtype'] = embedding_dtype
    if max_results is not None:
        output_spec['max_results'] = max_results

    schema = {
        'instance_table': {
            'columns': {
                column: {'dtype': dtype} for column, dtype in dtypes.items()
            },
        },
        'related_tables': {},
        'relationships': [],
    }

    return {
        'model': 'tabicl',
        'task': task_spec,
        'schema': schema,
        'context': {
            'instance_table': _table_payload(context, dtypes),
            'related_tables': {},
        },
        'predict': {
            'instance_table': _table_payload(predict, dtypes),
            'related_tables': {},
        },
        'output': output_spec,
        'metadata': {'request_id': request_id or _new_request_id()},
    }


class TabICLAdapter(ModelAdapter):
    name = 'tabicl'

    def predict(self, client: TFMClient, **kwargs: Any) -> pd.DataFrame:
        payload = build_request(**kwargs)
        response = client.predict(payload)
        return parse_prediction_response(response)

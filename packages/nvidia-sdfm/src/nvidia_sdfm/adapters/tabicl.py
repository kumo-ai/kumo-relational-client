# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid
from typing import Any

import pandas as pd

from nvidia_sdfm.base import ModelAdapter, ModelCapabilities
from nvidia_sdfm.core.dtypes import (
    infer_tfm_dtype,
    serialize_column,
    widen_tfm_dtype,
)
from nvidia_sdfm.core.response import parse_prediction_response
from nvidia_sdfm.core.transport import Transport
from nvidia_sdfm.errors import SdfmError
from nvidia_sdfm.requests import TabICLRequest

_CLASSIFICATION_KINDS = frozenset({
    'classification',
    'binary_classification',
    'multiclass_classification',
})

_TASK_KINDS = _CLASSIFICATION_KINDS | frozenset({'regression'})

# The NIM serves the two canonical kinds only; the finer-grained names are
# aliases for the same estimator and must be normalized before they go out.
_WIRE_TASK_KINDS = {
    'binary_classification': 'classification',
    'multiclass_classification': 'classification',
}

# TabICL builds its KV cache for a fixed number of classes; beyond that the
# model raises and the NIM answers with an opaque 500.
_MAX_CLASSES = 10

# Fields TabICL can produce per task kind. Anything else is accepted by the
# contract but dropped server-side without comment.
_OUTPUT_FIELDS = {
    'classification': ('prediction', 'probabilities'),
    'regression': ('prediction', 'quantiles'),
}


def _new_request_id() -> str:
    return f'sdfm_{uuid.uuid4().hex[:12]}'


def _check_columns(name: str, frame: pd.DataFrame) -> None:
    if frame.columns.has_duplicates:
        duplicates = sorted({str(column) for column in
                             frame.columns[frame.columns.duplicated()]})
        raise SdfmError(
            f'{name} has duplicate column name(s) {duplicates}',
            code='INVALID_REQUEST',
        )
    non_strings = [column for column in frame.columns
                   if not isinstance(column, str)]
    if non_strings:
        raise SdfmError(
            f'{name} column names must be strings; got {non_strings}',
            code='INVALID_REQUEST',
        )


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
    if task not in _TASK_KINDS:
        raise SdfmError(
            f'task must be one of {sorted(_TASK_KINDS)}; got {task!r}',
            code='INVALID_REQUEST',
        )
    wire_task = _WIRE_TASK_KINDS.get(task, task)

    _check_columns('context', context)
    _check_columns('predict', predict)

    if target not in context.columns:
        raise SdfmError(
            f'target column {target!r} not found in context',
            code='INVALID_REQUEST',
        )
    if len(context) == 0:
        raise SdfmError(
            'context is empty; TabICL needs at least one labelled row',
            code='INVALID_REQUEST',
        )
    if len(predict) == 0:
        raise SdfmError(
            'predict is empty; TabICL needs at least one row to score',
            code='INVALID_REQUEST',
        )

    unlabelled = context[target].isna()
    if bool(unlabelled.any()):
        rows = list(context.index[unlabelled][:10])
        raise SdfmError(
            f'context target {target!r} has {int(unlabelled.sum())} missing '
            f'value(s) at rows {rows}; drop or fill them',
            code='INVALID_REQUEST',
        )

    unsupported = [field for field in outputs
                   if field not in _OUTPUT_FIELDS[wire_task]]
    if unsupported:
        raise SdfmError(
            f'TabICL does not produce {unsupported} for a {wire_task} task; '
            f'supported output fields: {list(_OUTPUT_FIELDS[wire_task])}',
            code='INVALID_REQUEST',
        )
    if quantile_levels is not None and not all(
            0.0 < level < 1.0 for level in quantile_levels):
        raise SdfmError(
            f'quantile_levels must lie strictly between 0 and 1; got '
            f'{quantile_levels}',
            code='INVALID_REQUEST',
        )

    # A column typed differently in the two frames travels under the wider of
    # the two dtypes, so neither frame's values are coerced to the other's.
    dtypes = {
        column: widen_tfm_dtype(
            infer_tfm_dtype(frame[column])
            for frame in (context, predict) if column in frame.columns
        )
        for column in dict.fromkeys([*context.columns, *predict.columns])
    }

    task_spec: dict[str, Any] = {
        'kind': wire_task,
        'target': {
            'column_name': target,
            'dtype': dtypes[target],
        },
    }
    if task in _CLASSIFICATION_KINDS:
        classes = sorted(str(value) for value in context[target].dropna().unique())
        if len(classes) > _MAX_CLASSES:
            raise SdfmError(
                f'TabICL supports at most {_MAX_CLASSES} classes; context '
                f'target {target!r} has {len(classes)}',
                code='INVALID_REQUEST',
            )
        task_spec['target']['classes'] = classes
        if positive_class is not None:
            wire_positive_class = str(positive_class)
            if wire_positive_class not in classes:
                raise SdfmError(
                    f'positive_class {positive_class!r} is not one of the '
                    f'context classes {classes}',
                    code='INVALID_REQUEST',
                )
            task_spec['target']['positive_class'] = wire_positive_class

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
    request_type = TabICLRequest

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model='tabicl',
            request_type=self.request_type.__name__,
            tasks=tuple(sorted(_TASK_KINDS)),
            outputs=('prediction', 'probabilities', 'quantiles'),
        )

    def predict(
        self,
        transport: Transport,
        request: TabICLRequest,
    ) -> pd.DataFrame:
        payload = build_request(
            context=request.context,
            predict=request.predict,
            task=request.task,
            target=request.target,
            outputs=request.outputs,
            positive_class=request.positive_class,
            prediction_statistic=request.prediction_statistic,
            quantile_levels=request.quantile_levels,
            score_format=request.score_format,
            embedding_dtype=request.embedding_dtype,
            max_results=request.max_results,
            request_id=request.request_id,
        )
        response = transport.predict(payload)
        return parse_prediction_response(
            response, requested_fields=request.outputs)

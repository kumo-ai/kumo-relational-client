# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
import pytest

from nvidia_sdfm.adapters.tabicl import build_request
from nvidia_sdfm.errors import SdfmError

from conftest import canonical_examples_available, load_canonical_example


def _dataframe_from_table(table: dict) -> pd.DataFrame:
    return pd.DataFrame(table['rows'], columns=table['columns'])


def _comparable(payload: dict) -> dict:
    return {
        'model': payload['model'],
        'task_kind': payload['task']['kind'],
        'target_column': payload['task']['target']['column_name'],
        'target_dtype': payload['task']['target']['dtype'],
        'target_classes': payload['task']['target'].get('classes'),
        'context_format': payload['context']['instance_table']['format'],
        'context_columns': payload['context']['instance_table']['columns'],
        'context_rows': payload['context']['instance_table']['rows'],
        'predict_columns': payload['predict']['instance_table']['columns'],
        'predict_rows': payload['predict']['instance_table']['rows'],
        'output_fields': payload['output']['fields'],
        'column_dtypes': {
            name: spec['dtype']
            for name, spec in payload['schema']['instance_table']['columns'].items()
        },
    }


@canonical_examples_available
@pytest.mark.parametrize(('example_file', 'target', 'task', 'outputs', 'extra'), [
    (
        'prediction_tabicl_arrays.json',
        'target_col',
        'classification',
        ['prediction', 'probabilities'],
        {},
    ),
    (
        'prediction_tabicl_numeric_classification.json',
        'target_col',
        'classification',
        ['prediction', 'probabilities'],
        {},
    ),
    (
        'prediction_tabicl_regression_quantiles.json',
        'target_col',
        'regression',
        ['prediction', 'quantiles'],
        {'prediction_statistic': 'mean', 'quantile_levels': [0.1, 0.5, 0.9]},
    ),
])
def test_build_request_matches_canonical_example(
    example_file, target, task, outputs, extra,
):
    canonical = load_canonical_example('tabicl', example_file)
    context = _dataframe_from_table(canonical['context']['instance_table'])
    predict = _dataframe_from_table(canonical['predict']['instance_table'])

    built = build_request(
        context=context,
        predict=predict,
        task=task,
        target=target,
        outputs=outputs,
        **extra,
    )

    assert _comparable(built) == _comparable(canonical)


def test_build_request_omits_positive_class_when_not_given(context_df, predict_df):
    payload = build_request(
        context=context_df,
        predict=predict_df,
        task='classification',
        target='target_col',
        outputs=['prediction'],
    )
    assert 'positive_class' not in payload['task']['target']


def test_build_request_includes_positive_class_when_given(context_df, predict_df):
    payload = build_request(
        context=context_df,
        predict=predict_df,
        task='classification',
        target='target_col',
        outputs=['prediction'],
        positive_class='yes',
    )
    assert payload['task']['target']['positive_class'] == 'yes'


def test_build_request_regression_has_no_classes(context_df, predict_df):
    context_df = context_df.assign(target_col=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    payload = build_request(
        context=context_df,
        predict=predict_df,
        task='regression',
        target='target_col',
        outputs=['prediction'],
    )
    assert 'classes' not in payload['task']['target']


def test_build_request_missing_target_raises(context_df, predict_df):
    with pytest.raises(SdfmError):
        build_request(
            context=context_df,
            predict=predict_df,
            task='classification',
            target='does-not-exist',
            outputs=['prediction'],
        )


def test_table_payload_preserves_rows_for_zero_column_frame():
    from nvidia_sdfm.adapters.tabicl import _table_payload

    frame = pd.DataFrame(index=range(3))
    payload = _table_payload(frame, {})
    assert payload['columns'] == []
    assert payload['rows'] == [[], [], []]


def test_build_request_generates_unique_request_ids(context_df, predict_df):
    first = build_request(
        context=context_df, predict=predict_df, task='classification',
        target='target_col', outputs=['prediction'],
    )
    second = build_request(
        context=context_df, predict=predict_df, task='classification',
        target='target_col', outputs=['prediction'],
    )
    assert first['metadata']['request_id'] != second['metadata']['request_id']

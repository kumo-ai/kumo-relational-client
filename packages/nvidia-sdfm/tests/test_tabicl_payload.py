# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
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


@pytest.mark.parametrize(('labels', 'positive_class', 'expected'), [
    ([0, 1, 0, 1, 1, 0], 1, '1'),
    ([True, False, True, False, False, True], True, 'True'),
    (np.array([0, 1, 0, 1, 1, 0], dtype=np.int64), np.int64(1), '1'),
])
def test_build_request_serializes_positive_class_like_classes(
    context_df, predict_df, labels, positive_class, expected,
):
    context_df = context_df.assign(target_col=labels)
    payload = build_request(
        context=context_df,
        predict=predict_df,
        task='binary_classification',
        target='target_col',
        outputs=['prediction'],
        positive_class=positive_class,
    )
    target = payload['task']['target']
    assert target['positive_class'] == expected
    assert target['positive_class'] in target['classes']
    assert all(isinstance(value, str) for value in target['classes'])
    assert isinstance(target['positive_class'], str)


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


def test_build_request_unknown_task_raises(context_df, predict_df):
    with pytest.raises(SdfmError) as err:
        build_request(
            context=context_df,
            predict=predict_df,
            task='__unknown_task__',
            target='y',
            outputs=['prediction'],
        )
    assert err.value.code == 'INVALID_REQUEST'
    assert 'task' in str(err.value)


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


# Regression tests for bugs/tabicl-capabilities-advertises-tasks-the-nim-
# rejects.md: the NIM serves 'classification' and 'regression' only.
@pytest.mark.parametrize('task', [
    'binary_classification',
    'multiclass_classification',
])
def test_build_request_normalizes_classification_aliases(
    context_df, predict_df, task,
):
    payload = build_request(
        context=context_df,
        predict=predict_df,
        task=task,
        target='target_col',
        outputs=['prediction'],
    )
    assert payload['task']['kind'] == 'classification'


def test_build_request_keeps_canonical_task_kinds_verbatim(
    context_df, predict_df,
):
    payload = build_request(
        context=context_df,
        predict=predict_df,
        task='classification',
        target='target_col',
        outputs=['prediction'],
    )
    assert payload['task']['kind'] == 'classification'


# Regression tests for bugs/tabicl-opaque-500-on-empty-frames-and-unlabelled-
# rows.md: all three used to reach the NIM and come back as a bare 500.
def test_build_request_empty_context_raises(predict_df, context_df):
    with pytest.raises(SdfmError) as err:
        build_request(
            context=context_df.iloc[:0],
            predict=predict_df,
            task='classification',
            target='target_col',
            outputs=['prediction'],
        )
    assert err.value.code == 'INVALID_REQUEST'
    assert 'context is empty' in str(err.value)


def test_build_request_empty_predict_raises(context_df, predict_df):
    with pytest.raises(SdfmError) as err:
        build_request(
            context=context_df,
            predict=predict_df.iloc[:0],
            task='classification',
            target='target_col',
            outputs=['prediction'],
        )
    assert err.value.code == 'INVALID_REQUEST'
    assert 'predict is empty' in str(err.value)


def test_build_request_unlabelled_context_rows_raise(context_df, predict_df):
    context_df.loc[[0, 2], 'target_col'] = None
    with pytest.raises(SdfmError) as err:
        build_request(
            context=context_df,
            predict=predict_df,
            task='classification',
            target='target_col',
            outputs=['prediction'],
        )
    assert err.value.code == 'INVALID_REQUEST'
    assert '2 missing value(s) at rows [0, 2]' in str(err.value)


# Regression test for bugs/tabicl-more-than-ten-classes-opaque-500.md.
def test_build_request_more_than_ten_classes_raises(predict_df):
    context = pd.DataFrame({
        'age': range(11),
        'target_col': [f'c{index}' for index in range(11)],
    })
    with pytest.raises(SdfmError) as err:
        build_request(
            context=context,
            predict=predict_df,
            task='classification',
            target='target_col',
            outputs=['prediction'],
        )
    assert err.value.code == 'INVALID_REQUEST'
    assert 'at most 10 classes' in str(err.value)


def test_build_request_allows_ten_classes(predict_df):
    context = pd.DataFrame({
        'age': range(10),
        'target_col': [f'c{index}' for index in range(10)],
    })
    payload = build_request(
        context=context,
        predict=predict_df,
        task='classification',
        target='target_col',
        outputs=['prediction'],
    )
    assert len(payload['task']['target']['classes']) == 10


# Regression tests for bugs/tabicl-predict-knobs-unvalidated-and-
# undocumented.md.
def test_build_request_unknown_positive_class_raises(context_df, predict_df):
    with pytest.raises(SdfmError) as err:
        build_request(
            context=context_df,
            predict=predict_df,
            task='classification',
            target='target_col',
            outputs=['prediction'],
            positive_class='not-a-class',
        )
    assert err.value.code == 'INVALID_REQUEST'
    assert "'not-a-class'" in str(err.value)


@pytest.mark.parametrize('levels', [[-0.5, 0.5], [0.0, 0.5], [0.5, 1.0]])
def test_build_request_out_of_range_quantile_levels_raise(
    context_df, predict_df, levels,
):
    context_df = context_df.assign(target_col=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    with pytest.raises(SdfmError) as err:
        build_request(
            context=context_df,
            predict=predict_df,
            task='regression',
            target='target_col',
            outputs=['prediction', 'quantiles'],
            quantile_levels=levels,
        )
    assert err.value.code == 'INVALID_REQUEST'
    assert 'quantile_levels' in str(err.value)


@pytest.mark.parametrize(('task', 'outputs'), [
    ('regression', ['prediction', 'probabilities']),
    ('classification', ['prediction', 'quantiles']),
    ('classification', ['embeddings']),
])
def test_build_request_unproducible_output_field_raises(
    context_df, predict_df, task, outputs,
):
    if task == 'regression':
        context_df = context_df.assign(
            target_col=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    with pytest.raises(SdfmError) as err:
        build_request(
            context=context_df,
            predict=predict_df,
            task=task,
            target='target_col',
            outputs=outputs,
        )
    assert err.value.code == 'INVALID_REQUEST'
    assert 'does not produce' in str(err.value)


# Regression tests for bugs/tabicl-request-builder-crashes-on-ndarray-and-
# duplicate-columns.md.
def test_build_request_duplicate_column_names_raise(predict_df):
    context = pd.DataFrame([[1, 2, 'a'], [3, 4, 'b']],
                           columns=['x', 'x', 'target_col'])
    with pytest.raises(SdfmError) as err:
        build_request(
            context=context,
            predict=predict_df,
            task='classification',
            target='target_col',
            outputs=['prediction'],
        )
    assert err.value.code == 'INVALID_REQUEST'
    assert "duplicate column name(s) ['x']" in str(err.value)


def test_build_request_non_string_column_names_raise(predict_df):
    context = pd.DataFrame({0: [1, 2], 1: [3, 4], 'target_col': ['a', 'b']})
    with pytest.raises(SdfmError) as err:
        build_request(
            context=context,
            predict=predict_df,
            task='classification',
            target='target_col',
            outputs=['prediction'],
        )
    assert err.value.code == 'INVALID_REQUEST'
    assert 'column names must be strings' in str(err.value)


def test_build_request_serializes_ndarray_cells(predict_df):
    context = pd.DataFrame({
        'embedding': [np.array([1, 2]), np.array([3, 4])],
        'target_col': ['a', 'b'],
    })
    payload = build_request(
        context=context,
        predict=predict_df,
        task='classification',
        target='target_col',
        outputs=['prediction'],
    )
    assert payload['context']['instance_table']['rows'] == [
        ['[1 2]', 'a'], ['[3 4]', 'b'],
    ]


def test_predict_frame_fractions_survive_an_int_typed_context_column():
    # Regression: bugs/tabicl-predict-frame-int-truncation.md -- the context
    # frame's int64 dtype used to be applied to the predict frame, flooring
    # every value with a bare ``int()``.
    context = pd.DataFrame({
        'price': [1, 2, 3, 4],
        'target_col': ['a', 'b', 'a', 'b'],
    })
    predict = pd.DataFrame({'price': [1.9, 2.5, 3.7]})
    payload = build_request(
        context=context,
        predict=predict,
        task='classification',
        target='target_col',
        outputs=['prediction'],
    )
    columns = payload['schema']['instance_table']['columns']
    assert columns['price']['dtype'] == 'float64'
    assert payload['predict']['instance_table']['rows'] == [
        [1.9], [2.5], [3.7],
    ]
    assert payload['context']['instance_table']['rows'] == [
        [1.0, 'a'], [2.0, 'b'], [3.0, 'a'], [4.0, 'b'],
    ]


def test_predict_frame_nan_upcast_does_not_truncate_the_column():
    # Regression: bugs/tabicl-predict-frame-int-truncation.md -- a single
    # missing value upcasts an integer column to float64 in pandas, which used
    # to hand the whole predict column to the truncating branch.
    context = pd.DataFrame({
        'price': [30, 41, 52, 63],
        'target_col': ['a', 'b', 'a', 'b'],
    })
    predict = pd.DataFrame({'price': [30.5, float('nan'), 41.99]})
    payload = build_request(
        context=context,
        predict=predict,
        task='classification',
        target='target_col',
        outputs=['prediction'],
    )
    assert payload['predict']['instance_table']['rows'] == [
        [30.5], [None], [41.99],
    ]


def test_build_request_rejects_non_finite_values(predict_df):
    # Regression: bugs/rfm-nonfinite-and-decimal-cells-raise-bare-json-errors.md
    # -- ``inf`` used to be emitted verbatim, producing invalid JSON.
    context = pd.DataFrame({
        'score': [1.0, float('inf'), 2.0],
        'target_col': ['a', 'b', 'a'],
    })
    with pytest.raises(SdfmError) as err:
        build_request(
            context=context,
            predict=predict_df[['score']],
            task='classification',
            target='target_col',
            outputs=['prediction'],
        )
    assert err.value.code == 'INVALID_REQUEST'
    assert 'not JSON-representable' in str(err.value)

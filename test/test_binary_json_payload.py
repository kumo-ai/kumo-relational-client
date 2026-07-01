import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from kumoapi.model_plan import RunMode
from kumoapi.rfm import RFMPredictRequest
from kumoapi.rfm.context import Context, Subgraph, Table
from kumoapi.rfm.inference import ClassificationInferenceConfig
from kumoapi.task import TaskType
from kumoapi.typing import Stype

from kumoai.rfm.payload import _instance_dataframe, predict_request_to_json


def _binary_context(
    y_train: pd.Series,
    y_test: pd.Series | None,
) -> Context:
    batch_size = len(y_train) + (len(y_test) if y_test is not None else 0)
    table_name = 'ENTITY'
    primary_key = 'ENTITY_ID'
    return Context(
        task_type=TaskType.BINARY_CLASSIFICATION,
        entity_table_names=(table_name,),
        subgraph=Subgraph(
            anchor_time=np.full(
                batch_size,
                pd.Timestamp.min.value,
                dtype=np.int64,
            ),
            table_dict={
                table_name:
                Table(
                    df=pd.DataFrame({
                        primary_key: np.arange(batch_size),
                    }),
                    row=None,
                    batch=np.arange(batch_size),
                    num_sampled_nodes=[batch_size],
                    stype_dict={primary_key: Stype.ID},
                    primary_key=primary_key,
                ),
            },
            link_dict={},
        ),
        y_train=y_train,
        y_test=y_test,
    )


def _binary_payload(context: Context) -> dict[str, Any]:
    request = RFMPredictRequest(
        context=context,
        run_mode=RunMode.FAST,
        inference_config=ClassificationInferenceConfig(),
    )
    return predict_request_to_json(request)


@pytest.mark.parametrize(
    'labels',
    [
        pd.Series([0, 1], name='TARGET', dtype='int64'),
        pd.Series(
            [np.int32(0), np.int64(1)],
            name='TARGET',
            dtype='object',
        ),
    ],
    ids=['python-integers', 'numpy-integers'],
)
def test_binary_integer_targets_serialize_as_json_booleans(
    labels: pd.Series,
) -> None:
    context = _binary_context(
        y_train=labels,
        y_test=pd.Series(
            [np.int64(1), np.int32(0)],
            name='TARGET',
            dtype='object',
        ),
    )

    instance_df, _, _ = _instance_dataframe(context)
    assert instance_df['TARGET'].tolist() == [False, True, True, False]
    assert all(isinstance(value, bool)
               for value in instance_df['TARGET'].tolist())

    payload = json.loads(json.dumps(_binary_payload(context)))
    table = payload['context']['instance_table']
    target_index = table['columns'].index('TARGET')
    target_values = [row[target_index] for row in table['rows']]
    assert target_values == [False, True]
    assert all(isinstance(value, bool) for value in target_values)
    assert payload['task']['target']['dtype'] == 'bool'
    assert 'TARGET' not in payload['predict']['instance_table']['columns']


def test_binary_boolean_targets_remain_booleans() -> None:
    context = _binary_context(
        y_train=pd.Series([False, True], name='TARGET', dtype='bool'),
        y_test=pd.Series([True], name='TARGET', dtype='bool'),
    )

    instance_df, _, _ = _instance_dataframe(context)

    values = instance_df['TARGET'].tolist()
    assert values == [False, True, True]
    assert all(isinstance(value, bool) for value in values)


@pytest.mark.parametrize('invalid_value', [2, -1, 0.5, 'true'])
def test_binary_target_rejects_values_outside_explicit_mapping(
    invalid_value: Any,
) -> None:
    context = _binary_context(
        y_train=pd.Series([0, invalid_value], name='TARGET', dtype='object'),
        y_test=None,
    )

    with pytest.raises(
        ValueError,
        match='must be booleans or numeric 0/1 values',
    ):
        _binary_payload(context)


def test_binary_prediction_target_rejects_invalid_numeric_value() -> None:
    context = _binary_context(
        y_train=pd.Series([0, 1], name='TARGET', dtype='int64'),
        y_test=pd.Series([2], name='TARGET', dtype='int64'),
    )

    with pytest.raises(
        ValueError,
        match=r'numeric 0/1 values, but got 2\.',
    ):
        _binary_payload(context)

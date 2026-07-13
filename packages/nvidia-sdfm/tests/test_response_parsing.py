from __future__ import annotations

import pytest

from nvidia_sdfm.core.response import parse_prediction_response
from nvidia_sdfm.errors import SdfmError


def test_parse_prediction_response_basic_fields():
    response = {
        'id': 'pred_abc',
        'model': 'tabicl',
        'predictions': [
            {'row_index': 0, 'prediction': 'yes', 'probabilities': {'yes': 0.9, 'no': 0.1}},
            {'row_index': 1, 'prediction': 'no', 'probabilities': {'yes': 0.2, 'no': 0.8}},
        ],
        'metadata': {'task_kind': 'classification'},
    }
    frame = parse_prediction_response(response)
    assert list(frame['row_index']) == [0, 1]
    assert list(frame['prediction']) == ['yes', 'no']
    assert frame.loc[0, 'probabilities'] == {'yes': 0.9, 'no': 0.1}


def test_parse_prediction_response_sorts_out_of_order_rows():
    response = {
        'predictions': [
            {'row_index': 2, 'prediction': 'c'},
            {'row_index': 0, 'prediction': 'a'},
            {'row_index': 1, 'prediction': 'b'},
        ],
    }
    frame = parse_prediction_response(response)
    assert list(frame['row_index']) == [0, 1, 2]
    assert list(frame['prediction']) == ['a', 'b', 'c']


def test_parse_prediction_response_drops_all_null_columns():
    response = {
        'predictions': [
            {'row_index': 0, 'prediction': 'a'},
            {'row_index': 1, 'prediction': 'b'},
        ],
    }
    frame = parse_prediction_response(response)
    assert 'quantiles' not in frame.columns
    assert 'embeddings' not in frame.columns


def test_parse_prediction_response_quantiles():
    response = {
        'predictions': [
            {'row_index': 0, 'quantiles': {'0.1': 1.0, '0.5': 2.0, '0.9': 3.0}},
        ],
    }
    frame = parse_prediction_response(response)
    assert frame.loc[0, 'quantiles'] == {'0.1': 1.0, '0.5': 2.0, '0.9': 3.0}


def test_parse_prediction_response_missing_predictions_key_raises():
    with pytest.raises(SdfmError):
        parse_prediction_response({'id': 'pred_abc'})


def test_parse_prediction_response_non_list_predictions_raises():
    with pytest.raises(SdfmError) as excinfo:
        parse_prediction_response({'predictions': {'row_index': 0}})
    assert excinfo.value.code == 'INVALID_RESPONSE'


def test_parse_prediction_response_non_dict_item_raises():
    with pytest.raises(SdfmError) as excinfo:
        parse_prediction_response({'predictions': ['not-an-object']})
    assert excinfo.value.code == 'INVALID_RESPONSE'


def test_parse_prediction_response_missing_row_index_raises():
    with pytest.raises(SdfmError):
        parse_prediction_response({'predictions': [{'prediction': 'a'}]})


def test_parse_prediction_response_empty_predictions():
    frame = parse_prediction_response({'predictions': []})
    assert len(frame) == 0

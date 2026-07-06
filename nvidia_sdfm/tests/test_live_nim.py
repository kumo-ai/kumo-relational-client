from __future__ import annotations

import os

import pandas as pd
import pytest

import nvidia_sdfm

_ENV_VAR = 'SDFM_NIM_BASE_URL'

pytestmark = [
    pytest.mark.live_nim,
    pytest.mark.skipif(
        not os.environ.get(_ENV_VAR),
        reason=f'set {_ENV_VAR} to run live TFM NIM tests',
    ),
]


@pytest.fixture(scope='module')
def nim_url() -> str:
    return os.environ[_ENV_VAR].rstrip('/')


@pytest.fixture(scope='module', autouse=True)
def _init_sdk(nim_url: str):
    nvidia_sdfm.init(url=nim_url)


def test_nim_reports_ready(nim_url: str):
    from nvidia_sdfm.core.transport import TFMClient

    assert TFMClient(nim_url).health_ready() is True


def test_tabicl_predict_returns_expected_shape():
    context = pd.DataFrame({
        'row_id': [f'ctx-{i}' for i in range(20)],
        'age': [20 + i for i in range(20)],
        'score': [0.05 * i for i in range(20)],
        'target_col': ['yes' if i % 2 == 0 else 'no' for i in range(20)],
    })
    predict = pd.DataFrame({
        'row_id': ['q-0', 'q-1'],
        'age': [33, 49],
        'score': [0.72, 0.30],
    })

    frame = nvidia_sdfm.predict(
        model='tabicl',
        context=context,
        predict=predict,
        task='classification',
        target='target_col',
        outputs=['prediction', 'probabilities'],
    )

    assert len(frame) == 2
    assert list(frame['row_index']) == [0, 1]
    assert set(frame['prediction']).issubset({'yes', 'no'})
    for probabilities in frame['probabilities']:
        assert set(probabilities) == {'yes', 'no'}
        assert abs(sum(probabilities.values()) - 1.0) < 1e-3


def test_tabicl_predict_regression_returns_quantiles():
    context = pd.DataFrame({
        'row_id': [f'ctx-{i}' for i in range(10)],
        'feature_a': [float(i) for i in range(10)],
        'target_col': [float(i) * 2.0 + 1.0 for i in range(10)],
    })
    predict = pd.DataFrame({
        'row_id': ['q-0'],
        'feature_a': [4.5],
    })

    frame = nvidia_sdfm.predict(
        model='tabicl',
        context=context,
        predict=predict,
        task='regression',
        target='target_col',
        outputs=['prediction', 'quantiles'],
        prediction_statistic='mean',
        quantile_levels=[0.1, 0.5, 0.9],
    )

    assert len(frame) == 1
    assert 'quantiles' in frame.columns

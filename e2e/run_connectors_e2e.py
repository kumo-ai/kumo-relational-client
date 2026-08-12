# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        '..',
        'packages',
        'nemotron-predict-client',
        'src',
    ),
)
import nemotron_predict
from nemotron_predict import PredictClient

_client: PredictClient | None = None


def predict_tabicl(*, context, predict, task, target, **kwargs):
    assert _client is not None
    handle = _client.tabular(context, target=target, task=task)
    return handle.predict(predict, **kwargs)


BASE_URL = os.environ.get('NEMOTRON_PREDICT_NIM_BASE_URL', '').rstrip('/')
DATA_DIR = os.environ.get(
    'NEMOTRON_PREDICT_DATA_DIR', os.path.join(os.path.dirname(__file__), 'data')
)

RESULTS: list[tuple[str, bool, str]] = []

JOB_OUTCOME_FEATURES = [
    'actual_runtime_hours',
    'queue_wait_minutes',
    'mean_mfu_pct',
    'gpu_hours',
    'preempted_count',
]
JOB_OUTCOME_CLASSES = {
    'completed',
    'failed',
    'preempted',
    'timeout',
    'cancelled',
}


def check(name: str, condition: bool, detail: str = '') -> None:
    RESULTS.append((name, bool(condition), detail))
    print(f'[{"PASS" if condition else "FAIL"}] {name} {detail}')


def section(title: str) -> None:
    print()
    print('=' * 10, title, '=' * 10)


def classify_job_outcomes(frame: pd.DataFrame, source: str) -> None:
    columns = [c for c in JOB_OUTCOME_FEATURES if c in frame.columns] + [
        'status'
    ]
    frame = frame[columns].dropna(subset=['status']).reset_index(drop=True)
    frame = frame.sample(frac=1.0, random_state=11).reset_index(drop=True)
    context = frame.iloc[:2000]
    predict_full = frame.iloc[2000:2300]
    predict = predict_full.drop(columns=['status']).reset_index(drop=True)

    result = predict_tabicl(
        context=context,
        predict=predict,
        task='classification',
        target='status',
        outputs=['prediction', 'probabilities'],
    )
    check(
        f'{source}: row count matches',
        len(result) == len(predict),
        f'got {len(result)}',
    )
    check(
        f'{source}: predictions are valid job statuses',
        set(result['prediction']).issubset(JOB_OUTCOME_CLASSES),
        str(set(result['prediction'])),
    )
    majority = predict_full['status'].value_counts(normalize=True).max()
    accuracy = float(
        (result['prediction'].values == predict_full['status'].values).mean()
    )
    check(
        f'{source}: accuracy in classical-baseline band (>=0.5; RF gets ~0.62 here, '
        f'signal in this dataset is deliberately relational, not row-local)',
        accuracy >= 0.5,
        f'accuracy={accuracy:.3f} majority={majority:.3f}',
    )


def local_csv_connector() -> None:
    section('local connector: job_outcomes.csv -> tabicl classification')
    frame = nemotron_predict.read(
        'local', path=os.path.join(DATA_DIR, 'job_outcomes.csv')
    )
    check(
        'local: read returns DataFrame',
        isinstance(frame, pd.DataFrame),
        f'shape={frame.shape}',
    )
    classify_job_outcomes(frame, 'local')


def sqlite_connector() -> None:
    section('sqlite connector: gpu_metrics_daily -> tabicl regression')
    frame = nemotron_predict.read(
        'sqlite',
        database=os.path.join(DATA_DIR, 'gpu_fleet.sqlite'),
        query='SELECT avg_util_pct, avg_sm_active_pct, avg_tensor_active_pct, '
        'avg_mem_util_pct, avg_temp_c, avg_power_w FROM gpu_metrics_daily '
        'WHERE avg_power_w IS NOT NULL',
    )
    check(
        'sqlite: read returns DataFrame',
        isinstance(frame, pd.DataFrame),
        f'shape={frame.shape}',
    )

    frame = (
        frame.dropna().sample(frac=1.0, random_state=11).reset_index(drop=True)
    )
    context = frame.iloc[:2000]
    predict_full = frame.iloc[2000:2200]
    predict = predict_full.drop(columns=['avg_power_w']).reset_index(drop=True)

    result = predict_tabicl(
        context=context,
        predict=predict,
        task='regression',
        target='avg_power_w',
        outputs=['prediction'],
    )
    check(
        'sqlite: row count matches',
        len(result) == len(predict),
        f'got {len(result)}',
    )
    correlation = float(
        pd.Series(result['prediction'].values).corr(
            pd.Series(predict_full['avg_power_w'].values)
        )
    )
    check(
        'sqlite: regression predictions correlate with actual power',
        correlation > 0.5,
        f'pearson={correlation:.3f}',
    )


def duckdb_connector() -> None:
    section('duckdb connector: gpu_allocations -> tabicl regression')
    frame = nemotron_predict.read(
        'duckdb',
        database=os.path.join(DATA_DIR, 'gpu_fleet_smoke.duckdb'),
        query='SELECT gpu_hours, allocation_type, mig_profile, '
        'EXTRACT(EPOCH FROM released_at - allocated_at) / 3600.0 AS wall_hours '
        'FROM gpu_allocations WHERE released_at IS NOT NULL',
    )
    check(
        'duckdb: read returns DataFrame',
        isinstance(frame, pd.DataFrame),
        f'shape={frame.shape}',
    )

    frame = frame.dropna(subset=['gpu_hours', 'wall_hours'])
    frame = frame.sample(frac=1.0, random_state=11).reset_index(drop=True)
    context = frame.iloc[:2000]
    predict_full = frame.iloc[2000:2200]
    predict = predict_full.drop(columns=['gpu_hours']).reset_index(drop=True)

    result = predict_tabicl(
        context=context,
        predict=predict,
        task='regression',
        target='gpu_hours',
        outputs=['prediction', 'quantiles'],
        prediction_statistic='mean',
        quantile_levels=[0.1, 0.5, 0.9],
    )
    check(
        'duckdb: row count matches',
        len(result) == len(predict),
        f'got {len(result)}',
    )
    correlation = float(
        pd.Series(result['prediction'].values).corr(
            pd.Series(predict_full['gpu_hours'].values)
        )
    )
    check(
        'duckdb: regression predictions correlate with actual gpu_hours',
        correlation > 0.5,
        f'pearson={correlation:.3f}',
    )
    ordered = result['quantiles'].apply(
        lambda q: q['0.1'] <= q['0.5'] <= q['0.9']
    )
    check('duckdb: quantiles are ordered', bool(ordered.all()))


def snowflake_connector() -> None:
    section(
        'snowflake connector: full-scale JOB_OUTCOMES -> tabicl classification'
    )
    frame = nemotron_predict.read(
        'snowflake',
        account=os.environ['SNOWFLAKE_ACCOUNT'],
        user=os.environ['SNOWFLAKE_USER'],
        password=os.environ['SNOWFLAKE_PASSWORD'],
        role=os.environ['SNOWFLAKE_ROLE'],
        warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
        query='SELECT ACTUAL_RUNTIME_HOURS, QUEUE_WAIT_MINUTES, MEAN_MFU_PCT, '
        'GPU_HOURS, PREEMPTED_COUNT, STATUS '
        'FROM MY_DATABASE.GPU_FLEET.JOB_OUTCOMES SAMPLE (5000 ROWS)',
    )
    check(
        'snowflake: read returns DataFrame',
        isinstance(frame, pd.DataFrame),
        f'shape={frame.shape}',
    )
    frame.columns = [c.lower() for c in frame.columns]
    classify_job_outcomes(frame, 'snowflake')


def main() -> None:
    if not BASE_URL:
        print('Set NEMOTRON_PREDICT_NIM_BASE_URL to run the connector checks.')
        sys.exit(2)
    print(f'Target NIM: {BASE_URL}')
    global _client
    _client = PredictClient(url=BASE_URL)

    for step in (
        local_csv_connector,
        sqlite_connector,
        duckdb_connector,
        snowflake_connector,
    ):
        try:
            step()
        except Exception as error:
            check(
                step.__name__, False, f'raised {type(error).__name__}: {error}'
            )

    section('summary')
    failed = [name for name, ok, _ in RESULTS if not ok]
    for name, ok, detail in RESULTS:
        print(f'{"PASS" if ok else "FAIL"}  {name}  {detail}')
    print(f'\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed')
    if failed:
        print('FAILED CHECKS:', failed)
        sys.exit(1)


if __name__ == '__main__':
    main()

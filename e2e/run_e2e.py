# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import uuid

import numpy as np
import pandas as pd
import requests
from sklearn.datasets import load_breast_cancer, load_wine

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'nvidia-sdfm', 'src'))
from nvidia_sdfm import SDFMClient, TabICLRequest  # noqa: E402

_client: SDFMClient | None = None


def predict_tabicl(**kwargs):
    assert _client is not None
    return _client.predict(TabICLRequest(**kwargs))

BASE_URL = os.environ.get('SDFM_NIM_BASE_URL', '').rstrip('/')

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = '') -> None:
    RESULTS.append((name, ok, detail))
    status = 'PASS' if ok else 'FAIL'
    print(f'[{status}] {name} {detail}')


def check(name: str, condition: bool, detail: str = '') -> None:
    record(name, bool(condition), detail)


def section(title: str) -> None:
    print()
    print('=' * 10, title, '=' * 10)


def contract_compliance_regressions() -> None:
    section('contract compliance (regression checks for the 4 fixed gaps)')
    body = {
        'model': 'tabicl',
        'task': {'kind': 'classification', 'target': {'column_name': 'label', 'dtype': 'string'}},
        'schema': {'instance_table': {'columns': {
            'x1': {'dtype': 'float64'}, 'x2': {'dtype': 'float64'}, 'label': {'dtype': 'string'},
        }}},
        'context': {'instance_table': {
            'format': 'arrays', 'columns': ['x1', 'x2', 'label'],
            'rows': [[0.1, 0.2, 'a'], [0.9, 0.8, 'b']],
        }},
        'predict': {'instance_table': {
            'format': 'arrays', 'columns': ['x1', 'x2'], 'rows': [[0.5, 0.5]],
        }},
        'output': {'fields': ['prediction']},
    }
    response = requests.post(BASE_URL + '/v1/predictions', json=body, timeout=30)
    item = response.json()['predictions'][0]
    check(
        'FIXED: PredictionItem includes row_index',
        item.get('row_index') == 0,
        f'item={item}',
    )

    timestamp_body = {
        'model': 'tabicl',
        'task': {'kind': 'classification', 'target': {'column_name': 'label', 'dtype': 'string'}},
        'schema': {'instance_table': {'columns': {
            'ts': {'dtype': 'timestamp[us]'}, 'label': {'dtype': 'string'},
        }}},
        'context': {'instance_table': {
            'format': 'arrays', 'columns': ['ts', 'label'],
            'rows': [['2025-01-01T00:00:00.000Z', 'a'], ['2025-01-02T00:00:00.000Z', 'b']],
        }},
        'predict': {'instance_table': {
            'format': 'arrays', 'columns': ['ts'], 'rows': [['2025-01-03T00:00:00.000Z']],
        }},
        'output': {'fields': ['prediction']},
    }
    ts_response = requests.post(BASE_URL + '/v1/predictions', json=timestamp_body, timeout=30)
    check(
        'FIXED: timestamp[us] dtype is accepted',
        ts_response.status_code == 200,
        f'status={ts_response.status_code} body={ts_response.text[:200]}',
    )

    malformed_ts_body = {
        **timestamp_body,
        'context': {'instance_table': {
            'format': 'arrays', 'columns': ['ts', 'label'],
            'rows': [['not-a-timestamp', 'a'], ['2025-01-02T00:00:00.000Z', 'b']],
        }},
    }
    malformed_response = requests.post(BASE_URL + '/v1/predictions', json=malformed_ts_body, timeout=30)
    check(
        'FIXED: malformed timestamp[us] cell is still rejected (not silently accepted)',
        malformed_response.status_code in (400, 422),
        f'status={malformed_response.status_code} body={malformed_response.text[:200]}',
    )

    session_body = {
        'model': 'tabicl',
        'task': {'kind': 'classification', 'target': {'column_name': 'label', 'dtype': 'string'}},
        'schema': {'instance_table': {'columns': {
            'x1': {'dtype': 'float64'}, 'label': {'dtype': 'string'},
        }}},
        'context': {'instance_table': {
            'format': 'arrays', 'columns': ['x1', 'label'], 'rows': [[0.1, 'a'], [0.9, 'b']],
        }},
    }
    session_response = requests.post(BASE_URL + '/v1/sessions', json=session_body, timeout=30)
    check(
        'FIXED: POST /v1/sessions returns 201 with a Location header',
        session_response.status_code == 201 and 'Location' in session_response.headers,
        f'status={session_response.status_code} headers={dict(session_response.headers)}',
    )
    if session_response.status_code == 201:
        session_id = session_response.json().get('session_id')
        check(
            'FIXED: Location header points at the session resource',
            session_response.headers.get('Location') == f'/v1/sessions/{session_id}',
            session_response.headers.get('Location'),
        )
        delete_response = requests.delete(f'{BASE_URL}/v1/sessions/{session_id}', timeout=30)
        check(
            'FIXED: DELETE /v1/sessions/{id} returns 204 with an empty body',
            delete_response.status_code == 204 and not delete_response.content,
            f'status={delete_response.status_code} body={delete_response.content!r}',
        )


def management_endpoints() -> None:
    section('management endpoints')
    for path in (
        '/v1/health/live',
        '/v1/health/ready',
        '/v1/metadata',
        '/v1/manifest',
        '/v1/license',
        '/v1/models',
        '/v1/metrics',
    ):
        response = requests.get(BASE_URL + path, timeout=30)
        check(f'GET {path}', response.status_code == 200, f'status={response.status_code}')
        if path == '/v1/metrics' and response.status_code == 200:
            check('metrics contain tabicl_ series', 'tabicl_' in response.text)


def sklearn_binary_classification() -> None:
    section('sklearn breast cancer: binary classification')
    data = load_breast_cancer(as_frame=True)
    frame = data.frame.rename(columns={'target': 'label'})
    frame['label'] = frame['label'].map({0: 'malignant', 1: 'benign'})
    context = frame.iloc[:400].reset_index(drop=True)
    predict = frame.iloc[400:420].drop(columns=['label']).reset_index(drop=True)

    result = predict_tabicl(
        context=context, predict=predict,
        task='classification', target='label',
        outputs=['prediction', 'probabilities'],
    )
    check('row count matches predict set', len(result) == len(predict), f'got {len(result)}')
    check(
        'predictions are valid classes',
        set(result['prediction']).issubset({'malignant', 'benign'}),
        str(set(result['prediction'])),
    )
    probability_sums = result['probabilities'].apply(lambda p: sum(p.values()))
    check(
        'probabilities sum to ~1',
        bool((probability_sums.sub(1.0).abs() < 1e-3).all()),
        str(probability_sums.tolist()),
    )


def sklearn_multiclass_classification() -> None:
    section('sklearn wine: multiclass classification')
    data = load_wine(as_frame=True)
    frame = data.frame.rename(columns={'target': 'label'})
    frame['label'] = frame['label'].astype(str)
    context = frame.iloc[:140].reset_index(drop=True)
    predict = frame.iloc[140:].drop(columns=['label']).reset_index(drop=True)

    result = predict_tabicl(
        context=context, predict=predict,
        task='classification', target='label',
        outputs=['prediction', 'probabilities'],
    )
    check('row count matches predict set', len(result) == len(predict), f'got {len(result)}')
    check(
        'predictions are valid classes',
        set(result['prediction']).issubset(set(frame['label'].unique())),
        str(set(result['prediction'])),
    )


def sklearn_regression_with_quantiles() -> None:
    section('synthetic regression with quantiles')
    rng = np.random.default_rng(42)
    n = 300
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(-5, 5, n)
    target = 3.0 * x1 - 2.0 * x2 + rng.normal(0, 0.5, n)
    frame = pd.DataFrame({'x1': x1, 'x2': x2, 'target': target})
    context = frame.iloc[:250].reset_index(drop=True)
    predict = frame.iloc[250:].drop(columns=['target']).reset_index(drop=True)

    result = predict_tabicl(
        context=context, predict=predict,
        task='regression', target='target',
        outputs=['prediction', 'quantiles'],
        prediction_statistic='mean',
        quantile_levels=[0.1, 0.5, 0.9],
    )
    check('row count matches predict set', len(result) == len(predict), f'got {len(result)}')
    check('quantiles present', 'quantiles' in result.columns)
    if 'quantiles' in result.columns:
        ordered = result['quantiles'].apply(
            lambda q: q['0.1'] <= q['0.5'] <= q['0.9'],
        )
        check('quantiles are monotonically ordered', bool(ordered.all()))


def real_dataset_mixed_dtypes() -> None:
    section('UCI Adult Income: real dataset, mixed dtypes, large int64 ids, timestamps')
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
        'income',
    ]
    path = os.path.join(os.path.dirname(__file__), 'adult.data')
    frame = pd.read_csv(path, names=columns, skipinitialspace=True).dropna()
    frame = frame[frame['workclass'] != '?'].reset_index(drop=True)
    frame['income'] = frame['income'].str.strip()

    huge_base = 9_007_199_254_740_991 + 1_000_000
    frame['row_id'] = [huge_base + i for i in range(len(frame))]
    frame['event_time'] = pd.date_range('2025-01-01', periods=len(frame), freq='min', tz='UTC')

    sample = frame.sample(n=3000, random_state=7).reset_index(drop=True)
    context = sample.iloc[:2500]
    predict_full = sample.iloc[2500:]
    predict = predict_full.drop(columns=['income']).reset_index(drop=True)

    from nvidia_sdfm.adapters.tabicl import build_request
    payload = build_request(
        context=context, predict=predict, task='classification', target='income',
        outputs=['prediction', 'probabilities'],
    )
    huge_id_sent = payload['context']['instance_table']['rows'][0][
        payload['context']['instance_table']['columns'].index('row_id')
    ]
    check(
        'huge int64 id was sent as a JSON string (safe-int guard)',
        isinstance(huge_id_sent, str),
        f'row_id sent as {type(huge_id_sent).__name__}: {huge_id_sent}',
    )

    result = predict_tabicl(
        context=context, predict=predict,
        task='classification', target='income',
        outputs=['prediction', 'probabilities'],
    )
    check('row count matches predict set', len(result) == len(predict), f'got {len(result)}')
    check(
        'predictions are valid income classes',
        set(result['prediction']).issubset({'<=50K', '>50K'}),
        str(set(result['prediction'])),
    )
    accuracy = float((result['prediction'].values == predict_full['income'].values).mean())
    check('accuracy is above trivial baseline (>0.5)', accuracy > 0.5, f'accuracy={accuracy:.3f}')


def sessions_lifecycle() -> None:
    section('sessions: create -> reuse -> delete -> predict-after-delete')
    body = {
        'model': 'tabicl',
        'task': {'kind': 'classification', 'target': {'column_name': 'label', 'dtype': 'string'}},
        'schema': {
            'instance_table': {
                'columns': {
                    'x1': {'dtype': 'float64'},
                    'x2': {'dtype': 'float64'},
                    'label': {'dtype': 'string'},
                },
            },
        },
        'context': {
            'instance_table': {
                'format': 'arrays',
                'columns': ['x1', 'x2', 'label'],
                'rows': [[0.1, 0.2, 'a'], [0.9, 0.8, 'b'], [0.15, 0.1, 'a'], [0.85, 0.9, 'b']],
            },
        },
    }
    create_response = requests.post(BASE_URL + '/v1/sessions', json=body, timeout=30)
    check(
        'session create returns 201',
        create_response.status_code == 201,
        f'status={create_response.status_code} body={create_response.text[:200]}',
    )
    if create_response.status_code != 201:
        return
    session_id = create_response.json()['session_id']

    predict_body = {
        'predict': {
            'instance_table': {
                'format': 'arrays', 'columns': ['x1', 'x2'], 'rows': [[0.88, 0.86]],
            },
        },
        'output': {'fields': ['prediction', 'probabilities']},
    }
    predict_response = requests.post(
        f'{BASE_URL}/v1/sessions/{session_id}/predictions', json=predict_body, timeout=30,
    )
    check('session predict returns 200', predict_response.status_code == 200,
          f'status={predict_response.status_code}')

    delete_response = requests.delete(f'{BASE_URL}/v1/sessions/{session_id}', timeout=30)
    check('session delete returns 204', delete_response.status_code == 204,
          f'status={delete_response.status_code}')

    after_delete_response = requests.post(
        f'{BASE_URL}/v1/sessions/{session_id}/predictions', json=predict_body, timeout=30,
    )
    check(
        'predict-after-delete returns 404 SESSION_NOT_FOUND',
        after_delete_response.status_code == 404,
        f'status={after_delete_response.status_code} body={after_delete_response.text[:200]}',
    )


def error_model() -> None:
    section('error model')
    unsupported_task = {
        'model': 'tabicl',
        'task': {'kind': 'forecasting', 'target': {'column_name': 'label', 'dtype': 'string'}},
        'schema': {'instance_table': {'columns': {'x': {'dtype': 'float64'}, 'label': {'dtype': 'string'}}}},
        'context': {'instance_table': {'format': 'arrays', 'columns': ['x', 'label'], 'rows': [[1.0, 'a']]}},
        'predict': {'instance_table': {'format': 'arrays', 'columns': ['x'], 'rows': [[1.0]]}},
        'output': {'fields': ['prediction']},
    }
    response = requests.post(BASE_URL + '/v1/predictions', json=unsupported_task, timeout=30)
    check(
        'unsupported task kind returns 4xx',
        400 <= response.status_code < 500,
        f'status={response.status_code} body={response.text[:200]}',
    )
    if response.status_code < 500:
        body = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
        check('error body has no raw row values', '"a"' not in json.dumps(body))

    malformed_response = requests.post(
        BASE_URL + '/v1/predictions', data='not json', timeout=30,
        headers={'Content-Type': 'application/json'},
    )
    check('malformed JSON returns 400', malformed_response.status_code == 400,
          f'status={malformed_response.status_code}')

    unknown_session_response = requests.post(
        f'{BASE_URL}/v1/sessions/{uuid.uuid4()}/predictions',
        json={'predict': {'instance_table': {'format': 'arrays', 'columns': ['x'], 'rows': [[1.0]]}},
              'output': {'fields': ['prediction']}},
        timeout=30,
    )
    check('unknown session id returns 404', unknown_session_response.status_code == 404,
          f'status={unknown_session_response.status_code}')


def admission_control() -> None:
    section('admission control: oversized context should reject cleanly, not crash')
    rng = np.random.default_rng(0)
    n_rows, n_features = 2_000, 200
    columns = {f'f{i}': {'dtype': 'float64'} for i in range(n_features)}
    columns['label'] = {'dtype': 'string'}
    body = {
        'model': 'tabicl',
        'task': {'kind': 'classification', 'target': {'column_name': 'label', 'dtype': 'string'}},
        'schema': {'instance_table': {'columns': columns}},
        'context': {
            'instance_table': {
                'format': 'arrays',
                'columns': list(columns),
                'rows': [
                    [*rng.uniform(-1, 1, n_features).tolist(), 'a' if i % 2 == 0 else 'b']
                    for i in range(n_rows)
                ],
            },
        },
        'predict': {
            'instance_table': {
                'format': 'arrays',
                'columns': [c for c in columns if c != 'label'],
                'rows': [rng.uniform(-1, 1, n_features).tolist()],
            },
        },
        'output': {'fields': ['prediction']},
    }
    try:
        response = requests.post(BASE_URL + '/v1/predictions', json=body, timeout=120)
        check(
            'oversized request returns 503 OOM or succeeds without crashing',
            response.status_code in (200, 503),
            f'status={response.status_code}',
        )
    except requests.RequestException as error:
        check('oversized request does not crash the connection', False, str(error))

    ready_response = requests.get(BASE_URL + '/v1/health/ready', timeout=30)
    check('service still ready after oversized request', ready_response.status_code == 200)


def concurrency() -> None:
    section('concurrency: N parallel requests')
    body = {
        'model': 'tabicl',
        'task': {'kind': 'classification', 'target': {'column_name': 'label', 'dtype': 'string'}},
        'schema': {'instance_table': {'columns': {
            'x1': {'dtype': 'float64'}, 'x2': {'dtype': 'float64'}, 'label': {'dtype': 'string'},
        }}},
        'context': {'instance_table': {
            'format': 'arrays', 'columns': ['x1', 'x2', 'label'],
            'rows': [[0.1, 0.2, 'a'], [0.9, 0.8, 'b'], [0.15, 0.1, 'a'], [0.85, 0.9, 'b']],
        }},
        'predict': {'instance_table': {
            'format': 'arrays', 'columns': ['x1', 'x2'], 'rows': [[0.5, 0.5]],
        }},
        'output': {'fields': ['prediction']},
    }

    def _call(_: int) -> int:
        response = requests.post(BASE_URL + '/v1/predictions', json=body, timeout=60)
        return response.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        statuses = list(executor.map(_call, range(20)))
    check(
        'all concurrent requests return 200 or clean 503',
        all(status in (200, 503) for status in statuses),
        str(statuses),
    )
    ready_response = requests.get(BASE_URL + '/v1/health/ready', timeout=30)
    check('service still ready after concurrency burst', ready_response.status_code == 200)


def main() -> None:
    if not BASE_URL:
        print('Set SDFM_NIM_BASE_URL to run the e2e checks.')
        sys.exit(2)
    print(f'Target NIM: {BASE_URL}')
    global _client
    _client = SDFMClient(url=BASE_URL)

    for step in (
        management_endpoints,
        contract_compliance_regressions,
        sklearn_binary_classification,
        sklearn_multiclass_classification,
        sklearn_regression_with_quantiles,
        real_dataset_mixed_dtypes,
        sessions_lifecycle,
        error_model,
        admission_control,
        concurrency,
    ):
        try:
            step()
        except Exception as error:  # noqa: BLE001
            record(step.__name__, False, f'raised {type(error).__name__}: {error}')

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

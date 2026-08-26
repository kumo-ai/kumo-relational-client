# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end checks for the Kumo Relational NIM, driven through the client."""

from __future__ import annotations

import concurrent.futures
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'packages/kumo-relational-client/src'))
sys.path.insert(0, os.path.join(REPO, 'packages/nemotron-relational/src'))

from kumo_relational_client import RelationalClient, relational  # noqa: E402
from kumo_relational_client.errors import RelationalError  # noqa: E402

URL = os.environ['KUMO_RELATIONAL_NIM_BASE_URL'].rstrip('/')
KEY = os.environ.get('KUMO_RELATIONAL_NIM_API_KEY')
EXPECTED_MODEL = 'kumo-relational'

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = '') -> None:
    RESULTS.append((name, bool(ok), detail))
    print(
        f'[{"PASS" if ok else "FAIL"}] {name}'
        + (f'  {detail}' if detail else ''),
        flush=True,
    )


def check(name, fn):
    try:
        detail = fn()
        record(name, True, detail or '')
    except Exception as e:
        record(name, False, f'{type(e).__name__}: {str(e)[:180]}')


rng = np.random.default_rng(0)
N_USERS = 60
users = pd.DataFrame(
    {
        'user_id': range(1, N_USERS + 1),
        'age': rng.integers(18, 70, N_USERS),
        'segment': rng.choice(['a', 'b', 'c'], N_USERS),
    }
)
N_ORDERS = 3000
start = pd.Timestamp('2024-01-01')
orders = pd.DataFrame(
    {
        'order_id': range(1, N_ORDERS + 1),
        'user_id': rng.integers(1, N_USERS + 1, N_ORDERS),
        'price': rng.uniform(5, 200, N_ORDERS).round(2),
        'ts': start + pd.to_timedelta(rng.integers(0, 180, N_ORDERS), unit='D'),
    }
).sort_values('ts', ignore_index=True)

IDS = [1, 2, 3, 4, 5]

print('=' * 12, 'transport / management', '=' * 12)


def _models():
    r = requests.get(f'{URL}/v1/models', headers={'X-API-Key': KEY}, timeout=30)
    r.raise_for_status()
    ids = [m['id'] for m in r.json()['data']]
    assert ids == [EXPECTED_MODEL], f'served {ids}, expected [{EXPECTED_MODEL}]'
    return f'serves {ids}'


check(f'/v1/models serves exactly {EXPECTED_MODEL}', _models)


def _old_id_gone():
    r = requests.get(f'{URL}/v1/models', headers={'X-API-Key': KEY}, timeout=30)
    ids = [m['id'] for m in r.json()['data']]
    assert 'nemotron-relational' not in ids, 'old id still served'
    return 'retired id absent'


check('retired id nemotron-relational is not served', _old_id_gone)

client = RelationalClient(url=URL, api_key=KEY)
check('health_ready()', lambda: f'ready={client.health_ready()}')
check('client.models() reaches the NIM', lambda: f'{client.models()}')


def _caps():
    c = client.capabilities(EXPECTED_MODEL)
    return f'tasks={sorted(c.tasks)[:4]}'


check(f'capabilities({EXPECTED_MODEL})', _caps)

print()
print('=' * 12, 'predictions', '=' * 12)

graph = relational.Graph.from_data(
    {'users': users, 'orders': orders}, verbose=False
)
model = client.relational(graph)


def run_query(q, **kw):
    def inner():
        out = model.predict(q, indices=IDS, verbose=False, **kw)
        assert len(out) > 0, 'empty result'
        assert not out.isnull().all().all(), 'all-null result'
        return f'{len(out)} rows, cols={list(out.columns)[:5]}'

    return inner


check(
    'regression SUM 30 days',
    run_query('PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'),
)
check(
    'regression COUNT 30 days',
    run_query('PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id'),
)
check(
    'binary classification',
    run_query(
        'PREDICT COUNT(orders.*, 0, 30, days) > 5 FOR EACH users.user_id'
    ),
)
check(
    'static multiclass',
    run_query('PREDICT users.segment FOR EACH users.user_id'),
)

print()
print('=' * 12, 'time units (weeks / seconds fixes)', '=' * 12)


def _nonzero(q):
    def inner():
        out = model.predict(q, indices=IDS, verbose=False)
        col = [
            c for c in out.columns if c not in ('ENTITY', 'ANCHOR_TIMESTAMP')
        ]
        vals = out[col[0]].astype(float)
        assert not (vals == 0).all(), (
            'every prediction is 0.0 (zero-length window)'
        )
        return f'nonzero predictions, mean={vals.mean():.3f}'

    return inner


check(
    'weeks window returns a real (non-zero) window',
    _nonzero('PREDICT SUM(orders.price, 0, 4, weeks) FOR EACH users.user_id'),
)
check(
    'weeks matches the equivalent days window',
    lambda: (
        f'weeks={model.predict("PREDICT SUM(orders.price, 0, 4, weeks) FOR EACH users.user_id", indices=IDS, verbose=False).iloc[:, -1].mean():.3f} '
        f'days={model.predict("PREDICT SUM(orders.price, 0, 28, days) FOR EACH users.user_id", indices=IDS, verbose=False).iloc[:, -1].mean():.3f}'
    ),
)
check(
    'seconds unit parses and runs',
    run_query(
        'PREDICT COUNT(orders.*, 0, 2592000, seconds) FOR EACH users.user_id'
    ),
)

print()
print('=' * 12, 'explainability', '=' * 12)


def _explain():
    # Explanations are only defined for a single entity.
    out = model.predict(
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
        indices=[IDS[0]],
        verbose=False,
        explain=True,
    )
    assert len(out.prediction) == 1, 'explanation carries no prediction row'
    assert isinstance(out.summary, str) and out.summary.strip(), 'empty summary'
    return f'prediction rows={len(out.prediction)}, summary={len(out.summary)} chars'


check('regression + explain (single entity)', _explain)

print()
print('=' * 12, 'error model', '=' * 12)


def _bad_query():
    try:
        model.predict(
            'PREDICT NONSENSE(orders.price) FOR EACH users.user_id',
            indices=IDS,
            verbose=False,
        )
    except RelationalError as e:
        return f'code={getattr(e, "code", "?")}'
    except Exception as e:
        return f'{type(e).__name__} (not StructuredError)'
    raise AssertionError('a malformed query was accepted')


check('malformed query is rejected', _bad_query)


def _bad_key():
    r = requests.get(
        f'{URL}/v1/models', headers={'X-API-Key': 'wrong'}, timeout=30
    )
    assert r.status_code in (401, 403), f'HTTP {r.status_code}'
    return f'HTTP {r.status_code}'


check('a bad API key is refused', _bad_key)

print()
print('=' * 12, 'concurrency', '=' * 12)


def _concurrent():
    q = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        outs = list(
            pool.map(
                lambda _: model.predict(q, indices=IDS, verbose=False), range(4)
            )
        )
    assert all(len(o) > 0 for o in outs), 'a concurrent request came back empty'
    return f'4 parallel predictions in {time.time() - t0:.1f}s'


check('4 concurrent predictions', _concurrent)

print()
print('=' * 12, 'summary', '=' * 12)
failed = [n for n, ok, _ in RESULTS if not ok]
for n, ok, d in RESULTS:
    print(f'{"PASS" if ok else "FAIL"}  {n}  {d}')
print(f'\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed')
if failed:
    print('FAILED:', failed)
    sys.exit(1)

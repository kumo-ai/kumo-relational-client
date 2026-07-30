# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import textwrap

_SCRIPT = textwrap.dedent(
    """
    import hashlib
    import json

    import numpy as np
    import pandas as pd

    from kumorfm.rfm import Graph, KumoRFM

    rng = np.random.default_rng(0)
    users = pd.DataFrame({'user_id': np.arange(12), 'age': rng.integers(18, 70, 12)})
    orders = pd.DataFrame({
        'order_id': np.arange(60),
        'user_id': rng.integers(0, 12, 60),
        'date': pd.to_datetime('2025-01-01') + pd.to_timedelta(
            rng.integers(0, 60, 60), unit='D'),
        'amount': rng.uniform(5, 100, 60).round(2),
    })
    graph = Graph.from_data({'users': users, 'orders': orders}, verbose=False)
    model = KumoRFM(graph, verbose=False)
    query = 'PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.user_id'
    task = model._get_task_table(model._parse_query(query), indices=[1, 2],
                                 random_seed=42)
    requests = model.materialize_task(task, random_seed=42, verbose=False)
    blob = json.dumps([r.payload for r in requests], sort_keys=True, default=str)
    print(hashlib.sha256(blob.encode()).hexdigest())
    """
)


def _payload_hash() -> str:
    result = subprocess.run([sys.executable, '-c', _SCRIPT], capture_output=True,
                            text=True, check=True)
    return result.stdout.strip().splitlines()[-1]


def test_payload_is_identical_across_processes() -> None:
    r"""A fixed ``random_seed`` must survive a fresh interpreter.

    Python randomizes string hashing per process, so iterating a ``set`` of
    column names leaks that ordering into the request. This has to run in
    subprocesses: within one process the ordering is stable and the regression
    is invisible.
    """
    assert _payload_hash() == _payload_hash()

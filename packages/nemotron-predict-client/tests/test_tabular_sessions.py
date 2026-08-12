# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import json
import threading
import time

import pandas as pd

from nemotron_predict import PredictClient
from nemotron_predict.adapters.tabular import (
    _predict_with_session,
    build_request,
)
from nemotron_predict.requests import NemotronTabularSession

_URL = 'http://nim.example.com:8000'
_SESSION_ID = 'sess_2c12f086caaa'


def _predictions() -> dict:
    return {
        'id': 'pred_1',
        'model': 'nemotron-tabular',
        'predictions': [
            {'row_index': 0, 'prediction': 'yes'},
            {'row_index': 1, 'prediction': 'no'},
        ],
        'metadata': {'task_kind': 'classification'},
    }


def _register_session_nim(requests_mock, session_id: str = _SESSION_ID):
    requests_mock.post(_URL + '/v1/predictions', json=_predictions())
    requests_mock.post(
        _URL + '/v1/sessions',
        json={'session_id': session_id, 'ttl_seconds': 3600},
        status_code=201,
    )
    requests_mock.post(
        f'{_URL}/v1/sessions/{session_id}/predictions', json=_predictions()
    )
    requests_mock.delete(f'{_URL}/v1/sessions/{session_id}', status_code=204)


def _paths(requests_mock) -> list[str]:
    return [f'{r.method} {r.path}' for r in requests_mock.request_history]


def _handle(client, context_df):
    return client.tabular(
        context_df, target='target_col', task='classification'
    )


def test_repeated_predict_pins_the_context_once(
    requests_mock, context_df, predict_df
):
    _register_session_nim(requests_mock)

    with PredictClient(url=_URL) as client:
        model = _handle(client, context_df)
        for _ in range(3):
            model.predict(predict_df)

    assert _paths(requests_mock) == [
        'POST /v1/predictions',
        'POST /v1/sessions',
        f'POST /v1/sessions/{_SESSION_ID}/predictions',
        f'POST /v1/sessions/{_SESSION_ID}/predictions',
    ]

    bodies = [r.json() for r in requests_mock.request_history]
    assert set(bodies[1]) == {'model', 'task', 'schema', 'context', 'metadata'}
    assert set(bodies[2]) == {'predict', 'output', 'metadata'}
    assert len(json.dumps(bodies[2])) < len(json.dumps(bodies[0])) / 2


def test_session_predictions_match_the_stateless_path(
    requests_mock, context_df, predict_df
):
    _register_session_nim(requests_mock)

    with PredictClient(url=_URL) as client:
        model = _handle(client, context_df)
        stateless = model.predict(predict_df)
        model.predict(predict_df)
        via_session = model.predict(predict_df)

    assert requests_mock.request_history[-1].path.endswith(
        f'/v1/sessions/{_SESSION_ID}/predictions'
    )
    pd.testing.assert_frame_equal(stateless, via_session)


def test_single_predict_never_opens_a_session(
    requests_mock, context_df, predict_df
):
    _register_session_nim(requests_mock)

    with PredictClient(url=_URL) as client:
        _handle(client, context_df).predict(predict_df)

    assert _paths(requests_mock) == ['POST /v1/predictions']


def test_nim_without_session_routes_stays_stateless(
    requests_mock, context_df, predict_df
):
    requests_mock.post(_URL + '/v1/predictions', json=_predictions())
    requests_mock.post(
        _URL + '/v1/sessions',
        status_code=404,
        json={'code': 'NOT_FOUND', 'detail': 'no such route'},
    )

    with PredictClient(url=_URL) as client:
        model = _handle(client, context_df)
        for _ in range(3):
            model.predict(predict_df)

    assert _paths(requests_mock) == [
        'POST /v1/predictions',
        'POST /v1/sessions',
        'POST /v1/predictions',
        'POST /v1/predictions',
    ]


def test_create_session_failure_falls_back_to_the_stateless_path(
    requests_mock, context_df, predict_df
):
    """A session is an optimisation. Failing the caller's prediction because the
    optimisation failed is the wrong trade, so any create failure -- not only
    the 404/405/501 that mean "no session routes here" -- serves the call
    statelessly. The 507 here is not a latching condition, so the next call
    tries again.
    """
    requests_mock.post(_URL + '/v1/predictions', json=_predictions())
    requests_mock.post(
        _URL + '/v1/sessions',
        status_code=507,
        json={'code': 'INSUFFICIENT_STORAGE', 'detail': 'full'},
    )

    with PredictClient(url=_URL) as client:
        model = _handle(client, context_df)
        model.predict(predict_df)
        result = model.predict(predict_df)

    assert list(result['prediction']) == ['yes', 'no']
    assert _paths(requests_mock) == [
        'POST /v1/predictions',
        'POST /v1/sessions',
        'POST /v1/predictions',
    ]


def test_transport_failure_on_create_falls_back_to_the_stateless_path(
    requests_mock, context_df, predict_df
):
    """A ``TRANSPORT_ERROR`` on create used to propagate and fail the prediction
    outright, even though the stateless path would have answered it.
    """
    requests_mock.post(_URL + '/v1/predictions', json=_predictions())
    requests_mock.post(_URL + '/v1/sessions', json={'ttl_seconds': 3600})

    with PredictClient(url=_URL) as client:
        model = _handle(client, context_df)
        model.predict(predict_df)
        result = model.predict(predict_df)

    assert list(result['prediction']) == ['yes', 'no']
    assert _paths(requests_mock)[-1] == 'POST /v1/predictions'


def test_expired_session_is_repinned_and_retried(
    requests_mock, context_df, predict_df
):
    _register_session_nim(requests_mock)
    requests_mock.post(
        f'{_URL}/v1/sessions/{_SESSION_ID}/predictions',
        [
            {
                'status_code': 404,
                'json': {'code': 'SESSION_NOT_FOUND', 'detail': 'gone'},
            },
            {'json': _predictions()},
        ],
    )

    with PredictClient(url=_URL) as client:
        model = _handle(client, context_df)
        model.predict(predict_df)
        frame = model.predict(predict_df)

    assert list(frame['prediction']) == ['yes', 'no']
    assert _paths(requests_mock) == [
        'POST /v1/predictions',
        'POST /v1/sessions',
        f'POST /v1/sessions/{_SESSION_ID}/predictions',
        'POST /v1/sessions',
        f'POST /v1/sessions/{_SESSION_ID}/predictions',
    ]


def test_a_changed_context_half_releases_the_old_session(
    requests_mock, context_df, predict_df
):
    _register_session_nim(requests_mock)

    with PredictClient(url=_URL) as client:
        model = _handle(client, context_df)
        model.predict(predict_df)
        model.predict(predict_df)
        # A string where the session pinned a float widens the wire dtype, so
        # the pinned schema no longer describes this request.
        model.predict(predict_df.astype({'score': str}))

    assert _paths(requests_mock)[-2:] == [
        f'DELETE /v1/sessions/{_SESSION_ID}',
        'POST /v1/predictions',
    ]


def test_dropping_the_handle_releases_the_session(
    requests_mock, context_df, predict_df
):
    _register_session_nim(requests_mock)

    client = PredictClient(url=_URL)
    model = _handle(client, context_df)
    model.predict(predict_df)
    model.predict(predict_df)

    # Dropping the last reference is what releases the session. CPython frees
    # it on the spot, but that is a refcounting detail rather than a promise,
    # so collect explicitly instead of letting the assertion depend on it.
    del model
    gc.collect()

    assert _paths(requests_mock)[-1] == f'DELETE /v1/sessions/{_SESSION_ID}'
    client.close()


def test_pinned_state_does_not_retain_a_copy_of_the_context(
    requests_mock, context_df, predict_df
):
    r"""The handle only has to detect that the pinned half changed, so it holds
    a digest. Keeping the sections themselves would park a second copy of the
    context in client memory for the life of the handle.
    """
    _register_session_nim(requests_mock)

    with PredictClient(url=_URL) as client:
        model = _handle(client, context_df)
        model.predict(predict_df)
        pinned = model._session.pinned

    assert isinstance(pinned, str)
    assert len(pinned) == 64
    # The context values must not be reachable from what the handle kept.
    assert 'ctx-0' not in pinned


def test_pinned_digest_changes_with_the_schema(context_df, predict_df):
    r"""The digest has to be sensitive to the same things the pinned context
    is, or a session would be reused for a request it cannot answer.
    """
    from nemotron_predict.adapters.tabular import _pinned_digest, build_request

    def digest(predict):
        return _pinned_digest(
            build_request(
                context=context_df,
                predict=predict,
                task='classification',
                target='target_col',
                outputs=['prediction'],
                request_id='fixed',
            )
        )

    baseline = digest(predict_df)

    assert digest(predict_df) == baseline
    # A widened dtype changes the schema the session pinned.
    assert digest(predict_df.astype({'score': str})) != baseline

    # ... as does a different target class list, via a different context.
    other_context = context_df.copy()
    other_context.loc[0, 'target_col'] = 'maybe'
    assert (
        _pinned_digest(
            build_request(
                context=other_context,
                predict=predict_df,
                task='classification',
                target='target_col',
                outputs=['prediction'],
                request_id='fixed',
            )
        )
        != baseline
    )


def test_pinned_digest_ignores_the_per_call_sections(context_df, predict_df):
    r"""`predict` / `output` / `metadata` travel with every call, so they must
    not invalidate the pinned context.
    """
    import pandas as pd

    from nemotron_predict.adapters.tabular import _pinned_digest, build_request

    first = _pinned_digest(
        build_request(
            context=context_df,
            predict=predict_df,
            task='classification',
            target='target_col',
            outputs=['prediction'],
            request_id='req-1',
        )
    )
    second = _pinned_digest(
        build_request(
            context=context_df,
            predict=pd.DataFrame(
                {'row_id': ['q-9'], 'age': [41], 'score': [0.5]}
            ),
            task='classification',
            target='target_col',
            outputs=['prediction', 'probabilities'],
            request_id='req-2',
        )
    )

    assert first == second


class _CountingTransport:
    r"""Stands in for a NIM whose session-create is slow enough to overlap."""

    url = _URL

    def __init__(self, create_delay: float = 0.2) -> None:
        self.created: list[str] = []
        self.deleted: list[str] = []
        self.stateless = 0
        self._create_delay = create_delay
        self._lock = threading.Lock()

    def predict(self, payload):
        with self._lock:
            self.stateless += 1
        return _predictions()

    def create_session(self, payload):
        with self._lock:
            session_id = f's{len(self.created)}'
            self.created.append(session_id)
        time.sleep(self._create_delay)
        return {'session_id': session_id}

    def session_predict(self, session_id, payload):
        return _predictions()

    def delete_session(self, session_id):
        self.deleted.append(session_id)


def test_concurrent_predicts_on_one_handle_open_one_session(
    context_df, predict_df
):
    """``NemotronTabularSession`` is mutable state on a handle the docstring recommends
    reusing. Without a lock, every thread reads ``id is None`` at once, they all
    create, the last writer wins, and the rest are pinned on the NIM with no id
    left to release them by -- silently, and paid for by whoever calls next.
    """
    payload = build_request(
        context=context_df,
        predict=predict_df,
        task='classification',
        target='target_col',
        outputs=['prediction'],
    )
    transport = _CountingTransport()
    session = NemotronTabularSession()

    _predict_with_session(transport, payload, session)

    errors: list[BaseException] = []

    def score() -> None:
        try:
            _predict_with_session(transport, payload, session)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=score) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert transport.created == ['s0']
    assert session.id == 's0'
    assert transport.deleted == []


def test_a_concurrent_context_change_releases_exactly_one_session(
    context_df, predict_df
):
    r"""The lock must not turn a re-pin into a leak of its own: the superseded
    id is handed back to be released once, outside the lock, and the next call
    opens a single replacement.
    """
    payload = build_request(
        context=context_df,
        predict=predict_df,
        task='classification',
        target='target_col',
        outputs=['prediction'],
    )
    other = build_request(
        context=context_df,
        predict=predict_df.astype({'score': str}),
        task='classification',
        target='target_col',
        outputs=['prediction'],
    )
    transport = _CountingTransport(create_delay=0.0)
    session = NemotronTabularSession()

    _predict_with_session(transport, payload, session)
    _predict_with_session(transport, payload, session)
    assert session.id == 's0'

    _predict_with_session(transport, other, session)
    assert transport.deleted == ['s0']
    assert session.id is None

    _predict_with_session(transport, other, session)
    assert transport.created == ['s0', 's1']
    assert session.id == 's1'

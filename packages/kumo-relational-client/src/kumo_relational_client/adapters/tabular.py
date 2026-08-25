# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import uuid
from numbers import Real
from typing import Any

import pandas as pd

from kumo_relational_client.base import ModelAdapter, ModelCapabilities
from kumo_relational_client.core.response import parse_prediction_response
from kumo_relational_client.core.transport import Transport, redact_url
from kumo_relational_client.errors import NimRequestError, RelationalError
from kumo_relational_client.requests import (
    KumoTabularRequest,
    KumoTabularSession,
)
from kumo_relational_client.wire import (
    encode_table,
    infer_tfm_dtype,
    widen_tfm_dtype,
)

_CLASSIFICATION_KINDS = frozenset(
    {
        'classification',
        'binary_classification',
        'multiclass_classification',
    }
)

_TASK_KINDS = _CLASSIFICATION_KINDS | frozenset({'regression'})

# The NIM serves the two canonical kinds only; the finer-grained names are
# aliases for the same estimator and must be normalized before they go out.
_WIRE_TASK_KINDS = {
    'binary_classification': 'classification',
    'multiclass_classification': 'classification',
}

# Kumo Tabular builds its KV cache for a fixed number of classes; beyond that the
# model raises and the NIM answers with an opaque 500.
_MAX_CLASSES = 10

# Fields Kumo Tabular can produce per task kind. Anything else is accepted by the
# contract but dropped server-side without comment.
_OUTPUT_FIELDS = {
    'classification': ('prediction', 'probabilities'),
    'regression': ('prediction', 'quantiles'),
}

# A session pins `model` + `task` + `schema` + `context`; only the remaining
# sections travel with each scoring call. `metadata` rides along on both halves
# because its request id correlates that call with the server's logs.
_SESSION_PINNED_SECTIONS = ('model', 'task', 'schema', 'context')
_SESSION_CREATE_SECTIONS = (*_SESSION_PINNED_SECTIONS, 'metadata')
_SESSION_PREDICT_SECTIONS = ('predict', 'output', 'metadata')

# A NIM that does not serve the session routes at all, as opposed to one that
# has merely forgotten this session.
_SESSION_UNSUPPORTED_STATUS = frozenset({404, 405, 501})


def _new_request_id() -> str:
    return f'nemotron_{uuid.uuid4().hex[:12]}'


def _check_columns(name: str, frame: pd.DataFrame) -> None:
    if frame.columns.has_duplicates:
        duplicates = sorted(
            {
                str(column)
                for column in frame.columns[frame.columns.duplicated()]
            }
        )
        raise RelationalError(
            f'{name} has duplicate column name(s) {duplicates}',
            code='INVALID_REQUEST',
        )
    non_strings = [
        column for column in frame.columns if not isinstance(column, str)
    ]
    if non_strings:
        raise RelationalError(
            f'{name} column names must be strings; got {non_strings}',
            code='INVALID_REQUEST',
        )


def _validate_outputs(outputs: Any, wire_task: str) -> None:
    if isinstance(outputs, (str, bytes)):
        raise RelationalError(
            f'outputs must be a list of field names, not a single string; '
            f'pass [{outputs!r}]. A bare string is iterable, so the check '
            f'below would otherwise read it one character at a time',
            code='INVALID_REQUEST',
        )
    if not isinstance(outputs, (list, tuple)):
        raise RelationalError(
            'outputs must be a list or tuple of field names; got '
            f'{type(outputs).__name__}',
            code='INVALID_REQUEST',
        )
    non_strings = [field for field in outputs if not isinstance(field, str)]
    if non_strings:
        raise RelationalError(
            f'outputs entries must be strings; got {non_strings}',
            code='INVALID_REQUEST',
        )
    unsupported = [
        field for field in outputs if field not in _OUTPUT_FIELDS[wire_task]
    ]
    if unsupported:
        raise RelationalError(
            f'Kumo Tabular does not produce {unsupported} for a {wire_task} task; '
            f'supported output fields: {list(_OUTPUT_FIELDS[wire_task])}',
            code='INVALID_REQUEST',
        )


def _validate_quantile_levels(quantile_levels: Any) -> None:
    if quantile_levels is None:
        return
    if not isinstance(quantile_levels, (list, tuple)):
        raise RelationalError(
            'quantile_levels must be a list or tuple of finite numbers; got '
            f'{type(quantile_levels).__name__}',
            code='INVALID_REQUEST',
        )
    invalid = [
        level
        for level in quantile_levels
        if (
            isinstance(level, bool)
            or not isinstance(level, Real)
            or not math.isfinite(level)
            or not 0.0 < level < 1.0
        )
    ]
    if invalid:
        raise RelationalError(
            f'quantile_levels must lie strictly between 0 and 1; got '
            f'{quantile_levels}',
            code='INVALID_REQUEST',
        )


def build_request(
    *,
    context: pd.DataFrame,
    predict: pd.DataFrame,
    task: str,
    target: str,
    outputs: list[str],
    positive_class: str | None = None,
    prediction_statistic: str | None = None,
    quantile_levels: list[float] | None = None,
    score_format: str | None = None,
    embedding_dtype: str | None = None,
    max_results: int | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    if task not in _TASK_KINDS:
        raise RelationalError(
            f'task must be one of {sorted(_TASK_KINDS)}; got {task!r}',
            code='INVALID_REQUEST',
        )
    wire_task = _WIRE_TASK_KINDS.get(task, task)

    _check_columns('context', context)
    _check_columns('predict', predict)

    if target not in context.columns:
        raise RelationalError(
            f'target column {target!r} not found in context',
            code='INVALID_REQUEST',
        )
    if len(context) == 0:
        raise RelationalError(
            'context is empty; Kumo Tabular needs at least one labelled row',
            code='INVALID_REQUEST',
        )
    if len(predict) == 0:
        raise RelationalError(
            'predict is empty; Kumo Tabular needs at least one row to score',
            code='INVALID_REQUEST',
        )

    unlabelled = context[target].isna()
    if bool(unlabelled.any()):
        rows = list(context.index[unlabelled][:10])
        raise RelationalError(
            f'context target {target!r} has {int(unlabelled.sum())} missing '
            f'value(s) at rows {rows}; drop or fill them',
            code='INVALID_REQUEST',
        )

    _validate_outputs(outputs, wire_task)
    _validate_quantile_levels(quantile_levels)

    # A column typed differently in the two frames travels under the wider of
    # the two dtypes, so neither frame's values are coerced to the other's.
    dtypes = {
        column: widen_tfm_dtype(
            infer_tfm_dtype(frame[column])
            for frame in (context, predict)
            if column in frame.columns
        )
        for column in dict.fromkeys([*context.columns, *predict.columns])
    }

    task_spec: dict[str, Any] = {
        'kind': wire_task,
        'target': {
            'column_name': target,
            'dtype': dtypes[target],
        },
    }
    if task in _CLASSIFICATION_KINDS:
        classes = sorted(
            str(value) for value in context[target].dropna().unique()
        )
        if len(classes) > _MAX_CLASSES:
            raise RelationalError(
                f'Kumo Tabular supports at most {_MAX_CLASSES} classes; context '
                f'target {target!r} has {len(classes)}',
                code='INVALID_REQUEST',
            )
        task_spec['target']['classes'] = classes
        if positive_class is not None:
            wire_positive_class = str(positive_class)
            if wire_positive_class not in classes:
                raise RelationalError(
                    f'positive_class {positive_class!r} is not one of the '
                    f'context classes {classes}',
                    code='INVALID_REQUEST',
                )
            task_spec['target']['positive_class'] = wire_positive_class

    output_spec: dict[str, Any] = {'fields': outputs}
    if prediction_statistic is not None:
        output_spec['prediction_statistic'] = prediction_statistic
    if quantile_levels is not None:
        output_spec['quantile_levels'] = quantile_levels
    if score_format is not None:
        output_spec['score_format'] = score_format
    if embedding_dtype is not None:
        output_spec['embedding_dtype'] = embedding_dtype
    if max_results is not None:
        output_spec['max_results'] = max_results

    schema = {
        'instance_table': {
            'columns': {
                column: {'dtype': dtype} for column, dtype in dtypes.items()
            },
        },
        'related_tables': {},
        'relationships': [],
    }

    return {
        'model': 'kumo-tabular',
        'task': task_spec,
        'schema': schema,
        'context': {
            'instance_table': encode_table(context, dtypes),
            'related_tables': {},
        },
        'predict': {
            'instance_table': encode_table(predict, dtypes),
            'related_tables': {},
        },
        'output': output_spec,
        'metadata': {'request_id': request_id or _new_request_id()},
    }


def _sections(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: payload[key] for key in keys if key in payload}


def _pinned_digest(payload: dict[str, Any]) -> str:
    r"""A fingerprint of the half of ``payload`` a session pins.

    Held instead of the sections themselves so a handle does not keep a second
    copy of the context alive for its lifetime. Covers ``model`` / ``task`` /
    ``schema`` / ``context`` in full, so a widened dtype, a changed target
    class list or a different positive class all change the digest -- which is
    what makes reusing a pinned context safe.
    """
    serialized = json.dumps(
        _sections(payload, _SESSION_PINNED_SECTIONS),
        sort_keys=True,
        separators=(',', ':'),
        default=str,
    )
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def _create_session(transport: Transport, payload: dict[str, Any]) -> str:
    body = transport.create_session(
        _sections(payload, _SESSION_CREATE_SECTIONS)
    )
    session_id = body.get('session_id')
    if not isinstance(session_id, str) or not session_id:
        raise RelationalError(
            f'Create-session response from {redact_url(transport.url)} did '
            f'not include a session_id',
            code='INVALID_RESPONSE',
        )
    return session_id


def delete_session_quietly(transport: Transport, session_id: str) -> None:
    r"""Best-effort session release; the NIM also reaps a session on its TTL."""
    with contextlib.suppress(RelationalError):
        transport.delete_session(session_id)


def _acquire_session(
    transport: Transport,
    payload: dict[str, Any],
    session: KumoTabularSession,
    pinned: str,
) -> tuple[str | None, str | None]:
    r"""Decide, under ``session.lock``, which session id should serve this call.

    Returns the id to score against (``None`` to score statelessly) and the id
    of a superseded session the caller should release. The release is handed
    back rather than done here so the network round trip happens outside the
    lock: it is bookkeeping, and holding the lock across it would serialize
    every concurrent caller behind it.

    Opening a session is an optimisation, so any failure to open one falls back
    to the stateless path instead of failing the caller's prediction -- a
    NIM without the session routes latches ``supported`` off so the handle
    stops asking, while a transient failure is simply retried next call.
    """
    with session.lock:
        if session.pinned != pinned:
            # First call, or a call whose context half differs (a widened
            # dtype, a different positive_class): the old session cannot
            # answer it.
            stale, session.id = session.id, None
            session.pinned = pinned
            return None, stale
        if session.id is not None:
            return session.id, None
        try:
            session.id = _create_session(transport, payload)
        except NimRequestError as error:
            if error.status_code in _SESSION_UNSUPPORTED_STATUS:
                session.supported = False
            return None, None
        except RelationalError:
            return None, None
        return session.id, None


def _refresh_session(
    transport: Transport,
    payload: dict[str, Any],
    session: KumoTabularSession,
    stale_id: str,
) -> str:
    r"""Re-pin the context after the NIM forgot ``stale_id``.

    Under the lock, and only when no concurrent caller has already replaced it,
    so a TTL boundary reached by several threads at once costs one new session
    rather than one per thread.
    """
    with session.lock:
        if session.id is not None and session.id != stale_id:
            return session.id
        session.id = _create_session(transport, payload)
        return session.id


def _predict_with_session(
    transport: Transport,
    payload: dict[str, Any],
    session: KumoTabularSession,
) -> dict[str, Any]:
    r"""Scores ``payload`` against ``session``'s pinned context when that is
    cheaper than sending the context again, and stateless otherwise.

    The result is the same either way -- the payload is built in full first,
    and the session holds exactly the sections that would otherwise be re-sent.

    A session is opened on the *second* call that pins the same context rather
    than the first: opening one costs an extra round trip and pins server
    memory, which a caller who scores a single table would pay for nothing. By
    the second call the handle is demonstrably being reused, and every call
    from then on sends the rows alone.
    """
    if not session.supported:
        return transport.predict(payload)

    pinned = _pinned_digest(payload)
    session_id, stale_id = _acquire_session(transport, payload, session, pinned)
    if stale_id is not None:
        delete_session_quietly(transport, stale_id)
    if session_id is None:
        return transport.predict(payload)

    predict_payload = _sections(payload, _SESSION_PREDICT_SECTIONS)
    try:
        return transport.session_predict(session_id, predict_payload)
    except NimRequestError as error:
        # The session expired or was evicted: pin the context again and retry
        # once, so a long-lived handle never fails on a TTL boundary.
        if error.status_code != 404:
            raise
        return transport.session_predict(
            _refresh_session(transport, payload, session, session_id),
            predict_payload,
        )


class KumoTabularAdapter(ModelAdapter):
    name = 'kumo-tabular'
    request_type = KumoTabularRequest

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model='kumo-tabular',
            request_type=self.request_type.__name__,
            tasks=tuple(sorted(_TASK_KINDS)),
            outputs=('prediction', 'probabilities', 'quantiles'),
        )

    def predict(
        self,
        transport: Transport,
        request: KumoTabularRequest,
    ) -> pd.DataFrame:
        payload = build_request(
            context=request.context,
            predict=request.predict,
            task=request.task,
            target=request.target,
            outputs=request.outputs,
            positive_class=request.positive_class,
            prediction_statistic=request.prediction_statistic,
            quantile_levels=request.quantile_levels,
            score_format=request.score_format,
            embedding_dtype=request.embedding_dtype,
            max_results=request.max_results,
            request_id=request.request_id,
        )
        if request.session is None:
            response = transport.predict(payload)
        else:
            response = _predict_with_session(
                transport, payload, request.session
            )
        return parse_prediction_response(
            response, requested_fields=request.outputs
        )

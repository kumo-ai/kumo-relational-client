# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import pandas as pd

from nvidia_sdfm.base import ModelAdapter, ModelCapabilities, request_type_names
from nvidia_sdfm.core.transport import Transport
from nvidia_sdfm.errors import MissingExtraError, SdfmError
from nvidia_sdfm.requests import KumoRFMRequest, KumoRFMTaskRequest

if TYPE_CHECKING:
    from kumorfm.rfm.rfm import Explanation

_UNSET = object()

_RESERVED_OPTIONS = {'indices', 'run_mode', 'batch_size', 'num_retries'}

RFM_TASK_TYPES = (
    'binary_classification',
    'multiclass_classification',
    'regression',
    'forecasting',
    'temporal_link_prediction',
)


def _load_engine() -> Any:
    r"""Import the optional ``kumorfm`` engine, surfacing a clear extra error."""
    try:
        import kumorfm.rfm as rfm_engine
    except ModuleNotFoundError as error:
        if error.name != 'kumorfm':
            raise
        raise MissingExtraError('kumorfm', 'kumorfm') from error
    return rfm_engine


def _is_explain_config(value: Any) -> bool:
    r"""Return ``True`` when ``value`` is a driver ``ExplainConfig`` instance.

    Imported lazily so this adapter still imports without the optional
    ``kumorfm`` engine installed.
    """
    try:
        from kumorfm.rfm.rfm import ExplainConfig
    except Exception:
        return False
    return isinstance(value, ExplainConfig)


def _resolve_explain(field_value: Any, options: dict[str, Any]) -> Any:
    r"""Resolve the effective ``explain`` setting from the request field and the
    legacy ``options['explain']`` compatibility path.

    The request ``explain`` field is canonical. ``options['explain']`` is still
    accepted for backward compatibility, but specifying both is rejected so a
    stale option cannot silently override the first-class field. The resulting
    value must be a ``bool``, an ``ExplainConfig``, or an ``ExplainConfig``
    dict (all accepted by the driver); anything else is an ``INVALID_REQUEST``
    rather than a deep engine-level failure.
    """
    option_value = options.pop('explain', _UNSET)
    explain = field_value
    if option_value is not _UNSET:
        if (field_value is True or isinstance(field_value, dict)
                or _is_explain_config(field_value)):
            raise SdfmError(
                "explain is set both as a request field and in options; "
                "specify it once (prefer the request field)",
                code='INVALID_REQUEST',
            )
        explain = option_value
    if (explain is not True and explain is not False
            and not isinstance(explain, dict)
            and not _is_explain_config(explain)):
        raise SdfmError(
            "explain must be a bool, an ExplainConfig, or an ExplainConfig "
            f"dict; got {type(explain).__name__}",
            code='INVALID_REQUEST',
        )
    return explain


def _reject_reserved_options(options: dict[str, Any]) -> None:
    r"""Reject first-class request fields smuggled through ``options``."""
    reserved = _RESERVED_OPTIONS & set(options)
    if reserved:
        raise SdfmError(
            f'KumoRFM request options contain reserved keys {sorted(reserved)}; '
            'set them as request fields instead',
            code='INVALID_REQUEST',
        )


def _validate_batch_size(batch_size: Any) -> None:
    if (batch_size is not None and batch_size != 'max'
            and not (isinstance(batch_size, int)
                     and not isinstance(batch_size, bool)
                     and batch_size > 0)):
        raise SdfmError(
            "batch_size must be a positive int or the literal 'max'; got "
            f"{batch_size!r}",
            code='INVALID_REQUEST',
        )


def _require_columns(frame: pd.DataFrame, name: str, columns: set[str]) -> None:
    missing = [column for column in sorted(columns)
               if column not in frame.columns]
    if missing:
        raise SdfmError(
            f'{name} table is missing required column(s) {missing}; '
            f'present columns: {list(frame.columns)}',
            code='INVALID_REQUEST',
        )


def _validate_task_request(request: KumoRFMTaskRequest) -> None:
    r"""Reject caller-supplied task requests with clear ``INVALID_REQUEST`` errors.

    Fills the gaps the engine ``TaskTable`` leaves as bare exceptions: an unknown
    ``task_type``, an ``entity_table`` absent from the graph, and context/predict
    frames missing the entity, target or time columns.
    """
    if request.task_type not in RFM_TASK_TYPES:
        raise SdfmError(
            f'task_type must be one of {list(RFM_TASK_TYPES)}; got '
            f'{request.task_type!r}',
            code='INVALID_REQUEST',
        )

    if isinstance(request.entity_table, str):
        entity_tables: tuple[str, ...] = (request.entity_table, )
    else:
        entity_tables = tuple(request.entity_table)
        if not 1 <= len(entity_tables) <= 2:
            raise SdfmError(
                'entity_table must be a table name or a (source, target) pair '
                f'for link prediction; got {request.entity_table!r}',
                code='INVALID_REQUEST',
            )
    tables = request.graph.tables
    missing = [table for table in entity_tables if table not in tables]
    if missing:
        raise SdfmError(
            f'entity_table {missing} not found in graph; available tables: '
            f'{sorted(tables)}',
            code='INVALID_REQUEST',
        )

    context_columns = {request.entity_column, request.target_column}
    if request.time_column is not None:
        context_columns.add(request.time_column)
    _require_columns(request.context, 'context', context_columns)
    _require_columns(request.predict, 'predict', {request.entity_column})


def _build_task_table(engine: Any, request: KumoRFMTaskRequest) -> Any:
    r"""Assemble the engine ``TaskTable`` from a caller-supplied context request.

    Mirrors the ``TaskTable`` the query path builds internally: ``time_column``
    falls back to ``ANCHOR_TIMESTAMP`` when the context carries one, otherwise the
    entity table's own time column.
    """
    time_column = request.time_column
    if time_column is None:
        has_anchor_timestamp = (
            'ANCHOR_TIMESTAMP' in request.context.columns
            or 'ANCHOR_TIMESTAMP' in request.predict.columns)
        time_column = ('ANCHOR_TIMESTAMP' if has_anchor_timestamp
                       else engine.TaskTable.ENTITY_TIME)
    return engine.TaskTable(
        task_type=request.task_type,
        context_df=request.context,
        pred_df=request.predict,
        entity_table_name=request.entity_table,
        entity_column=request.entity_column,
        target_column=request.target_column,
        time_column=time_column,
        num_forecasts=request.num_forecasts,
        step_size=request.step_size,
    )


def _coerce_result(result: Any, wants_explanation: bool) -> 'pd.DataFrame | Explanation':
    if wants_explanation:
        from kumorfm.rfm.rfm import Explanation
        if not isinstance(result, Explanation):
            raise TypeError(
                'expected an Explanation result for explain=True; got '
                f'{type(result).__name__}',
            )
        return result
    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            'expected a DataFrame result; pass explain=False (the '
            'default) to get a plain prediction DataFrame',
        )
    return result


class KumoRFMAdapter(ModelAdapter):
    name = 'kumo-rfm'
    request_type = (KumoRFMRequest, KumoRFMTaskRequest)

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model='kumo-rfm',
            request_type=request_type_names(self.request_type),
            tasks=RFM_TASK_TYPES,
            outputs=('prediction', 'probabilities', 'explanation'),
        )

    def predict(
        self,
        transport: Transport,
        request: 'KumoRFMRequest | KumoRFMTaskRequest',
    ) -> 'pd.DataFrame | Explanation':
        engine = _load_engine()

        _validate_batch_size(request.batch_size)
        options = dict(request.options)
        _reject_reserved_options(options)
        explain = _resolve_explain(request.explain, options)
        if isinstance(request, KumoRFMTaskRequest):
            _validate_task_request(request)

        engine.init(url=transport.url, api_key=transport.api_key,
                    verify_ssl=transport.verify_ssl,
                    _token=engine._SDFM_CLIENT_TOKEN)
        model = engine.KumoRFM(request.graph)
        if request.batch_size is not None:
            batch_ctx = model.batch_mode(request.batch_size,
                                         num_retries=request.num_retries)
        else:
            batch_ctx = contextlib.nullcontext()
        with batch_ctx:
            if isinstance(request, KumoRFMTaskRequest):
                result = model.predict_task(
                    _build_task_table(engine, request),
                    run_mode=request.run_mode,
                    explain=explain,
                    **options,
                )
            else:
                result = model.predict(
                    request.query,
                    indices=request.indices,
                    run_mode=request.run_mode,
                    explain=explain,
                    **options,
                )
        return _coerce_result(result, explain is not False)

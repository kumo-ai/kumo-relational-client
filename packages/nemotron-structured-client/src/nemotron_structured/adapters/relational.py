# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import pandas as pd

from nemotron_structured.base import (
    ModelAdapter,
    ModelCapabilities,
    request_type_names,
)
from nemotron_structured.core.serving import ServingTarget
from nemotron_structured.core.transport import Transport, redact_url
from nemotron_structured.errors import (
    MissingExtraError,
    NimRequestError,
    StructuredError,
)
from nemotron_structured.requests import (
    NemotronRelationalRequest,
    NemotronRelationalTaskRequest,
)

if TYPE_CHECKING:
    from nemotron_relational.rfm.rfm import Explanation

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
    r"""Import the optional ``nemotron_relational`` engine, surfacing a clear extra error."""
    try:
        import nemotron_relational.rfm as rfm_engine
    except ModuleNotFoundError as error:
        if error.name != 'nemotron_relational':
            raise
        raise MissingExtraError('relational', 'nemotron-relational') from error
    return rfm_engine


def _is_explain_config(value: Any) -> bool:
    r"""Return ``True`` when ``value`` is a driver ``ExplainConfig`` instance.

    Imported lazily so this adapter still imports without the optional
    ``nemotron_relational`` engine installed.
    """
    try:
        from nemotron_relational.rfm.rfm import ExplainConfig
    except Exception:
        return False
    return isinstance(value, ExplainConfig)


def _engine_http_error_types() -> tuple[type[BaseException], ...]:
    r"""nemotron_relational's ``HTTPException``, or an empty tuple if it cannot be imported.

    Looked up lazily for the same reason the engine itself is: this adapter
    must import without the optional ``nemotron_relational`` engine installed. ``except ()``
    matches nothing, which is the right behaviour when there is no engine to
    have raised.
    """
    try:
        from nemotron_relational.exceptions import HTTPException
    except Exception:
        return ()
    return (HTTPException,)


class _ServingPlatform(NamedTuple):
    """How one managed platform is reached and what it needs installed.

    The extra and the dependency travel with the initializer because they are
    only ever needed together: the ``ImportError`` they answer is raised by the
    initializer itself.
    """

    initialize: Callable[[Any, ServingTarget], None]
    extra: str
    dependency: str


_SERVING_PLATFORMS: dict[str, _ServingPlatform] = {
    'databricks': _ServingPlatform(
        initialize=lambda engine, target: engine.init_databricks_serving(
            target.endpoint,
            workspace_client=target.platform_client,
            _token=engine._CLIENT_TOKEN,
        ),
        extra='databricks-serving',
        dependency='databricks-sdk',
    ),
    'snowflake': _ServingPlatform(
        initialize=lambda engine, target: engine.init_snowflake_serving(
            target.endpoint,
            session=target.platform_client,
            _token=engine._CLIENT_TOKEN,
        ),
        extra='snowflake-serving',
        dependency='snowflake-snowpark-python',
    ),
}


def _init_serving(rfm_engine: Any, target: ServingTarget) -> None:
    r"""Initialize the engine against a serving endpoint.

    Every failure is translated at this boundary, the way
    :mod:`nemotron_structured.core.connectors` translates the connector layer's. The
    ``ImportError`` case is the reason this exists: nemotron_relational names *its own*
    extra (``pip install 'nemotron_relational[databricks-serving]'``), which a user who
    installed ``nemotron-structured-client[databricks-serving]`` never asked for and cannot
    act on.
    """
    platform = _SERVING_PLATFORMS.get(target.kind)
    if platform is None:
        raise StructuredError(
            f'unknown serving platform {target.kind!r}',
            code='INVALID_CONFIGURATION',
        )
    try:
        platform.initialize(rfm_engine, target)
    except ImportError as error:
        raise MissingExtraError(platform.extra, platform.dependency) from error
    except ValueError as error:
        raise StructuredError(
            str(error), code='INVALID_CONFIGURATION'
        ) from error
    except _engine_http_error_types() as error:
        raise StructuredError(
            f'could not reach the serving endpoint '
            f'{target.endpoint!r}: {getattr(error, "detail", error)}',
            code='SERVING_INIT_FAILED',
            details={'status_code': getattr(error, 'status_code', None)},
        ) from error


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
        if (
            field_value is True
            or isinstance(field_value, dict)
            or _is_explain_config(field_value)
        ):
            raise StructuredError(
                'explain is set both as a request field and in options; '
                'specify it once (prefer the request field)',
                code='INVALID_REQUEST',
            )
        explain = option_value
    if (
        explain is not True
        and explain is not False
        and not isinstance(explain, dict)
        and not _is_explain_config(explain)
    ):
        raise StructuredError(
            'explain must be a bool, an ExplainConfig, or an ExplainConfig '
            f'dict; got {type(explain).__name__}',
            code='INVALID_REQUEST',
        )
    return explain


def _reject_reserved_options(options: dict[str, Any]) -> None:
    r"""Reject first-class request fields smuggled through ``options``."""
    reserved = _RESERVED_OPTIONS & set(options)
    if reserved:
        raise StructuredError(
            f'Nemotron Relational request options contain reserved keys {sorted(reserved)}; '
            'set them as request fields instead',
            code='INVALID_REQUEST',
        )


def _driver_error_types() -> tuple[Any, Any]:
    r"""The driver's classified error types, or ``(None, None)`` without it."""
    try:
        from nemotron_relational.exceptions import (
            InvalidResponseError,
            NimFailureError,
        )
    except Exception:
        return None, None
    return NimFailureError, InvalidResponseError


def _engine_init_error_types() -> tuple[type[BaseException], ...]:
    r"""The engine's connection-time error types, widened if it is too old.

    ``ClientInitializationError`` subclasses :class:`ValueError`, which is what
    the engine raised before it grew a root exception, so falling back to
    ``ValueError`` keeps this boundary working against either version.
    """
    try:
        from nemotron_relational.exceptions import ClientInitializationError
    except Exception:
        return (ValueError,)
    return (ClientInitializationError,)


def _translate_init_error(error: BaseException) -> StructuredError:
    r"""Map a connection-time engine failure onto this client's hierarchy.

    Establishing the connection happens outside the prediction call, so these
    failures used to escape as the engine's bare ``ValueError`` and no
    ``except StructuredError`` could catch a wrong API key or an unreachable NIM.
    The codes separate what a caller can act on: credentials are not worth a
    retry, an unreachable or slow endpoint may be.
    """
    try:
        from nemotron_relational.exceptions import (
            AuthenticationError,
            NimTimeoutError,
            NimUnreachableError,
        )
    except Exception:
        return StructuredError(str(error), code='TRANSPORT_ERROR')
    if isinstance(error, AuthenticationError):
        return StructuredError(str(error), code='AUTHENTICATION_FAILED')
    if isinstance(error, NimTimeoutError | NimUnreachableError):
        return StructuredError(str(error), code='TRANSPORT_ERROR')
    return StructuredError(str(error), code='INVALID_CONFIGURATION')


def _describe(transport: Transport | ServingTarget) -> str:
    r"""Name the deployment for an error message, whichever transport it is.

    Reading ``transport.url`` is not safe here: a ``ServingTarget`` raises from
    that property by design, and this is called from an ``except`` block, so
    the raise replaced the failure being reported. Every engine error on the
    serving path surfaced as 'the serving endpoint has no URL' with the real
    cause discarded.

    The URL is redacted because a caller may carry the credential in it, and
    this name is spliced into an error message.
    """
    if isinstance(transport, ServingTarget):
        return f'serving endpoint {transport.endpoint!r}'
    return str(redact_url(transport.url))


def _translate_engine_error(error: Exception, url: str) -> StructuredError:
    r"""Map a driver failure onto this client's own exception hierarchy.

    The driver cannot depend on ``nemotron_structured``, so its NIM failures arrive as
    ``NimFailureError`` (a ``RuntimeError``) carrying the status and the
    problem document's ``invalid_params``. Translating here means ``except
    StructuredError`` catches every failure of the prediction call itself, as it
    already does for Nemotron Tabular. It does not cover graph construction: the engine
    validates the graph in its own constructor, outside this call, so a graph
    that fails validation still surfaces the engine's ``ValueError``.

    Caller-input failures the engine validates itself (an anchor time before
    the context window, an unknown option) arrive as ``ValueError``/
    ``TypeError`` with messages that are already actionable: those keep their
    message and become ``INVALID_REQUEST``, not ``INTERNAL_ERROR``. So does a
    ``LookupError``: a name the caller supplied that the engine looked up and
    did not find is a caller mistake however deep the lookup was, and telling
    the user to file a bug for their own typo spends the trust
    ``INTERNAL_ERROR`` exists to carry. ``KeyError`` stringifies as the
    ``repr`` of its argument, so its message is unwrapped rather than shown as
    ``"'nope'"``.

    A response that does not match the contract is ``INVALID_RESPONSE``, so a
    malformed server answer is never blamed on the request. Only genuinely
    unanticipated failures get ``INTERNAL_ERROR``, and they name the endpoint
    so the report is actionable.
    """
    failure_type, invalid_response_type = _driver_error_types()
    if failure_type is not None and isinstance(error, failure_type):
        details = (
            {'invalid_params': error.invalid_params}
            if error.invalid_params
            else {}
        )
        if error.status_code is None:
            return StructuredError(
                str(error), code='TRANSPORT_ERROR', details=details
            )
        return NimRequestError(
            error.status_code, code=None, message=str(error), details=details
        )
    if invalid_response_type is not None and isinstance(
        error, invalid_response_type
    ):
        return StructuredError(str(error), code='INVALID_RESPONSE')
    if isinstance(error, KeyError):
        message = str(error.args[0]) if error.args else str(error)
        return StructuredError(message, code='INVALID_REQUEST')
    if isinstance(error, (ValueError, TypeError, LookupError)):
        return StructuredError(str(error), code='INVALID_REQUEST')
    return StructuredError(
        f'The kumo-relational prediction at {url} failed unexpectedly with '
        f'{type(error).__name__}: {error}',
        code='INTERNAL_ERROR',
    )


def _validate_num_retries(num_retries: Any) -> None:
    if (
        not isinstance(num_retries, int)
        or isinstance(num_retries, bool)
        or num_retries < 0
    ):
        raise StructuredError(
            f'num_retries must be a non-negative int; got {num_retries!r}',
            code='INVALID_REQUEST',
        )


def _validate_batch_size(batch_size: Any) -> None:
    if (
        batch_size is not None
        and batch_size != 'max'
        and not (
            isinstance(batch_size, int)
            and not isinstance(batch_size, bool)
            and batch_size > 0
        )
    ):
        raise StructuredError(
            "batch_size must be a positive int or the literal 'max'; got "
            f'{batch_size!r}',
            code='INVALID_REQUEST',
        )


def _require_columns(frame: pd.DataFrame, name: str, columns: set[str]) -> None:
    missing = [
        column for column in sorted(columns) if column not in frame.columns
    ]
    if missing:
        raise StructuredError(
            f'{name} table is missing required column(s) {missing}; '
            f'present columns: {list(frame.columns)}',
            code='INVALID_REQUEST',
        )


def _validate_task_request(request: NemotronRelationalTaskRequest) -> None:
    r"""Reject caller-supplied task requests with clear ``INVALID_REQUEST`` errors.

    Fills the gaps the engine ``TaskTable`` leaves as bare exceptions: an unknown
    ``task_type``, an ``entity_table`` absent from the graph, context/predict
    frames missing the entity, target or time columns, and a feature column
    present in only one of the two frames.

    Anything in ``context`` beyond the entity, target and time columns becomes
    a task feature, and the engine then reads the same column out of
    ``predict``. Present in only one frame it raises a bare ``KeyError`` naming
    the column but not the constraint, which the error classifier could only
    report as an internal failure.
    """
    if request.task_type not in RFM_TASK_TYPES:
        raise StructuredError(
            f'task_type must be one of {list(RFM_TASK_TYPES)}; got '
            f'{request.task_type!r}',
            code='INVALID_REQUEST',
        )

    if isinstance(request.entity_table, str):
        entity_tables: tuple[str, ...] = (request.entity_table,)
    else:
        entity_tables = tuple(request.entity_table)
        if not 1 <= len(entity_tables) <= 2:
            raise StructuredError(
                'entity_table must be a table name or a (source, target) pair '
                f'for link prediction; got {request.entity_table!r}',
                code='INVALID_REQUEST',
            )
    tables = request.graph.tables
    missing = [table for table in entity_tables if table not in tables]
    if missing:
        raise StructuredError(
            f'entity_table {missing} not found in graph; available tables: '
            f'{sorted(tables)}',
            code='INVALID_REQUEST',
        )

    context_columns = {request.entity_column, request.target_column}
    if request.time_column is not None:
        context_columns.add(request.time_column)
    _require_columns(request.context, 'context', context_columns)
    _require_columns(request.predict, 'predict', {request.entity_column})

    reserved = set(context_columns)
    reserved.add(
        request.time_column
        if request.time_column is not None
        else 'ANCHOR_TIMESTAMP'
    )
    context_only = [
        column
        for column in request.context.columns
        if column not in reserved and column not in set(request.predict.columns)
    ]
    if context_only:
        raise StructuredError(
            f'context carries feature column(s) {context_only} that predict '
            f'does not; supply them in both frames or drop them from context',
            code='INVALID_REQUEST',
        )


def _build_task_table(
    engine: Any, request: NemotronRelationalTaskRequest
) -> Any:
    r"""Assemble the engine ``TaskTable`` from a caller-supplied context request.

    Mirrors the ``TaskTable`` the query path builds internally: ``time_column``
    falls back to ``ANCHOR_TIMESTAMP`` when the context carries one, otherwise the
    entity table's own time column.
    """
    time_column = request.time_column
    if time_column is None:
        has_anchor_timestamp = (
            'ANCHOR_TIMESTAMP' in request.context.columns
            or 'ANCHOR_TIMESTAMP' in request.predict.columns
        )
        time_column = (
            'ANCHOR_TIMESTAMP'
            if has_anchor_timestamp
            else engine.TaskTable.ENTITY_TIME
        )
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


def _coerce_result(
    result: Any, wants_explanation: bool
) -> pd.DataFrame | Explanation:
    if wants_explanation:
        from nemotron_relational.rfm.rfm import Explanation

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


def _build_engine_model(
    engine: Any, graph: Any, api_client: Any, options: dict[str, Any]
) -> Any:
    r"""`verbose` reaches the constructor because it owns the
    graph-materialization banner; a reused model prints none, which is correct
    since no materialization happens."""
    if 'verbose' in options:
        return engine.NemotronRelational(
            graph, verbose=options['verbose'], _client=api_client
        )
    return engine.NemotronRelational(graph, _client=api_client)


def _graph_signature(graph: Any) -> Any | None:
    r"""A comparable description of what materializing this graph would use,
    or ``None``.

    Asks the graph rather than reading its tables here: which counts are cheap
    is a property of the backend, and only the engine knows that.

    Any graph that cannot describe itself is simply not cached. Correctness
    does not depend on the cache, only speed does.
    """
    describe = getattr(graph, '_materialization_signature', None)
    if describe is None:
        describe = getattr(graph, '_to_api_graph_definition', None)
    if describe is None:
        return None
    try:
        return describe()
    except Exception:
        return None


class NemotronRelationalAdapter(ModelAdapter):
    name = 'kumo-relational'
    request_type = (NemotronRelationalRequest, NemotronRelationalTaskRequest)

    def __init__(self) -> None:
        self._opened_engine_client = False
        self._engine_cache = threading.local()

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model='kumo-relational',
            request_type=request_type_names(self.request_type),
            tasks=RFM_TASK_TYPES,
            outputs=('prediction', 'probabilities', 'explanation'),
        )

    def predict(
        self,
        transport: Transport | ServingTarget,
        request: NemotronRelationalRequest | NemotronRelationalTaskRequest,
    ) -> pd.DataFrame | Explanation:
        engine = _load_engine()

        _validate_batch_size(request.batch_size)
        _validate_num_retries(request.num_retries)
        options = dict(request.options)
        _reject_reserved_options(options)
        explain = _resolve_explain(request.explain, options)
        if isinstance(request, NemotronRelationalTaskRequest):
            _validate_task_request(request)

        # The engine's endpoint is process-global, so configuring it and
        # resolving the resulting client has to be one atomic step, and the
        # result has to be bound to this prediction. Reading the global back
        # later instead would let a concurrent prediction from a differently
        # configured client re-point this one -- see `init_client`.
        if isinstance(transport, ServingTarget):
            _init_serving(engine, transport)
            self._opened_engine_client = True
            api_client = None
        else:
            try:
                api_client = engine.init_client(
                    url=transport.url,
                    api_key=transport.api_key,
                    verify_ssl=transport.verify_ssl,
                    timeout=transport.timeout,
                    max_retries=transport.max_retries,
                    _token=engine._CLIENT_TOKEN,
                )
            except _engine_init_error_types() as error:
                raise _translate_init_error(error) from error
            self._opened_engine_client = True
        model = self._engine_for(engine, request.graph, api_client, options)
        if request.batch_size is not None:
            batch_ctx = model.batch_mode(
                request.batch_size, num_retries=request.num_retries
            )
        elif request.num_retries:
            # Without a batch context the retry count used to be dropped, so
            # 'num_retries' was a silent no-op on the default path.
            batch_ctx = model.retry(request.num_retries)
        else:
            batch_ctx = contextlib.nullcontext()
        try:
            with batch_ctx:
                if isinstance(request, NemotronRelationalTaskRequest):
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
        except StructuredError:
            raise
        except Exception as error:
            raise _translate_engine_error(
                error, _describe(transport)
            ) from error
        return _coerce_result(result, explain is not False)

    def _engine_for(
        self,
        engine: Any,
        graph: Any,
        api_client: Any,
        options: dict[str, Any],
    ) -> Any:
        r"""The engine model for this graph, materialized at most once.

        Constructing the model materializes the graph: sanitizing every table,
        collecting primary keys and building the adjacency. That cost is
        independent of the query, so rebuilding it per prediction repeats
        identical work; on a graph with tens of millions of nodes it dominates
        the request and makes a retry loop unusable.

        One entry, held per thread. Per thread because a model carries the
        mutable batch size and retry count that ``batch_mode`` and ``retry``
        set and restore around a call: two threads sharing one model would
        interleave those and restore each other's values. One entry because
        the case worth serving is repeated questions of a single graph;
        holding every graph ever predicted against would retain each one's
        materialization for the life of the client.

        Reused only when the graph is the same object, its schema still
        compares equal, and the endpoint resolved to the same client object.
        The engine client is a process-wide singleton rebuilt whenever the URL
        or credential changes, so identity is what keeps a reconfigured
        endpoint from being served a model bound to the old one. A managed
        serving target resolves no client at all, so nothing is cached there
        rather than letting two endpoints share an entry keyed on ``None``.

        A graph is materialized as it was when first predicted against.
        Editing table data in place afterwards, while holding on to the same
        graph, is not picked up; build the graph again to pick it up.
        """
        if api_client is None:
            return _build_engine_model(engine, graph, api_client, options)

        definition = _graph_signature(graph)
        if definition is None:
            return _build_engine_model(engine, graph, api_client, options)

        cached = getattr(self._engine_cache, 'entry', None)
        if cached is not None:
            if (
                cached[0] is graph
                and cached[1] is api_client
                and cached[2] == definition
            ):
                return cached[3]

            # Let the old materialization go before building its replacement.
            # Holding both at once doubles the peak, and one of them can be
            # gigabytes: rel-amazon's edges alone are over a gigabyte before
            # the tables, the ID maps and the Python objects around them.
            # Indexed above rather than unpacked so that no named local keeps
            # it alive past this point.
            self._engine_cache.entry = None
            del cached

        model = _build_engine_model(engine, graph, api_client, options)
        self._engine_cache.entry = (graph, api_client, definition, model)
        return model

    def close(self) -> None:
        r"""Releases the engine's pooled connections for this thread.

        The engine keeps its own connection pool, separate from the client's
        transport, so closing the client alone would leave it open. Skipped
        when this adapter never configured the engine, so that closing a
        client that only ever used another model does not import the driver.
        """
        self._engine_cache = threading.local()
        if not self._opened_engine_client:
            return
        self._opened_engine_client = False
        engine = _load_engine()
        engine.close_client(_token=engine._CLIENT_TOKEN)

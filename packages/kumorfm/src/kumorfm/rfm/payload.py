# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
from dataclasses import dataclass
from decimal import Decimal
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from kumorfm.api.model_plan import RunMode
from kumorfm.api.rfm import RFMPredictRequest
from kumorfm.api.rfm.context import Context, EdgeLayout, REV_REL
from kumorfm.api.rfm.inference import (
    ClassificationInferenceConfig,
    InferenceConfig,
    RegressionInferenceConfig,
)
from kumorfm.api.task import TaskType
from kumorfm.api.typing import Stype
from kumorfm.client.generated.tfm_api import (
    TFM_MODEL_KUMO_RFM,
    TFM_OUTPUT_FIELD_EMBEDDINGS,
    TFM_OUTPUT_FIELD_EXPLANATION,
    TFM_OUTPUT_FIELD_PREDICTION,
    TFM_OUTPUT_FIELD_PROBABILITIES,
    TFM_OUTPUT_FIELD_QUANTILES,
    TFM_OUTPUT_FIELD_RANKINGS,
)

INSTANCE_ID = 'instance_id'
SYNTHETIC_NODE_ID = '__node_id'
ENTITY_REFERENCE_PREFIX = '__kumo_entity_ref'
ANCHOR_TIME_PREFIX = '__kumo_anchor_time'

JSON_SAFE_INT_MAX = 9007199254740991
JSON_SAFE_INT_MIN = -9007199254740991

_NON_FINITE_MSG = ("encountered a non-finite value ({value}); NIM requests "
                   "must contain finite numbers or nulls")


@dataclass(frozen=True)
class PayloadTables:
    context_instance_table: pd.DataFrame
    predict_instance_table: pd.DataFrame
    related_table_frames: dict[str, pd.DataFrame]
    context_related_tables: dict[str, pd.DataFrame]
    predict_related_tables: dict[str, pd.DataFrame]
    related_table_primary_keys: dict[str, str | list[str]]
    related_table_stype_overrides: dict[str, dict[str, Stype]]
    related_table_key_columns: dict[str, str]
    entity_reference_columns: dict[str, str]
    anchor_time_column: str | None
    relationships: list[dict[str, Any]]


def predict_request_to_json(
    request: RFMPredictRequest,
    *,
    explain: bool = False,
) -> dict[str, Any]:
    output_fields = _output_fields(
        request.context.task_type,
        return_embeddings=request.return_embeddings,
        explain=explain,
        quantile_output=(
            isinstance(request.inference_config, RegressionInferenceConfig)
            and request.inference_config.output_type == 'quantiles'
        ),
    )
    payload = _base_payload(
        context=request.context,
        run_mode=request.run_mode,
        use_prediction_time=request.use_prediction_time,
        inference_config=request.inference_config,
        output={'fields': output_fields},
        metadata={
            'source_query': request.query,
        },
    )
    return payload


MAX_TABLE_ROWS = 10_000

_SESSION_CREATE_SECTIONS = ('model', 'task', 'schema', 'context', 'metadata')
_SESSION_PREDICT_SECTIONS = ('predict', 'output', 'inference', 'metadata')


def session_create_payload(payload: dict[str, Any]) -> dict[str, Any]:
    r"""The context-only body for ``POST /v1/sessions``.

    A session pins ``model`` + ``task`` + ``schema`` + ``context`` once; this
    keeps exactly those sections (plus ``metadata``) from a full prediction
    payload and drops the per-call ``predict`` / ``output`` / ``inference``.
    """
    return {key: payload[key]
            for key in _SESSION_CREATE_SECTIONS if key in payload}


def session_predict_payload(payload: dict[str, Any]) -> dict[str, Any]:
    r"""The per-call body for ``POST /v1/sessions/{id}/predictions``.

    The session already holds ``model`` / ``task`` / ``schema`` / ``context``,
    so only ``predict`` + ``output`` + ``inference`` (plus ``metadata``) travel
    with each prediction.
    """
    return {key: payload[key]
            for key in _SESSION_PREDICT_SECTIONS if key in payload}


def payload_size_bytes(payload: dict[str, Any]) -> int:
    return len(
        json.dumps(
            payload,
            allow_nan=False,
            separators=(',', ':'),
        ).encode('utf-8'))


def validate_payload_table_rows(
    payload: dict[str, Any],
    *,
    batch_index: int,
    limit: int = MAX_TABLE_ROWS,
) -> None:
    r"""Validate instance-table row limits in serialized requests.

    The Kumo RFM NIM row-caps only context/predict instance tables. Related
    tables are bounded by the overall payload-size limit instead.
    """
    for section_name in ('context', 'predict'):
        table = payload[section_name]['instance_table']
        num_rows = len(table['rows'])
        if num_rows > limit:
            path = f'{section_name}.instance_table'
            raise ValueError(
                f"Request batch {batch_index} table '{path}' contains "
                f"{num_rows:,} rows, exceeding the {limit:,}-row limit")


def context_size_stats(context: Context) -> str:
    table_stats = []
    for table_name, table in context.subgraph.table_dict.items():
        size = int(table.df.memory_usage(deep=True).sum())
        table_stats.append((table_name, sum(table.num_sampled_nodes), size))
    table_stats = sorted(table_stats, key=lambda item: item[2], reverse=True)
    num_nodes = sum(num_nodes for _, num_nodes, _ in table_stats)
    num_edges = sum(
        link.num_edges for link in context.subgraph.link_dict.values())
    top = table_stats[:5]
    top_repr = ', '.join(
        f'{name}: {size / (1024 * 1024):.2f}MB' for name, _, size in top)
    return (f"Current context contains {num_nodes:,} nodes and "
            f"{num_edges:,} edges across {len(table_stats)} tables. "
            f"Top-{len(top)} tables contributing most to the context size: "
            f"{top_repr}")


def _base_payload(
    *,
    context: Context,
    run_mode: RunMode,
    use_prediction_time: bool,
    inference_config: InferenceConfig | None,
    output: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    tables = _payload_tables(context)
    metadata = {
        key: value
        for key, value in metadata.items()
        if value is not None and value != ''
    }
    metadata['num_context_examples'] = context.num_train
    metadata['num_prediction_examples'] = context.num_test

    return {
        'model': TFM_MODEL_KUMO_RFM,
        'task': _task_spec(context, tables),
        'schema': _schema_spec(context, tables),
        'context': {
            'instance_table':
            _dataframe_table(
                _instance_payload_dataframe(
                    tables.context_instance_table,
                    context,
                ), 'instance_table'),
            'related_tables': {
                table_name: _dataframe_table(df, table_name)
                for table_name, df in tables.context_related_tables.items()
            },
        },
        'predict': {
            'instance_table':
            _dataframe_table(
                _instance_payload_dataframe(
                    tables.predict_instance_table,
                    context,
                ), 'instance_table'),
            'related_tables': {
                table_name: _dataframe_table(df, table_name)
                for table_name, df in tables.predict_related_tables.items()
            },
        },
        'output': output,
        'inference': {
            'run_mode': RunMode(run_mode).value,
            'use_prediction_time': use_prediction_time,
            'inference_config': _inference_config_to_json(inference_config),
        },
        'metadata': metadata,
    }


def _payload_tables(context: Context) -> PayloadTables:
    instance_df, entity_reference_columns, anchor_time_column = (
        _instance_dataframe(context)
    )
    related_table_frames: dict[str, pd.DataFrame] = {}
    related_table_primary_keys: dict[str, str | list[str]] = {}
    related_table_stype_overrides: dict[str, dict[str, Stype]] = {}
    related_table_key_columns: dict[str, str] = {}

    for table_name, table in context.subgraph.table_dict.items():
        df, key_column = _occurrence_dataframe(table)
        stype_overrides = {INSTANCE_ID: Stype.ID, **table.stype_dict}
        if key_column != table.primary_key:
            stype_overrides[key_column] = Stype.ID

        related_table_frames[table_name] = df
        related_table_key_columns[table_name] = key_column
        related_table_primary_keys[table_name] = (
            [INSTANCE_ID, key_column]
        )
        related_table_stype_overrides[table_name] = stype_overrides

    relationships = _populate_relationship_columns(
        context,
        instance_df,
        related_table_frames,
        related_table_key_columns,
        related_table_stype_overrides,
        entity_reference_columns,
    )

    context_related_tables: dict[str, pd.DataFrame] = {}
    predict_related_tables: dict[str, pd.DataFrame] = {}
    for table_name, df in related_table_frames.items():
        instance_ids = df[INSTANCE_ID].to_numpy()
        context_mask = instance_ids < context.num_train
        predict_mask = instance_ids >= context.num_train
        context_related_tables[table_name] = df.loc[context_mask].reset_index(
            drop=True)
        predict_related_tables[table_name] = df.loc[predict_mask].reset_index(
            drop=True)

    return PayloadTables(
        context_instance_table=instance_df.iloc[:context.num_train].
        reset_index(drop=True),
        predict_instance_table=instance_df.iloc[context.num_train:].
        reset_index(drop=True).drop(
            columns=[context.y_train.name or 'TARGET'],
            errors='ignore',
        ),
        related_table_frames=related_table_frames,
        context_related_tables=context_related_tables,
        predict_related_tables=predict_related_tables,
        related_table_primary_keys=related_table_primary_keys,
        related_table_stype_overrides=related_table_stype_overrides,
        related_table_key_columns=related_table_key_columns,
        entity_reference_columns=entity_reference_columns,
        anchor_time_column=anchor_time_column,
        relationships=relationships,
    )


def _instance_dataframe(
    context: Context,
) -> tuple[pd.DataFrame, dict[str, str], str | None]:
    # Request-local transport keys are independent of user entity values.
    df = pd.DataFrame({INSTANCE_ID: range(context.subgraph.batch_size)})
    occupied = {INSTANCE_ID, context.y_train.name or 'TARGET'}
    if context.task_table is not None:
        occupied.update(context.task_table.df.columns)

    entity_names = context.entity_table_names
    entity_reference_columns: dict[str, str] = {}
    for i, table_name in enumerate(entity_names):
        values = _entity_values(context, table_name)
        if values is None:
            raise ValueError(
                f"Entity table {table_name!r} does not expose a primary key "
                "for Universal payload materialization.")
        column_name = _unique_internal_column(
            occupied,
            f'{ENTITY_REFERENCE_PREFIX}_{i}',
        )
        occupied.add(column_name)
        entity_reference_columns[table_name] = column_name
        df[column_name] = values

    anchor_values = np.asarray(context.subgraph.anchor_time, dtype=np.int64)
    anchor_time_column: str | None = None
    if len(anchor_values) and not np.all(
            anchor_values == pd.Timestamp.min.value):
        anchor_time_column = _unique_internal_column(
            occupied,
            ANCHOR_TIME_PREFIX,
        )
        occupied.add(anchor_time_column)
        df[anchor_time_column] = pd.to_datetime(anchor_values)

    target_name = context.y_train.name or 'TARGET'
    df[target_name] = None
    train_targets = [
        _target_json_value(context.task_type, value)
        for value in context.y_train.tolist()
    ]
    if train_targets:
        df.iloc[:context.num_train,
                df.columns.get_loc(target_name)] = _object_array(train_targets)
    if context.y_test is not None:
        test_targets = [
            _target_json_value(context.task_type, value)
            for value in context.y_test.tolist()
        ]
        if test_targets:
            df.iloc[
                context.num_train:context.num_train + len(test_targets),
                df.columns.get_loc(target_name),
            ] = _object_array(test_targets)
    elif context.num_test > 0:
        df.loc[context.num_train:, target_name] = None

    if context.task_table is not None:
        for column_name in context.task_table.df.columns:
            df[column_name] = context.task_table.df[column_name].tolist()

    return df, entity_reference_columns, anchor_time_column


def _object_array(values: list[Any]) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


def _entity_values(context: Context, table_name: str) -> list[Any] | None:
    table = context.subgraph.table_dict.get(table_name)
    if table is None or table.primary_key is None:
        return None
    if table.primary_key not in table.df:
        return None

    values: list[Any] = []
    batch = np.asarray(table.batch)
    df = table.df.reset_index(drop=True)
    row = np.asarray(table.row) if table.row is not None else None
    # One pass for every instance's first occurrence, rather than a full scan
    # of `batch` per instance (`np.unique` returns the first index per value
    # whether or not `batch` is sorted).
    first_occurrence: dict[int, int] = {}
    if batch.size > 0:
        unique, first = np.unique(batch, return_index=True)
        first_occurrence = dict(zip(unique.tolist(), first.tolist()))
    for instance_id in range(context.subgraph.batch_size):
        position = first_occurrence.get(instance_id)
        if position is None:
            values.append(None)
            continue
        row_index = position if row is None else int(row[position])
        if row_index >= len(df):
            values.append(None)
            continue
        values.append(_json_value(df.iloc[row_index][table.primary_key]))
    return values


def _occurrence_dataframe(table: Any) -> tuple[pd.DataFrame, str]:
    df = table.df.reset_index(drop=True)
    batch = np.asarray(table.batch)
    if table.row is not None:
        row = np.asarray(table.row)
        if len(row) == len(batch) and len(row) > 0:
            if int(row.max()) >= len(df) or int(row.min()) < 0:
                raise ValueError(
                    "Sampled RFM context references rows outside the table "
                    "frame.")
            df = df.iloc[row].reset_index(drop=True)
        elif len(row) == 0:
            df = df.iloc[0:0].copy()
        elif len(df) != len(batch):
            raise ValueError(
                "Sampled RFM context has incompatible row and batch lengths.")
    elif len(df) != len(batch):
        raise ValueError(
            "Sampled RFM context cannot be converted to table-oriented JSON "
            "because rows are not aligned with batch instances.")
    if len(df) != len(batch):
        raise ValueError(
            "Sampled RFM context has incompatible table and batch lengths.")

    df = df.copy(deep=False)
    df.insert(0, INSTANCE_ID, batch)
    if table.primary_key is None or table.primary_key not in df:
        key_column = _unique_internal_column(
            set(df.columns),
            SYNTHETIC_NODE_ID,
        )
        df.insert(1, key_column, np.arange(len(df), dtype=np.int64))
    else:
        key_column = table.primary_key
    return df, key_column


def _populate_relationship_columns(
    context: Context,
    instance_df: pd.DataFrame,
    related_table_frames: dict[str, pd.DataFrame],
    related_table_key_columns: dict[str, str],
    related_table_stype_overrides: dict[str, dict[str, Stype]],
    entity_reference_columns: dict[str, str],
) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []

    for table_name in context.entity_table_names:
        source_column = entity_reference_columns[table_name]
        target_column = related_table_key_columns.get(table_name)
        if target_column is None:
            raise ValueError(
                f"Entity table {table_name!r} is missing from the sampled "
                "subgraph.")
        table = context.subgraph.table_dict[table_name]
        if table.primary_key is None or target_column != table.primary_key:
            raise ValueError(
                f"Entity table {table_name!r} requires a declared scalar "
                "primary key.")
        relationships.append({
            'source_columns': [source_column],
            'target_table': table_name,
            'target_columns': [target_column],
        })

    processed_edges: set[tuple[str, str, str]] = set()
    for edge_type, link in context.subgraph.link_dict.items():
        logical_edge_type = _logical_forward_edge_type(edge_type)
        if logical_edge_type in processed_edges:
            continue
        src_table, fkey, dst_table = logical_edge_type
        if src_table not in related_table_frames:
            continue
        if dst_table not in related_table_frames:
            continue

        src_df = related_table_frames[src_table]
        dst_df = related_table_frames[dst_table]
        dst_key = related_table_key_columns[dst_table]
        edge_pairs = _logical_forward_edge_pairs(context, edge_type, link)
        if not edge_pairs:
            continue

        # Gather from the key column in one take: `dst_df.iloc[i][dst_key]`
        # materializes a whole row Series per edge, which dominates request
        # assembly on any non-trivial subgraph. Later pairs still overwrite
        # earlier ones, as the equivalent loop did.
        fkey_values: list[Any] = [None] * len(src_df)
        pairs = np.asarray(edge_pairs, dtype=np.int64).reshape(-1, 2)
        in_range = (pairs[:, 0] < len(src_df)) & (pairs[:, 1] < len(dst_df))
        src_positions = pairs[in_range, 0].tolist()
        if not src_positions:
            continue
        gathered = dst_df[dst_key].iloc[pairs[in_range, 1]]
        for position, value in zip(src_positions, gathered):
            fkey_values[position] = _json_value(value)
        if all(value is None for value in fkey_values):
            continue

        src_df[fkey] = fkey_values
        related_table_stype_overrides[src_table][fkey] = Stype.ID
        relationships.append({
            'source_table': src_table,
            'source_columns': [fkey],
            'target_table': dst_table,
            'target_columns': [dst_key],
        })
        processed_edges.add(logical_edge_type)

    return relationships


def _logical_forward_edge_type(
        edge_type: tuple[str, str, str]) -> tuple[str, str, str]:
    src_table, fkey, dst_table = edge_type
    if fkey.startswith(REV_REL):
        return (dst_table, fkey[len(REV_REL):], src_table)
    return edge_type


def _logical_forward_edge_pairs(
    context: Context,
    edge_type: tuple[str, str, str],
    link: Any,
) -> list[tuple[int, int]]:
    pairs = _edge_pairs(context, edge_type, link)
    if edge_type[1].startswith(REV_REL):
        return [(dst, src) for src, dst in pairs]
    return pairs


def _edge_pairs(
    context: Context,
    edge_type: tuple[str, str, str],
    link: Any,
) -> list[tuple[int, int]]:
    if link.layout == EdgeLayout.REV:
        rev_edge_type = context.subgraph.rev_edge_type(edge_type)
        rev_link = context.subgraph.link_dict.get(rev_edge_type)
        if rev_link is None:
            return []
        return [(dst, src)
                for src, dst in _edge_pairs(context, rev_edge_type, rev_link)]

    num_edges = int(link.num_edges)
    if num_edges == 0:
        return []

    if link.layout == EdgeLayout.CSC:
        if link.col is None:
            return []
        target_count = context.subgraph.table_dict[edge_type[2]].num_rows
        colptr = np.asarray(link.col)
        if len(colptr) != target_count + 1:
            return []
        row = (np.arange(num_edges, dtype=np.int64)
               if link.row is None else np.asarray(link.row))
        pairs: list[tuple[int, int]] = []
        for dst_index in range(target_count):
            start = int(colptr[dst_index])
            end = int(colptr[dst_index + 1])
            pairs.extend((int(src_index), dst_index)
                         for src_index in row[start:end])
        return pairs

    row = (np.arange(num_edges, dtype=np.int64)
           if link.row is None else np.asarray(link.row))
    col = (np.arange(num_edges, dtype=np.int64)
           if link.col is None else np.asarray(link.col))
    return [(int(src), int(dst)) for src, dst in zip(row, col)]


def _task_spec(context: Context, tables: PayloadTables) -> dict[str, Any]:
    target_name = context.y_train.name or 'TARGET'
    spec: dict[str, Any] = {
        'kind': TaskType(context.task_type).value,
        'target': {
            'column_name': target_name,
            'dtype': _target_dtype(context.task_type, context.y_train),
        },
        'entity_table_names': list(context.entity_table_names),
    }
    if tables.anchor_time_column is not None:
        spec['anchor_time_column'] = tables.anchor_time_column
    if TaskType(context.task_type) == TaskType.BINARY_CLASSIFICATION:
        spec['target']['classes'] = ['false', 'true']
        spec['target']['positive_class'] = 'true'
    elif TaskType(context.task_type) == TaskType.MULTICLASS_CLASSIFICATION:
        spec['target']['classes'] = [
            str(value) for value in pd.unique(context.y_train)
        ]
    if context.top_k is not None:
        spec['top_k'] = context.top_k
    if context.step_size is not None:
        spec['step_size'] = context.step_size
    if context.num_forecasts != 1:
        spec['num_forecasts'] = context.num_forecasts
    return spec


def _schema_spec(
    context: Context,
    tables: PayloadTables,
) -> dict[str, Any]:
    spec = {
        'instance_table':
        _schema_for_dataframe(
            pd.concat([
                _instance_payload_dataframe(
                    tables.context_instance_table,
                    context,
                ),
                _instance_payload_dataframe(
                    tables.predict_instance_table,
                    context,
                ),
            ],
                      ignore_index=True),
            stype_overrides={
                INSTANCE_ID: Stype.ID,
                **{
                    column: Stype.ID
                    for column in tables.entity_reference_columns.values()
                },
                **({
                    tables.anchor_time_column: Stype.timestamp
                } if tables.anchor_time_column is not None else {}),
                context.y_train.name or 'TARGET': _target_stype(
                    context.task_type),
            },
            primary_key=_instance_primary_key(context),
        ),
        'related_tables': {
            table_name:
            _schema_for_dataframe(
                df,
                stype_overrides=tables.related_table_stype_overrides[
                    table_name],
                primary_key=tables.related_table_primary_keys[table_name],
            )
            for table_name, df in tables.related_table_frames.items()
        },
        'relationships': tables.relationships,
    }
    target_name = context.y_train.name or 'TARGET'
    spec['instance_table']['columns'][target_name]['dtype'] = _target_dtype(
        context.task_type,
        context.y_train,
    )
    if tables.anchor_time_column is not None:
        anchor_schema = spec['instance_table']['columns'][
            tables.anchor_time_column
        ]
        anchor_schema['nullable'] = False
    return spec


def _schema_for_dataframe(
    df: pd.DataFrame,
    *,
    stype_overrides: dict[str, Stype],
    primary_key: str | list[str],
) -> dict[str, Any]:
    primary_keys = (
        {primary_key} if isinstance(primary_key, str) else set(primary_key)
    )
    return {
        'columns': {
            column_name: {
                'dtype': _dtype_name(df[column_name]),
                **({
                    'stype': _stype_name(stype_overrides[column_name])
                } if column_name in stype_overrides else {}),
                **({
                    'nullable': False
                } if column_name in primary_keys else {}),
            }
            for column_name in df.columns
        },
        'primary_key': primary_key,
    }


def _instance_primary_key(context: Context) -> str:
    return INSTANCE_ID


def _instance_payload_dataframe(df: pd.DataFrame,
                                context: Context) -> pd.DataFrame:
    return df.copy(deep=False)


def _unique_internal_column(occupied: set[str], prefix: str) -> str:
    if prefix not in occupied:
        return prefix
    suffix = 1
    while f'{prefix}_{suffix}' in occupied:
        suffix += 1
    return f'{prefix}_{suffix}'


def _dataframe_table(
    df: pd.DataFrame,
    table_name: str | None = None,
) -> dict[str, Any]:
    columns = df.columns.tolist()
    rows = []
    for row_index, row in enumerate(df.itertuples(index=False, name=None)):
        cells = []
        for column_index, value in enumerate(row):
            try:
                cells.append(_cell_json_value(value))
            except ValueError as error:
                column = columns[column_index]
                qualified = (f"'{table_name}.{column}'"
                             if table_name is not None else f"'{column}'")
                raise ValueError(
                    f"Column {qualified} row {row_index}: {error}") from error
        rows.append(cells)
    return {'format': 'arrays', 'columns': columns, 'rows': rows}


def _output_fields(
    task_type: TaskType,
    *,
    return_embeddings: bool,
    explain: bool,
    quantile_output: bool = False,
) -> list[str]:
    if TaskType(task_type).is_link_pred:
        fields = [TFM_OUTPUT_FIELD_RANKINGS]
    elif quantile_output:
        fields = [TFM_OUTPUT_FIELD_QUANTILES]
    else:
        fields = [TFM_OUTPUT_FIELD_PREDICTION]
    if TaskType(task_type).is_classification:
        fields.append(TFM_OUTPUT_FIELD_PROBABILITIES)
    if return_embeddings:
        fields.append(TFM_OUTPUT_FIELD_EMBEDDINGS)
    if explain:
        fields.append(TFM_OUTPUT_FIELD_EXPLANATION)
    return fields


def _inference_config_to_json(
        inference_config: InferenceConfig | None) -> dict[str, Any] | None:
    if inference_config is None:
        return None

    out: dict[str, Any] = {
        'kind': 'base',
        'num_estimators': int(inference_config.num_estimators),
        'column_shuffle': bool(inference_config.column_shuffle),
        'category_shuffle': bool(inference_config.category_shuffle),
        'hop_shuffle': bool(inference_config.hop_shuffle),
    }
    if isinstance(inference_config, ClassificationInferenceConfig):
        out['kind'] = 'classification'
        out['class_shuffle'] = bool(inference_config.class_shuffle)
    elif isinstance(inference_config, RegressionInferenceConfig):
        out['kind'] = 'regression'
        out['target_transforms'] = [
            _json_value(value) for value in inference_config.target_transforms
        ]
        out['output_type'] = inference_config.output_type
    return out


def _target_stype(task_type: TaskType) -> Stype:
    if TaskType(task_type).is_link_pred:
        return Stype.multicategorical
    if TaskType(task_type).is_classification:
        return Stype.categorical
    return Stype.numerical


def _target_dtype(task_type: TaskType, target: pd.Series) -> str:
    if TaskType(task_type).is_link_pred:
        return 'stringlist'
    if TaskType(task_type) == TaskType.BINARY_CLASSIFICATION:
        return 'bool'
    return _dtype_name(target)


def _target_json_value(task_type: TaskType, value: Any) -> Any:
    task_type = TaskType(task_type)
    if task_type.is_link_pred:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                "Link prediction target values must be stringlist arrays, "
                f"but got {value!r}.")
        result: list[str] = []
        for item in value:
            if _is_null_like(item):
                raise ValueError(
                    "Link prediction target stringlist values must not contain "
                    f"null items, but got {value!r}.")
            item_value = _json_value(item)
            if item_value is None:
                raise ValueError(
                    "Link prediction target stringlist values must not contain "
                    f"null items, but got {value!r}.")
            result.append(str(item_value))
        return result

    value = _json_value(value)
    if task_type != TaskType.BINARY_CLASSIFICATION or value is None:
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, Real) and value in (0, 1):
        return bool(value)
    raise ValueError(
        "Binary classification target values must be booleans or numeric "
        f"0/1 values, but got {value!r}.")


def _dtype_name(data: pd.Series) -> str:
    list_dtype = _list_dtype_name(data)
    if list_dtype is not None:
        return list_dtype

    dtype = data.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return 'bool'
    if pd.api.types.is_integer_dtype(dtype):
        return 'int64'
    if pd.api.types.is_float_dtype(dtype):
        return 'float32' if dtype.itemsize <= 4 else 'float64'
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return 'timestamp[us]'
    return 'string'


def _list_dtype_name(data: pd.Series) -> str | None:
    if not pd.api.types.is_object_dtype(data.dtype):
        return None

    found_list = False
    found_scalar = False
    for value in data:
        if _is_null_like(value):
            continue
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            found_scalar = True
            continue
        found_list = True
        for item in value:
            if _is_null_like(item):
                raise ValueError(
                    "stringlist columns must not contain null items, "
                    f"but got {value!r}.")
            item_value = _json_value(item)
            if item_value is None:
                raise ValueError(
                    "stringlist columns must not contain null items, "
                    f"but got {value!r}.")

    if found_list and found_scalar:
        raise ValueError(
            "Columns with stringlist values must contain only arrays or nulls.")
    if not found_list:
        return None
    return 'stringlist'


def _is_null_like(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def _stype_name(stype: Stype) -> str:
    if stype == Stype.ID:
        return 'ID'
    return Stype(stype).value


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return _timestamp_json_value(value)
    if isinstance(value, np.datetime64):
        return _timestamp_json_value(pd.Timestamp(value))
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return None
        raise ValueError(_NON_FINITE_MSG.format(value=value))
    if isinstance(value, Decimal):
        if value.is_nan():
            return None
        if not value.is_finite():
            raise ValueError(_NON_FINITE_MSG.format(value=value))
        return str(value)  # The wire dtype of a `Decimal` column is `string`.
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _cell_json_value(value: Any) -> Any:
    r"""Serializes a table cell, honoring the JSON safe-integer rule.

    ``int64`` values outside the JavaScript safe-integer range must travel as
    base-10 strings; sending them as JSON numbers is rejected by the NIM.
    """
    value = _json_value(value)
    if isinstance(value, int) and not isinstance(value, bool):
        if value > JSON_SAFE_INT_MAX or value < JSON_SAFE_INT_MIN:
            return str(value)
    return value


def _timestamp_json_value(value: pd.Timestamp) -> str | None:
    if pd.isna(value):
        return None
    if value.tzinfo is not None:
        value = value.tz_convert('UTC').tz_localize(None)
    return value.isoformat(timespec='microseconds') + 'Z'

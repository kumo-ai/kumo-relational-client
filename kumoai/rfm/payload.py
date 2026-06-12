import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from kumoapi.model_plan import RunMode
from kumoapi.rfm import RFMEvaluateRequest, RFMPredictRequest
from kumoapi.rfm.context import Context, EdgeLayout, REV_REL
from kumoapi.rfm.inference import (
    ClassificationInferenceConfig,
    InferenceConfig,
    RegressionInferenceConfig,
)
from kumoapi.task import TaskType
from kumoapi.typing import Stype

INSTANCE_ID = 'instance_id'
SYNTHETIC_NODE_ID = '__node_id'


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
    relationships: list[dict[str, Any]]


def predict_request_to_json(
    request: RFMPredictRequest,
    *,
    explain: bool = False,
    skip_summary: bool = False,
) -> dict[str, Any]:
    output_fields = _output_fields(
        request.context.task_type,
        return_embeddings=request.return_embeddings,
        explain=explain,
    )
    payload = _base_payload(
        context=request.context,
        run_mode=request.run_mode,
        use_prediction_time=request.use_prediction_time,
        inference_config=request.inference_config,
        output={'fields': output_fields},
        metadata={
            'source_query': request.query,
            'operation': 'explain' if explain else 'predict',
        },
    )
    if explain:
        payload['metadata']['explain'] = {
            'generate_summary': not skip_summary,
        }
    return payload


def evaluate_request_to_json(request: RFMEvaluateRequest) -> dict[str, Any]:
    output: dict[str, Any] = {'fields': ['metrics']}
    if request.metrics is not None:
        output['metrics'] = request.metrics
    return _base_payload(
        context=request.context,
        run_mode=request.run_mode,
        use_prediction_time=request.use_prediction_time,
        inference_config=request.inference_config,
        output=output,
        metadata={'operation': 'evaluate'},
    )


def payload_size_bytes(payload: dict[str, Any]) -> int:
    return len(
        json.dumps(
            payload,
            allow_nan=False,
            separators=(',', ':'),
        ).encode('utf-8'))


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
        'version': 'v1',
        'model': 'kumo-rfm',
        'task': _task_spec(context),
        'schema': _schema_spec(context, tables),
        'context': {
            'instance_table':
            _dataframe_table(tables.context_instance_table),
            'related_tables': {
                table_name: _dataframe_table(df)
                for table_name, df in tables.context_related_tables.items()
            },
        },
        'predict': {
            'instance_table':
            _dataframe_table(tables.predict_instance_table),
            'related_tables': {
                table_name: _dataframe_table(df)
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
    instance_df = _instance_dataframe(context)
    related_table_frames: dict[str, pd.DataFrame] = {}
    related_table_primary_keys: dict[str, str | list[str]] = {}
    related_table_stype_overrides: dict[str, dict[str, Stype]] = {}
    related_table_key_columns: dict[str, str] = {}

    for table_name, table in context.subgraph.table_dict.items():
        df = _occurrence_dataframe(table)
        key_column = _table_key_column(table, df)
        stype_overrides = {INSTANCE_ID: Stype.ID, **table.stype_dict}
        if key_column == SYNTHETIC_NODE_ID:
            stype_overrides[SYNTHETIC_NODE_ID] = Stype.ID

        related_table_frames[table_name] = df
        related_table_key_columns[table_name] = key_column
        related_table_primary_keys[table_name] = [INSTANCE_ID, key_column]
        related_table_stype_overrides[table_name] = stype_overrides

    relationships = _populate_relationship_columns(
        context,
        instance_df,
        related_table_frames,
        related_table_key_columns,
        related_table_stype_overrides,
    )

    context_related_tables: dict[str, pd.DataFrame] = {}
    predict_related_tables: dict[str, pd.DataFrame] = {}
    for table_name, df in related_table_frames.items():
        if INSTANCE_ID not in df:
            context_related_tables[table_name] = df.copy(deep=False)
            predict_related_tables[table_name] = df.iloc[0:0].copy()
            continue
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
        reset_index(drop=True),
        related_table_frames=related_table_frames,
        context_related_tables=context_related_tables,
        predict_related_tables=predict_related_tables,
        related_table_primary_keys=related_table_primary_keys,
        related_table_stype_overrides=related_table_stype_overrides,
        related_table_key_columns=related_table_key_columns,
        relationships=relationships,
    )


def _instance_dataframe(context: Context) -> pd.DataFrame:
    df = pd.DataFrame({INSTANCE_ID: range(context.subgraph.batch_size)})
    entity_names = context.entity_table_names
    for i, table_name in enumerate(entity_names):
        values = _entity_values(context, table_name)
        if values is None:
            continue
        column_name = 'ENTITY' if len(entity_names) == 1 else f'ENTITY_{i}'
        df[column_name] = values

    df['ANCHOR_TIMESTAMP'] = pd.to_datetime(context.subgraph.anchor_time)

    target_name = context.y_train.name or 'TARGET'
    df[target_name] = None
    df.loc[:context.num_train - 1, target_name] = [
        _json_value(value) for value in context.y_train.tolist()
    ]
    if context.y_test is not None:
        df.loc[context.num_train:, target_name] = [
            _json_value(value) for value in context.y_test.tolist()
        ]
    elif context.num_test > 0:
        df.loc[context.num_train:, target_name] = None

    if context.task_table is not None:
        for column_name in context.task_table.df.columns:
            df[column_name] = context.task_table.df[column_name].tolist()

    return df


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
    for instance_id in range(context.subgraph.batch_size):
        rows = np.flatnonzero(batch == instance_id)
        if len(rows) == 0:
            values.append(None)
            continue
        row_index = int(rows[0])
        if row is not None:
            row_index = int(row[row_index])
        if row_index >= len(df):
            values.append(None)
            continue
        values.append(_json_value(df.iloc[row_index][table.primary_key]))
    return values


def _occurrence_dataframe(table: Any) -> pd.DataFrame:
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
        df.insert(1, SYNTHETIC_NODE_ID, np.arange(len(df), dtype=np.int64))
    return df


def _table_key_column(table: Any, df: pd.DataFrame) -> str:
    if table.primary_key is not None and table.primary_key in df:
        return table.primary_key
    return SYNTHETIC_NODE_ID


def _populate_relationship_columns(
    context: Context,
    instance_df: pd.DataFrame,
    related_table_frames: dict[str, pd.DataFrame],
    related_table_key_columns: dict[str, str],
    related_table_stype_overrides: dict[str, dict[str, Stype]],
) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []

    for i, table_name in enumerate(context.entity_table_names):
        source_column = 'ENTITY' if len(
            context.entity_table_names) == 1 else f'ENTITY_{i}'
        target_column = related_table_key_columns.get(table_name)
        if source_column not in instance_df or target_column is None:
            continue
        if target_column == SYNTHETIC_NODE_ID:
            continue
        relationships.append({
            'source_columns': [INSTANCE_ID, source_column],
            'target_table': table_name,
            'target_columns': [INSTANCE_ID, target_column],
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

        fkey_values: list[Any] = [None] * len(src_df)
        for src_index, dst_index in edge_pairs:
            if src_index >= len(src_df) or dst_index >= len(dst_df):
                continue
            fkey_values[src_index] = _json_value(
                dst_df.iloc[dst_index][dst_key])
        if all(value is None for value in fkey_values):
            continue

        src_df[fkey] = fkey_values
        related_table_stype_overrides[src_table][fkey] = Stype.ID
        relationships.append({
            'source_table': src_table,
            'source_columns': [INSTANCE_ID, fkey],
            'target_table': dst_table,
            'target_columns': [INSTANCE_ID, dst_key],
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


def _task_spec(context: Context) -> dict[str, Any]:
    target_name = context.y_train.name or 'TARGET'
    spec: dict[str, Any] = {
        'kind': TaskType(context.task_type).value,
        'target': {
            'column_name': target_name,
            'dtype': _dtype_name(context.y_train),
        },
        'entity_table_names': list(context.entity_table_names),
    }
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
    return {
        'instance_table':
        _schema_for_dataframe(
            pd.concat([
                tables.context_instance_table,
                tables.predict_instance_table,
            ],
                      ignore_index=True),
            stype_overrides={
                INSTANCE_ID: Stype.ID,
                'ENTITY': Stype.ID,
                'ENTITY_0': Stype.ID,
                'ENTITY_1': Stype.ID,
                'ANCHOR_TIMESTAMP': Stype.timestamp,
                context.y_train.name or 'TARGET': _target_stype(
                    context.task_type),
            },
            primary_key=INSTANCE_ID,
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


def _schema_for_dataframe(
    df: pd.DataFrame,
    *,
    stype_overrides: dict[str, Stype],
    primary_key: str | list[str],
) -> dict[str, Any]:
    return {
        'columns': {
            column_name: {
                'dtype': _dtype_name(df[column_name]),
                **({
                    'stype': _stype_name(stype_overrides[column_name])
                } if column_name in stype_overrides else {}),
            }
            for column_name in df.columns
        },
        'primary_key': primary_key,
    }


def _dataframe_table(df: pd.DataFrame) -> dict[str, Any]:
    return {
        'format': 'arrays',
        'columns': df.columns.tolist(),
        'rows': [[_json_value(value) for value in row]
                 for row in df.itertuples(index=False, name=None)],
    }


def _output_fields(
    task_type: TaskType,
    *,
    return_embeddings: bool,
    explain: bool,
) -> list[str]:
    fields = ['prediction']
    if TaskType(task_type).is_classification:
        fields.append('probabilities')
    if return_embeddings:
        fields.append('embedding')
    if explain:
        fields.extend(['explanation', 'summary'])
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
    if TaskType(task_type).is_classification:
        return Stype.categorical
    return Stype.numerical


def _dtype_name(data: pd.Series) -> str:
    dtype = data.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return 'bool'
    if pd.api.types.is_integer_dtype(dtype):
        return 'int64'
    if pd.api.types.is_float_dtype(dtype):
        return 'float32' if dtype.itemsize <= 4 else 'float64'
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return 'timestamp[ns]'
    return 'string'


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
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.datetime64):
        ts = pd.Timestamp(value)
        return None if pd.isna(ts) else ts.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value

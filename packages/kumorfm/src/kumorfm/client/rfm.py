from collections.abc import Mapping, Sequence
import math
from typing import Any

from kumorfm.api.rfm import RFMPredictResponse

from kumorfm.client import KumoClient
from kumorfm.client.generated.tfm_api import (
    PredictionItem,
    PredictionResponse,
    TFMOperations,
)
from kumorfm.client.utils import raise_on_error


class RFMAPI:
    r"""Typed API definition for Kumo RFM (Relational Foundation Model)."""
    def __init__(self, client: KumoClient) -> None:
        self._client = client

    def predict(
        self,
        request: Mapping[str, Any],
        *,
        entity_ids: Sequence[Any],
        instance_ids: Sequence[Any],
        anchor_times: Sequence[Any] | None = None,
    ) -> RFMPredictResponse:
        """Make predictions using the RFM model.

        Args:
            request: The predict request as a universal TFM JSON envelope.
            entity_ids: Private, batch-local entity values ordered like the
                predict instance table. Response row indexes are correlated
                back to these values.
            instance_ids: Generated transport keys ordered like the predict
                instance table, used only to validate response correlation.
            anchor_times: Anchor timestamps ordered like the predict instance
                table. The NIM response does not echo them back, so they are
                re-attached here to keep the ``ANCHOR_TIMESTAMP`` column of
                the prediction table.

        Returns:
            RFMPredictResponse containing the predictions
        """
        response = self._client._request(
            TFMOperations.run_prediction.endpoint,
            json=request,
            headers={'Content-Type': 'application/json'},
        )
        raise_on_error(response)
        prediction_response = PredictionResponse.from_dict(response.json())
        return _prediction_response_to_rfm(
            prediction_response,
            entity_ids=entity_ids,
            instance_ids=instance_ids,
            anchor_times=anchor_times,
        )


def _prediction_response_to_rfm(
    response: PredictionResponse,
    *,
    entity_ids: Sequence[Any],
    instance_ids: Sequence[Any],
    anchor_times: Sequence[Any] | None = None,
) -> RFMPredictResponse:
    rows = _correlated_prediction_rows(
        response,
        entity_ids=entity_ids,
        instance_ids=instance_ids,
        anchor_times=anchor_times,
    )

    columns: list[str] = []
    for row in rows:
        for column_name in row:
            if column_name not in columns:
                columns.append(column_name)
    return RFMPredictResponse(prediction={
        'columns': columns,
        'data': [[row.get(column_name) for column_name in columns]
                 for row in rows],
    })


def _prediction_item_to_row(
    item: PredictionItem,
    *,
    entity_id: Any,
    anchor_time: Any = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {'ENTITY': entity_id}
    if anchor_time is not None:
        row['ANCHOR_TIMESTAMP'] = anchor_time
    if item.prediction is not None:
        row['PREDICTION'] = item.prediction
    if item.probabilities is not None:
        for name, value in item.probabilities.items():
            row[f'{name}_PROB'] = value
    if item.scores is not None:
        row['scores'] = list(item.scores)
    if item.rankings is not None:
        row['rankings'] = [dict(ranking) for ranking in item.rankings]
    if item.embeddings is not None:
        row['embeddings'] = list(item.embeddings)
    if item.quantiles is not None:
        for name, value in item.quantiles.items():
            row[f'q_{name}'] = value
    if item.explanation is not None:
        row['explanation'] = item.explanation
    return row


def _prediction_item_to_ranking_rows(
    item: PredictionItem,
    *,
    entity_id: Any,
    anchor_time: Any = None,
) -> list[dict[str, Any]]:
    if item.rankings is None:
        raise ValueError('Kumo RFM ranking response is missing rankings.')
    if item.prediction is not None:
        raise ValueError(
            'Kumo RFM ranking response must not include prediction.')

    rows: list[dict[str, Any]] = []
    for ranking in item.rankings:
        if 'id' not in ranking:
            raise ValueError('Kumo RFM ranking item is missing id.')
        if 'score' not in ranking:
            raise ValueError('Kumo RFM ranking item is missing score.')
        extra_fields = set(ranking) - {'id', 'score'}
        if extra_fields:
            unexpected = ', '.join(sorted(extra_fields))
            raise ValueError(
                'Kumo RFM ranking item includes unexpected fields: '
                f'{unexpected}.')
        ranking_id = ranking['id']
        if not isinstance(ranking_id, str):
            raise ValueError('Kumo RFM ranking item id must be a string.')
        score = float(ranking['score'])
        if not math.isfinite(score):
            raise ValueError('Kumo RFM ranking item score must be finite.')

        row: dict[str, Any] = {'ENTITY': entity_id}
        if anchor_time is not None:
            row['ANCHOR_TIMESTAMP'] = anchor_time
        row['CLASS'] = ranking_id
        row['SCORE'] = score
        if item.explanation is not None:
            row['explanation'] = item.explanation
        rows.append(row)
    return rows


def _correlated_prediction_rows(
    response: PredictionResponse,
    *,
    entity_ids: Sequence[Any],
    instance_ids: Sequence[Any],
    anchor_times: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    entities = list(entity_ids)
    instances = list(instance_ids)
    anchors = list(anchor_times) if anchor_times is not None else None
    expected_count = len(entities)
    if len(instances) != expected_count:
        raise ValueError(
            'Kumo RFM request identity mappings have different lengths: '
            f'{expected_count} entities and {len(instances)} instances.')
    if anchors is not None and len(anchors) != expected_count:
        raise ValueError(
            'Kumo RFM request identity mappings have different lengths: '
            f'{expected_count} entities and {len(anchors)} anchor times.')
    is_forecast = _is_forecast_response(response)
    forecast_steps_by_index: dict[int, set[int]] = {}
    if not is_forecast and len(response.predictions) != expected_count:
        raise ValueError(
            'Kumo RFM prediction response count does not match the request: '
            f'expected {expected_count}, got {len(response.predictions)}.')

    rows_by_index: dict[int, list[dict[str, Any]]] = {}
    for item in response.predictions:
        row_index = item.row_index
        if row_index is None:
            raise ValueError(
                'Kumo RFM prediction response is missing row_index.')
        if row_index < 0 or row_index >= expected_count:
            raise ValueError(
                'Kumo RFM prediction response row_index is out of range: '
                f'{row_index}.')
        if is_forecast:
            if item.forecast_step is None:
                raise ValueError(
                    'Kumo RFM forecasting response is missing forecast_step.')
            if item.forecast_step <= 0:
                raise ValueError(
                    'Kumo RFM forecasting response forecast_step must be positive.')
            forecast_steps = forecast_steps_by_index.setdefault(row_index, set())
            if item.forecast_step in forecast_steps:
                raise ValueError(
                    'Kumo RFM forecasting response contains duplicate '
                    f'forecast_step {item.forecast_step} for row_index {row_index}.')
            forecast_steps.add(item.forecast_step)
        elif row_index in rows_by_index:
            raise ValueError(
                'Kumo RFM prediction response contains duplicate row_index: '
                f'{row_index}.')

        expected_id = str(instances[row_index])
        if item.id != expected_id:
            raise ValueError(
                'Kumo RFM prediction response id does not match request '
                f'instance_id at row_index {row_index}: expected '
                f'{expected_id!r}, got {item.id!r}.')
        rows = _prediction_item_rows_for_response(
            response,
            item,
            entity_id=entities[row_index],
            anchor_time=anchors[row_index] if anchors is not None else None,
        )
        rows_by_index.setdefault(row_index, []).extend(rows)

    missing = [index for index in range(expected_count)
               if index not in rows_by_index]
    if missing:
        raise ValueError(
            'Kumo RFM prediction response is missing row_index values: '
            f'{missing}.')

    return [row for index in range(expected_count)
            for row in rows_by_index[index]]


def _prediction_item_rows_for_response(
    response: PredictionResponse,
    item: PredictionItem,
    *,
    entity_id: Any,
    anchor_time: Any = None,
) -> list[dict[str, Any]]:
    if _is_ranking_response(response):
        return _prediction_item_to_ranking_rows(
            item, entity_id=entity_id, anchor_time=anchor_time)
    row = _prediction_item_to_row(
        item, entity_id=entity_id, anchor_time=anchor_time)
    if _is_forecast_response(response):
        row['forecast_step'] = item.forecast_step
    return [row]


def _is_ranking_response(response: PredictionResponse) -> bool:
    task_kind = str(response.metadata.get('task_kind', '')).lower()
    output_type = str(response.metadata.get('output_type', '')).lower()
    return (
        task_kind in {'ranking', 'temporal_link_prediction'}
        or output_type == 'rankings'
    )


def _is_forecast_response(response: PredictionResponse) -> bool:
    task_kind = str(response.metadata.get('task_kind', '')).lower()
    output_type = str(response.metadata.get('output_type', '')).lower()
    return task_kind in {'forecast', 'forecasting'} or output_type == 'forecast'

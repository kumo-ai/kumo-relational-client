from collections.abc import Mapping, Sequence
from typing import Any

from kumoapi.rfm import RFMPredictResponse

from kumoai.client import KumoClient
from kumoai.client.generated.tfm_api import (
    PredictionItem,
    PredictionResponse,
    TFMOperations,
)
from kumoai.client.utils import raise_on_error


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
    ) -> RFMPredictResponse:
        """Make predictions using the RFM model.

        Args:
            request: The predict request as a universal TFM JSON envelope.
            entity_ids: Private, batch-local entity values ordered like the
                predict instance table. Response row indexes are correlated
                back to these values.
            instance_ids: Generated transport keys ordered like the predict
                instance table, used only to validate response correlation.

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
        )


def _prediction_response_to_rfm(
    response: PredictionResponse,
    *,
    entity_ids: Sequence[Any],
    instance_ids: Sequence[Any],
) -> RFMPredictResponse:
    rows = _correlated_prediction_rows(
        response,
        entity_ids=entity_ids,
        instance_ids=instance_ids,
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
) -> dict[str, Any]:
    row: dict[str, Any] = {'ENTITY': entity_id}
    if item.prediction is not None:
        row['prediction'] = item.prediction
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


def _correlated_prediction_rows(
    response: PredictionResponse,
    *,
    entity_ids: Sequence[Any],
    instance_ids: Sequence[Any],
) -> list[dict[str, Any]]:
    entities = list(entity_ids)
    instances = list(instance_ids)
    expected_count = len(entities)
    if len(instances) != expected_count:
        raise ValueError(
            'Kumo RFM request identity mappings have different lengths: '
            f'{expected_count} entities and {len(instances)} instances.')
    if len(response.predictions) != expected_count:
        raise ValueError(
            'Kumo RFM prediction response count does not match the request: '
            f'expected {expected_count}, got {len(response.predictions)}.')

    rows_by_index: dict[int, dict[str, Any]] = {}
    for item in response.predictions:
        row_index = item.row_index
        if row_index is None:
            raise ValueError(
                'Kumo RFM prediction response is missing row_index.')
        if row_index < 0 or row_index >= expected_count:
            raise ValueError(
                'Kumo RFM prediction response row_index is out of range: '
                f'{row_index}.')
        if row_index in rows_by_index:
            raise ValueError(
                'Kumo RFM prediction response contains duplicate row_index: '
                f'{row_index}.')

        expected_id = str(instances[row_index])
        if item.id != expected_id:
            raise ValueError(
                'Kumo RFM prediction response id does not match request '
                f'instance_id at row_index {row_index}: expected '
                f'{expected_id!r}, got {item.id!r}.')
        rows_by_index[row_index] = _prediction_item_to_row(
            item,
            entity_id=entities[row_index],
        )

    return [rows_by_index[index] for index in range(expected_count)]

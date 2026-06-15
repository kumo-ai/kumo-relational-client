from collections.abc import Mapping
from typing import Any

from kumoapi.json_serde import to_json_dict
from kumoapi.rfm import (
    RFMEvaluateResponse,
    RFMParseQueryRequest,
    RFMParseQueryResponse,
    RFMPredictResponse,
    RFMValidateQueryRequest,
    RFMValidateQueryResponse,
)

from kumoai.client import KumoClient
from kumoai.client.endpoints import RFMEndpoints
from kumoai.client.generated.tfm_api import (
    PredictionItem,
    PredictionResponse,
    TFMOperations,
)
from kumoai.client.utils import parse_response, raise_on_error


class RFMAPI:
    r"""Typed API definition for Kumo RFM (Relational Foundation Model)."""
    def __init__(self, client: KumoClient) -> None:
        self._client = client

    def predict(self, request: Mapping[str, Any]) -> RFMPredictResponse:
        """Make predictions using the RFM model.

        Args:
            request: The predict request as a universal TFM JSON envelope.

        Returns:
            RFMPredictResponse containing the predictions
        """
        response = self._client._request(
            TFMOperations.create_prediction.endpoint,
            json=request,
            headers={'Content-Type': 'application/json'},
        )
        raise_on_error(response)
        prediction_response = PredictionResponse.from_dict(response.json())
        return _prediction_response_to_rfm(prediction_response)

    def evaluate(self, request: bytes) -> RFMEvaluateResponse:
        """Evaluate the RFM model on the given context.

        Args:
            request: The evaluate request as serialized protobuf.

        Returns:
            RFMEvaluateResponse containing the computed metrics
        """
        # Evaluation is intentionally not part of the TFM prediction spec.
        response = self._client._request(
            RFMEndpoints.evaluate, data=request,
            headers={'Content-Type': 'application/x-protobuf'})
        raise_on_error(response)
        return parse_response(RFMEvaluateResponse, response)

    def validate_query(
        self,
        request: RFMValidateQueryRequest,
    ) -> RFMValidateQueryResponse:
        """Validate a predictive query against a graph.

        Args:
            request: The request object containing
                the query and graph definition

        Returns:
            RFMValidateQueryResponse containing the QueryDefinition
        """
        response = self._client._request(RFMEndpoints.validate_query,
                                         json=to_json_dict(request))
        raise_on_error(response)
        return parse_response(RFMValidateQueryResponse, response)

    def parse_query(
        self,
        request: RFMParseQueryRequest,
    ) -> RFMParseQueryResponse:
        """Validate a predictive query against a graph.

        Args:
            request: The request object containing
                the query and graph definition

        Returns:
            RFMParseQueryResponse containing the QueryDefinition
        """
        response = self._client._request(RFMEndpoints.parse_query,
                                         json=to_json_dict(request))
        raise_on_error(response)
        return parse_response(RFMParseQueryResponse, response)


def _prediction_response_to_rfm(
        response: PredictionResponse) -> RFMPredictResponse:
    rows: list[dict[str, Any]] = [
        _prediction_item_to_row(item) for item in response.predictions
    ]
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


def _prediction_item_to_row(item: PredictionItem) -> dict[str, Any]:
    row: dict[str, Any] = {'ENTITY': _coerce_prediction_id(item.id)}
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


def _coerce_prediction_id(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        return value

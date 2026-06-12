from collections.abc import Mapping
from typing import Any

from kumoapi.json_serde import to_json_dict
from kumoapi.rfm import (
    RFMEvaluateResponse,
    RFMExplanationResponse,
    RFMParseQueryRequest,
    RFMParseQueryResponse,
    RFMPredictResponse,
    RFMValidateQueryRequest,
    RFMValidateQueryResponse,
)

from kumoai.client import KumoClient
from kumoai.client.endpoints import RFMEndpoints
from kumoai.client.generated.tfm_api import TFMOperations
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
        return parse_response(RFMPredictResponse, response)

    def explain(
        self,
        request: Mapping[str, Any],
        skip_summary: bool = False,
    ) -> RFMExplanationResponse:
        """Explain the RFM model on the given context.

        Args:
            request: The predict request as a universal TFM JSON envelope.
            skip_summary: Whether to skip generating a human-readable summary
                of the explanation.

        Returns:
            RFMPredictResponse containing the explanations
        """
        request = dict(request)
        metadata = dict(request.get('metadata', {}))
        metadata['explain'] = {'generate_summary': not skip_summary}
        request['metadata'] = metadata
        response = self._client._request(
            TFMOperations.create_prediction.endpoint,
            json=request,
            headers={'Content-Type': 'application/json'},
        )
        raise_on_error(response)
        return parse_response(RFMExplanationResponse, response)

    def evaluate(self, request: Mapping[str, Any]) -> RFMEvaluateResponse:
        """Evaluate the RFM model on the given context.

        Args:
            request: The evaluate request as a universal TFM JSON envelope.

        Returns:
            RFMEvaluateResponse containing the computed metrics
        """
        response = self._client._request(
            TFMOperations.create_prediction.endpoint,
            json=request,
            headers={'Content-Type': 'application/json'},
        )
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

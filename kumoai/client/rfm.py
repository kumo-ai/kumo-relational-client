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
from kumoai.client.utils import parse_response, raise_on_error


class RFMAPI:
    r"""Typed API definition for Kumo RFM (Relational Foundation Model)."""
    def __init__(self, client: KumoClient) -> None:
        self._client = client

    def predict(self, request: bytes) -> RFMPredictResponse:
        """Make predictions using the RFM model.

        Args:
            request: The predict request as serialized protobuf.

        Returns:
            RFMPredictResponse containing the predictions
        """
        response = self._client._request(
            RFMEndpoints.predict,
            data=request,
            headers={'Content-Type': 'application/x-protobuf'},
        )
        raise_on_error(response)
        return parse_response(RFMPredictResponse, response)

    def explain(
        self,
        request: bytes,
        skip_summary: bool = False,
    ) -> RFMExplanationResponse:
        """Explain the RFM model on the given context.

        Args:
            request: The predict request as serialized protobuf.
            skip_summary: Whether to skip generating a human-readable summary
                of the explanation.

        Returns:
            RFMPredictResponse containing the explanations
        """
        params: dict[str, Any] = {'generate_summary': not skip_summary}
        response = self._client._request(
            RFMEndpoints.explain, data=request, params=params,
            headers={'Content-Type': 'application/x-protobuf'})
        raise_on_error(response)
        return parse_response(RFMExplanationResponse, response)

    def evaluate(self, request: bytes) -> RFMEvaluateResponse:
        """Evaluate the RFM model on the given context.

        Args:
            request: The evaluate request as serialized protobuf.

        Returns:
            RFMEvaluateResponse containing the computed metrics
        """
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

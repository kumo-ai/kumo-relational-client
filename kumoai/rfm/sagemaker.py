import base64
import json
from typing import Any

import requests

from kumoai.client import KumoClient
from kumoai.client.client import _raise_init_error
from kumoai.client.endpoints import Endpoint, HTTPMethod
from kumoai.exceptions import HTTPException


class SageMakerResponseAdapter(requests.Response):
    def __init__(self, sm_response: Any) -> None:
        super().__init__()
        # Read the body bytes
        self._content = sm_response['Body'].read()
        self.status_code = 200
        self.headers['Content-Type'] = sm_response.get('ContentType',
                                                       'application/json')
        # Optionally, you can store original sm_response for debugging
        self.sm_response = sm_response

    @property
    def text(self) -> str:
        assert isinstance(self._content, bytes)
        return self._content.decode('utf-8')

    def json(self, **kwargs) -> dict[str, Any]:
        return json.loads(self.text, **kwargs)


class KumoClient_SageMakerAdapter(KumoClient):
    def __init__(self, region: str, endpoint_name: str):
        import boto3
        self._client = boto3.client(
            service_name="sagemaker-runtime",
            region_name=region,
        )
        self._endpoint_name = endpoint_name

        # Recording buffers.
        self._recording_active = False
        self._recorded_reqs: list[dict[str, Any]] = []
        self._recorded_resps: list[dict[str, Any]] = []

    def authenticate(self) -> None:
        # TODO(siyang): call /ping to verify?
        pass

    def _request(self, endpoint: Endpoint, **kwargs: Any) -> requests.Response:
        assert endpoint.method == HTTPMethod.POST
        if 'json' in kwargs:
            payload = json.dumps(kwargs.pop('json'))
        elif 'data' in kwargs:
            raw_payload = kwargs.pop('data')
            assert isinstance(raw_payload, bytes)
            payload = base64.b64encode(raw_payload).decode()
        else:
            raise HTTPException(400, 'Unable to send data to KumoRFM.')

        request = {
            'method': endpoint.get_path().rsplit('/')[-1],
            'payload': payload,
        }
        response = self._client.invoke_endpoint(
            EndpointName=self._endpoint_name,
            ContentType="application/json",
            Body=json.dumps(request),
        )

        adapted_response = SageMakerResponseAdapter(response)

        # If validation is active, store input/output
        if self._recording_active:
            self._recorded_reqs.append(request)
            self._recorded_resps.append(adapted_response.json())

        return adapted_response

    def start_recording(self) -> None:
        """Start recording requests/responses to/from sagemaker endpoint."""
        assert not self._recording_active
        self._recording_active = True
        self._recorded_reqs.clear()
        self._recorded_resps.clear()

    def end_recording(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Stop recording and return recorded requests/responses."""
        assert self._recording_active
        self._recording_active = False
        recorded = list(zip(self._recorded_reqs, self._recorded_resps))
        self._recorded_reqs.clear()
        self._recorded_resps.clear()
        return recorded


class KumoClient_SageMakerProxy_Local(KumoClient):
    def __init__(self, url: str):
        self._client = KumoClient(url, api_key=None)
        self._client._api_url = self._client._url
        self._endpoint = Endpoint('/invocations', HTTPMethod.POST)

    def authenticate(self) -> None:
        url = self._client._url
        try:
            self._client._session.get(
                url + '/ping',
                verify=self._client._verify_ssl).raise_for_status()
        except requests.RequestException as e:
            _raise_init_error(url, e)

    def _request(self, endpoint: Endpoint, **kwargs: Any) -> requests.Response:
        assert endpoint.method == HTTPMethod.POST
        if 'json' in kwargs:
            payload = json.dumps(kwargs.pop('json'))
        elif 'data' in kwargs:
            raw_payload = kwargs.pop('data')
            assert isinstance(raw_payload, bytes)
            payload = base64.b64encode(raw_payload).decode()
        else:
            raise HTTPException(400, 'Unable to send data to KumoRFM.')
        return self._client._request(
            self._endpoint,
            json={
                'method': endpoint.get_path().rsplit('/')[-1],
                'payload': payload,
            },
            **kwargs,
        )

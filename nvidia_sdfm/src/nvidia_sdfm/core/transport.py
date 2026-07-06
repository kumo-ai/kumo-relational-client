from __future__ import annotations

from typing import Any

import requests

from nvidia_sdfm.errors import NimRequestError, SdfmError

_PREDICTIONS_PATH = '/v1/predictions'
_HEALTH_READY_PATH = '/v1/health/ready'
_DEFAULT_TIMEOUT_SECONDS = 60


class TFMClient:
    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        *,
        verify_ssl: bool = True,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._url = url.rstrip('/')
        self._api_key = api_key
        self._verify_ssl = verify_ssl
        self._timeout = timeout
        self._session = requests.Session()
        if api_key:
            self._session.headers['X-API-Key'] = api_key

    @property
    def url(self) -> str:
        return self._url

    @property
    def api_key(self) -> str | None:
        return self._api_key

    def health_ready(self) -> bool:
        response = self._session.get(
            self._url + _HEALTH_READY_PATH,
            timeout=self._timeout,
            verify=self._verify_ssl,
        )
        return response.status_code == 200

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(_PREDICTIONS_PATH, payload)

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(
            'Session endpoints are not wired yet; see nvidia_sdfm/README.md',
        )

    def session_predict(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError(
            'Session endpoints are not wired yet; see nvidia_sdfm/README.md',
        )

    def delete_session(self, session_id: str) -> None:
        raise NotImplementedError(
            'Session endpoints are not wired yet; see nvidia_sdfm/README.md',
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._session.post(
                self._url + path,
                json=payload,
                timeout=self._timeout,
                verify=self._verify_ssl,
            )
        except requests.RequestException as error:
            raise SdfmError(
                f'Request to {self._url + path} failed: {error}',
                code='TRANSPORT_ERROR',
            ) from error
        if response.status_code >= 400:
            raise _to_nim_error(response)
        try:
            body = response.json()
        except ValueError as error:
            raise SdfmError(
                f'Invalid JSON response from {self._url + path}',
                code='TRANSPORT_ERROR',
            ) from error
        if not isinstance(body, dict):
            raise SdfmError(
                f'Expected a JSON object from {self._url + path}, '
                f'got {type(body).__name__}',
                code='TRANSPORT_ERROR',
            )
        return body


def _to_nim_error(response: requests.Response) -> NimRequestError:
    try:
        body = response.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    code = body.get('code')
    message = body.get('detail') or body.get('title') or response.text or (
        f'NIM request failed with status {response.status_code}'
    )
    details = {
        key: value
        for key, value in body.items()
        if key not in ('code', 'detail', 'title', 'type', 'status', 'instance')
    }
    return NimRequestError(
        response.status_code,
        code=code,
        message=message,
        details=details,
    )

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from kumorfm.rfm import explain_summary
from kumorfm.rfm.explain_summary import (
    SUMMARY_ERROR_MESSAGE,
    SUMMARY_UNAVAILABLE_MESSAGE,
    SYSTEM_PROMPT,
    generate_summary,
)

try:
    import openai  # noqa: F401
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

requires_openai = pytest.mark.skipif(
    not _HAS_OPENAI, reason='openai (kumorfm[explain]) is not installed')


class _FakeCompletions:
    def __init__(self, content: str | None = 'the summary',
                 record: dict[str, Any] | None = None,
                 error: Exception | None = None) -> None:
        self._content = content
        self._record = record
        self._error = error

    def create(self, *, model: str, messages: list[dict[str, str]]) -> Any:
        if self._record is not None:
            self._record.update(model=model, messages=messages)
        if self._error is not None:
            raise self._error
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeClient:
    def __init__(self, **kwargs: Any) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(**kwargs))


def _system(record: dict[str, Any]) -> str:
    return record['messages'][0]['content']


def _user(record: dict[str, Any]) -> str:
    return record['messages'][1]['content']


def test_system_prompt_includes_both_sources() -> None:
    assert 'Explanation summary module' in SYSTEM_PROMPT
    assert 'Understanding the Global View: Cohorts' in SYSTEM_PROMPT
    assert 'Understanding the Local View: Subgraph' in SYSTEM_PROMPT


def test_generate_summary_passes_context_and_returns_text() -> None:
    record: dict[str, Any] = {}
    client = _FakeClient(content='Order frequency fell.', record=record)
    out = generate_summary(
        query='PREDICT ... FOR users.user_id=1',
        prediction=[{'ENTITY': 1, 'prediction': 0.5}],
        cohorts=[{'column_name': 'COUNT(*)'}],
        subgraphs=['SG0', 'SG1'],
        client=client,
        model='test-model',
    )
    assert out == 'Order frequency fell.'
    assert record['model'] == 'test-model'
    assert _system(record) == SYSTEM_PROMPT
    assert 'USER QUERY: PREDICT ... FOR users.user_id=1' in _user(record)
    assert "MODEL PREDICTION: [{'ENTITY': 1, 'prediction': 0.5}]" in _user(record)
    assert "COLUMN ANALYSIS: [{'column_name': 'COUNT(*)'}]" in _user(record)
    assert 'SUBGRAPH EXPLANATION: SG0' in _user(record)
    assert 'SG1' not in _user(record)


def test_generate_summary_unavailable_without_key(monkeypatch: Any) -> None:
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    out = generate_summary('q', 'p', ['c'], ['s'])
    assert out == SUMMARY_UNAVAILABLE_MESSAGE


def test_generate_summary_handles_api_error() -> None:
    client = _FakeClient(error=RuntimeError('boom'))
    out = generate_summary('q', 'p', ['c'], ['s'], client=client)
    assert out == SUMMARY_ERROR_MESSAGE


def test_generate_summary_empty_content_is_error() -> None:
    client = _FakeClient(content=None)
    out = generate_summary('q', 'p', ['c'], ['s'], client=client)
    assert out == SUMMARY_ERROR_MESSAGE


def test_generate_summary_uses_env_model(monkeypatch: Any) -> None:
    monkeypatch.setenv('KUMORFM_EXPLAIN_MODEL', 'env-model')
    record: dict[str, Any] = {}
    client = _FakeClient(record=record)
    generate_summary('q', 'p', ['c'], ['s'], client=client)
    assert record['model'] == 'env-model'


def test_generate_summary_explicit_model_overrides_env(monkeypatch: Any) -> None:
    monkeypatch.setenv('KUMORFM_EXPLAIN_MODEL', 'env-model')
    record: dict[str, Any] = {}
    client = _FakeClient(record=record)
    generate_summary('q', 'p', ['c'], ['s'], client=client, model='explicit')
    assert record['model'] == 'explicit'


def test_generate_summary_empty_subgraphs() -> None:
    record: dict[str, Any] = {}
    client = _FakeClient(record=record)
    generate_summary('q', 'p', ['c'], [], client=client)
    assert 'SUBGRAPH EXPLANATION: ' in _user(record)


def test_make_client_none_without_key() -> None:
    assert explain_summary._make_client(None, None, 5.0) is None
    assert explain_summary._make_client('https://ep/v1', None, 5.0) is None


@requires_openai
def test_make_client_targets_custom_endpoint() -> None:
    client = explain_summary._make_client('https://ep.example/v1', 'k', 5.0)
    assert client is not None
    assert str(client.base_url).rstrip('/') == 'https://ep.example/v1'


def test_make_client_uses_env_endpoint(monkeypatch: Any) -> None:
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://env.example/v1')
    monkeypatch.setenv('OPENAI_API_KEY', 'envkey')
    record: dict[str, Any] = {}

    def fake_make_client(base_url: str | None, api_key: str | None,
                         timeout: float) -> Any:
        record.update(base_url=base_url, api_key=api_key)
        return _FakeClient()

    monkeypatch.setattr(explain_summary, '_make_client', fake_make_client)
    generate_summary('q', 'p', ['c'], ['s'])
    assert record['base_url'] == 'https://env.example/v1'
    assert record['api_key'] == 'envkey'


def test_generate_summary_explicit_endpoint_overrides_env(monkeypatch: Any) -> None:
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://env.example/v1')
    monkeypatch.setenv('OPENAI_API_KEY', 'envkey')
    record: dict[str, Any] = {}

    def fake_make_client(base_url: str | None, api_key: str | None,
                         timeout: float) -> Any:
        record.update(base_url=base_url, api_key=api_key)
        return _FakeClient()

    monkeypatch.setattr(explain_summary, '_make_client', fake_make_client)
    generate_summary('q', 'p', ['c'], ['s'],
                     base_url='https://explicit/v1', api_key='ekey')
    assert record['base_url'] == 'https://explicit/v1'
    assert record['api_key'] == 'ekey'


def test_explanation_accessors_ignore_malformed_details() -> None:
    import pandas as pd

    from kumorfm.rfm.rfm import Explanation

    empty = pd.DataFrame()
    for details in (
        None,
        {'format': 'natural_language_summary', 'summary': 'x'},
        {'format': 'kumo_rfm_v2_1', 'details': 'oops'},
        {'format': 'kumo_rfm_v2_1',
         'details': {'cohorts': 'x', 'subgraphs': {'a': 1}}},
    ):
        e = Explanation(prediction=empty, summary='', details=details)
        assert e.cohorts == []
        assert e.subgraphs == []

    good = Explanation(
        prediction=empty, summary='',
        details={'format': 'kumo_rfm_v2_1',
                 'details': {'cohorts': [{'c': 1}], 'subgraphs': [{'s': 2}]}})
    assert good.cohorts == [{'c': 1}]
    assert good.subgraphs == [{'s': 2}]


def test_nim_failure_error_explain_reports_gpu_pressure() -> None:
    from kumorfm.exceptions import HTTPException
    from kumorfm.rfm.rfm import _nim_failure_error

    msg = str(_nim_failure_error(HTTPException(503, 'Service Unavailable'),
                                 explain=True))
    assert 'temporarily unavailable' in msg
    assert 'GPU memory pressure' in msg
    assert 'this explanation' in msg
    assert 'one at a time' in msg


def test_nim_failure_error_prediction_has_no_pacing_hint() -> None:
    from kumorfm.exceptions import HTTPException
    from kumorfm.rfm.rfm import _nim_failure_error

    msg = str(_nim_failure_error(HTTPException(500, 'boom'), explain=False))
    assert 'this prediction' in msg
    assert 'GPU memory pressure' in msg
    assert 'one at a time' not in msg


def test_nim_failure_error_connection_drop_is_unavailable() -> None:
    import requests

    from kumorfm.rfm.rfm import _nim_failure_error

    msg = str(_nim_failure_error(
        requests.exceptions.ConnectionError('connection reset'), explain=True))
    assert 'temporarily unavailable' in msg


def test_nim_failure_error_client_error_stays_generic() -> None:
    from kumorfm.exceptions import HTTPException
    from kumorfm.rfm.rfm import _nim_failure_error

    msg = str(_nim_failure_error(
        HTTPException(400, '{"detail": "bad predictive query"}'), explain=True))
    assert 'bad predictive query' in msg
    assert 'create an issue' in msg
    assert 'GPU memory pressure' not in msg

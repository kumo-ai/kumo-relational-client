from __future__ import annotations

import sys

import pandas as pd
import pytest

from nvidia_sdfm.adapters.rfm import RFMAdapter
from nvidia_sdfm.core.transport import TFMClient
from nvidia_sdfm.errors import MissingExtraError

try:
    import kumoai.rfm as kumoai_rfm
except (ImportError, RuntimeError) as error:
    kumoai_rfm = None
    _KUMOAI_UNUSABLE = str(error)
else:
    _KUMOAI_UNUSABLE = ''

requires_kumoai = pytest.mark.skipif(
    kumoai_rfm is None,
    reason=f'kumoai.rfm is not usable in this environment: {_KUMOAI_UNUSABLE}',
)


@pytest.fixture
def client() -> TFMClient:
    return TFMClient('http://nim.example.com:8000', api_key='secret')


@requires_kumoai
def test_predict_forwards_url_and_api_key_to_kumoai_init(monkeypatch, client):
    captured = {}
    monkeypatch.setattr(
        kumoai_rfm, 'init',
        lambda **kwargs: captured.update(init=kwargs),
    )

    class FakeKumoRFM:
        def __init__(self, graph):
            captured['graph'] = graph

        def predict(self, query, **kwargs):
            captured['predict'] = {'query': query, **kwargs}
            return pd.DataFrame({'entity': kwargs.get('indices')})

    monkeypatch.setattr(kumoai_rfm, 'KumoRFM', FakeKumoRFM)

    result = RFMAdapter().predict(
        client, graph='fake-graph', query='PREDICT target FOR entity=1',
        indices=[1, 2, 3],
    )

    assert captured['init'] == {'url': client.url, 'api_key': client.api_key}
    assert captured['graph'] == 'fake-graph'
    assert captured['predict']['query'] == 'PREDICT target FOR entity=1'
    assert captured['predict']['run_mode'] == 'fast'
    assert list(result['entity']) == [1, 2, 3]


@requires_kumoai
def test_predict_forwards_custom_run_mode_and_kwargs(monkeypatch, client):
    captured = {}

    class FakeKumoRFM:
        def __init__(self, graph):
            pass

        def predict(self, query, **kwargs):
            captured.update(kwargs)
            return pd.DataFrame({'entity': [1]})

    monkeypatch.setattr(kumoai_rfm, 'init', lambda **kwargs: None)
    monkeypatch.setattr(kumoai_rfm, 'KumoRFM', FakeKumoRFM)

    RFMAdapter().predict(
        client, graph='g', query='PREDICT x', indices=[1],
        run_mode='best', num_hops=3,
    )

    assert captured['run_mode'] == 'best'
    assert captured['num_hops'] == 3


@requires_kumoai
def test_predict_raises_type_error_on_non_dataframe_result(monkeypatch, client):
    class FakeKumoRFM:
        def __init__(self, graph):
            pass

        def predict(self, query, **kwargs):
            return object()

    monkeypatch.setattr(kumoai_rfm, 'init', lambda **kwargs: None)
    monkeypatch.setattr(kumoai_rfm, 'KumoRFM', FakeKumoRFM)

    with pytest.raises(TypeError):
        RFMAdapter().predict(client, graph='g', query='PREDICT x')


def test_predict_raises_missing_extra_error_when_kumoai_not_installed(
    monkeypatch, client,
):
    monkeypatch.setitem(sys.modules, 'kumoai.rfm', None)
    with pytest.raises(MissingExtraError):
        RFMAdapter().predict(client, graph='g', query='PREDICT x')

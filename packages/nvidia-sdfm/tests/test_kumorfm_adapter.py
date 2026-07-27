from __future__ import annotations

import pandas as pd
import pytest

from nvidia_sdfm.adapters.kumorfm import KumoRFMAdapter
from nvidia_sdfm.core.transport import Transport
from nvidia_sdfm.errors import MissingExtraError, SdfmError
from nvidia_sdfm.requests import KumoRFMRequest

try:
    import kumorfm.rfm as rfm_engine
except (ImportError, RuntimeError) as error:
    rfm_engine = None
    _ENGINE_UNUSABLE = str(error)
else:
    _ENGINE_UNUSABLE = ''

requires_engine = pytest.mark.skipif(
    rfm_engine is None,
    reason=f'kumorfm.rfm is not usable in this environment: {_ENGINE_UNUSABLE}',
)


@pytest.fixture
def client() -> Transport:
    return Transport('https://nim.example.com:8000', api_key='secret')


@requires_engine
def test_predict_forwards_url_and_api_key_to_engine_init(monkeypatch, client):
    captured = {}
    monkeypatch.setattr(
        rfm_engine, 'init',
        lambda **kwargs: captured.update(init=kwargs),
    )

    class FakeKumoRFM:
        def __init__(self, graph):
            captured['graph'] = graph

        def predict(self, query, **kwargs):
            captured['predict'] = {'query': query, **kwargs}
            return pd.DataFrame({'entity': kwargs.get('indices')})

    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    result = KumoRFMAdapter().predict(client, KumoRFMRequest(
        graph='fake-graph', query='PREDICT target FOR entity=1',
        indices=[1, 2, 3]))

    assert captured['init'] == {
        'url': client.url,
        'api_key': client.api_key,
        'verify_ssl': client.verify_ssl,
        '_token': rfm_engine._SDFM_CLIENT_TOKEN,
    }
    assert captured['graph'] == 'fake-graph'
    assert captured['predict']['query'] == 'PREDICT target FOR entity=1'
    assert captured['predict']['run_mode'] == 'fast'
    assert list(result['entity']) == [1, 2, 3]


@requires_engine
def test_predict_forwards_custom_run_mode_and_options(monkeypatch, client):
    captured = {}

    class FakeKumoRFM:
        def __init__(self, graph):
            pass

        def predict(self, query, **kwargs):
            captured.update(kwargs)
            return pd.DataFrame({'entity': [1]})

    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)
    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    KumoRFMAdapter().predict(client, KumoRFMRequest(
        graph='g', query='PREDICT x', indices=[1], run_mode='best',
        options={'num_hops': 3}))

    assert captured['run_mode'] == 'best'
    assert captured['num_hops'] == 3


@requires_engine
def test_predict_rejects_reserved_option_keys(monkeypatch, client):
    called = {}
    monkeypatch.setattr(rfm_engine, 'init',
                        lambda **kwargs: called.setdefault('init', True))
    with pytest.raises(SdfmError) as excinfo:
        KumoRFMAdapter().predict(client, KumoRFMRequest(
            graph='g', query='PREDICT x', options={'run_mode': 'best'}))
    assert excinfo.value.code == 'INVALID_REQUEST'
    assert 'init' not in called


@requires_engine
def test_predict_raises_type_error_on_non_dataframe_result(monkeypatch, client):
    class FakeKumoRFM:
        def __init__(self, graph):
            pass

        def predict(self, query, **kwargs):
            return object()

    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)
    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    with pytest.raises(TypeError):
        KumoRFMAdapter().predict(
            client, KumoRFMRequest(graph='g', query='PREDICT x'))


@requires_engine
def test_adapter_authorizes_engine_init(monkeypatch, client):
    captured = {}
    monkeypatch.setattr(rfm_engine, 'init',
                        lambda **kwargs: captured.update(kwargs))

    class FakeKumoRFM:
        def __init__(self, graph):
            pass

        def predict(self, query, **kwargs):
            return pd.DataFrame({'entity': [1]})

    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)
    KumoRFMAdapter().predict(
        client, KumoRFMRequest(graph='g', query='PREDICT x'))

    assert captured['_token'] is rfm_engine._SDFM_CLIENT_TOKEN


def test_kumorfm_shim_does_not_expose_driver():
    from nvidia_sdfm import kumorfm as kumorfm_shim

    assert 'KumoRFM' not in kumorfm_shim.__all__
    assert 'Graph' in kumorfm_shim.__all__
    with pytest.raises(AttributeError):
        kumorfm_shim.KumoRFM


def _patch_engine_import(monkeypatch, error: BaseException) -> None:
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'kumorfm.rfm':
            raise error
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)


def test_predict_raises_missing_extra_error_when_engine_not_installed(
    monkeypatch, client,
):
    _patch_engine_import(
        monkeypatch,
        ModuleNotFoundError("No module named 'kumorfm'", name='kumorfm'),
    )
    with pytest.raises(MissingExtraError):
        KumoRFMAdapter().predict(
            client, KumoRFMRequest(graph='g', query='PREDICT x'))


def test_predict_propagates_broken_driver_error(monkeypatch, client):
    _patch_engine_import(
        monkeypatch, RuntimeError('kumolib native extension failed to load'))
    with pytest.raises(RuntimeError, match='native extension'):
        KumoRFMAdapter().predict(
            client, KumoRFMRequest(graph='g', query='PREDICT x'))


def test_predict_propagates_transitive_import_error(monkeypatch, client):
    _patch_engine_import(
        monkeypatch,
        ModuleNotFoundError("No module named 'some_dep'", name='some_dep'),
    )
    with pytest.raises(ModuleNotFoundError, match='some_dep'):
        KumoRFMAdapter().predict(
            client, KumoRFMRequest(graph='g', query='PREDICT x'))


@requires_engine
def test_predict_explain_field_returns_explanation(monkeypatch, client):
    from kumorfm.rfm.rfm import Explanation

    captured = {}
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    class FakeKumoRFM:
        def __init__(self, graph):
            pass

        def predict(self, query, **kwargs):
            captured.update(kwargs)
            if kwargs.get('explain'):
                return Explanation(
                    prediction=pd.DataFrame({'ENTITY': [1]}),
                    summary='Price drove the prediction.',
                    details={'format': 'kumo_rfm_v2_1'},
                )
            return pd.DataFrame({'ENTITY': [1]})

    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    result = KumoRFMAdapter().predict(client, KumoRFMRequest(
        graph='fake-graph', query='PREDICT t FOR e=1', explain=True))

    assert isinstance(result, Explanation)
    assert result.summary == 'Price drove the prediction.'
    assert captured['explain'] is True


@requires_engine
def test_predict_explain_via_options_returns_explanation(monkeypatch, client):
    from kumorfm.rfm.rfm import Explanation

    captured = {}
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    class FakeKumoRFM:
        def __init__(self, graph):
            pass

        def predict(self, query, **kwargs):
            captured.update(kwargs)
            return Explanation(
                prediction=pd.DataFrame({'ENTITY': [1]}),
                summary='',
                details={'skip_summary': True},
            )

    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    result = KumoRFMAdapter().predict(client, KumoRFMRequest(
        graph='fake-graph', query='PREDICT t FOR e=1',
        options={'explain': {'skip_summary': True}}))

    assert isinstance(result, Explanation)
    assert captured['explain'] == {'skip_summary': True}


@requires_engine
def test_predict_without_explain_still_requires_dataframe(monkeypatch, client):
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    class FakeKumoRFM:
        def __init__(self, graph):
            pass

        def predict(self, query, **kwargs):
            return object()

    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    with pytest.raises(TypeError, match='expected a DataFrame result'):
        KumoRFMAdapter().predict(client, KumoRFMRequest(
            graph='fake-graph', query='PREDICT t FOR e=1'))


@requires_engine
def test_predict_explain_rejects_non_explanation_result(monkeypatch, client):
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    class FakeKumoRFM:
        def __init__(self, graph):
            pass

        def predict(self, query, **kwargs):
            return pd.DataFrame({'ENTITY': [1]})

    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    with pytest.raises(TypeError, match='expected an Explanation result'):
        KumoRFMAdapter().predict(client, KumoRFMRequest(
            graph='fake-graph', query='PREDICT t FOR e=1', explain=True))


def test_capabilities_advertise_explanation():
    assert 'explanation' in KumoRFMAdapter().capabilities().outputs


@requires_engine
@pytest.mark.parametrize('field_value', [True, {}, {'skip_summary': True}])
def test_predict_rejects_explain_specified_twice(monkeypatch, client, field_value):
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)
    monkeypatch.setattr(rfm_engine, 'KumoRFM', lambda graph: None)

    with pytest.raises(SdfmError) as err:
        KumoRFMAdapter().predict(client, KumoRFMRequest(
            graph='fake-graph', query='PREDICT t FOR e=1',
            explain=field_value, options={'explain': False}))
    assert err.value.code == 'INVALID_REQUEST'


@requires_engine
@pytest.mark.parametrize('bad', [0, '', 'yes', 1.0])
def test_predict_rejects_malformed_explain_values(monkeypatch, client, bad):
    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)
    monkeypatch.setattr(rfm_engine, 'KumoRFM', lambda graph: None)

    with pytest.raises(SdfmError) as err:
        KumoRFMAdapter().predict(client, KumoRFMRequest(
            graph='fake-graph', query='PREDICT t FOR e=1',
            options={'explain': bad}))
    assert err.value.code == 'INVALID_REQUEST'


@requires_engine
def test_sdfm_client_predict_explain_returns_explanation(monkeypatch):
    """Exercise issue #19: client.kumorfm(graph).predict(explain=True)."""
    from kumorfm.rfm.rfm import Explanation

    from nvidia_sdfm import SDFMClient

    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)

    class FakeKumoRFM:
        def __init__(self, graph):
            pass

        def predict(self, query, **kwargs):
            assert kwargs['explain'] is True
            return Explanation(
                prediction=pd.DataFrame({'ENTITY': [1]}),
                summary='',
                details={'format': 'kumo_rfm_v2_1', 'details': {'cohorts': []}},
            )

    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    monkeypatch.setattr(
        'nvidia_sdfm.core.transport.Transport.health_ready', lambda self: True)
    with SDFMClient(url='http://127.0.0.1:18001') as sdfm:
        result = sdfm.kumorfm('fake-graph').predict(
            'PREDICT t FOR e=1', explain=True)

    assert isinstance(result, Explanation)
    assert result.details['format'] == 'kumo_rfm_v2_1'


@requires_engine
def test_predict_accepts_explain_config_object(monkeypatch, client):
    """ExplainConfig instances are a valid driver explain input, not rejected
    as INVALID_REQUEST (the driver's predict accepts bool|ExplainConfig|dict)."""
    from kumorfm.rfm.rfm import ExplainConfig, Explanation

    monkeypatch.setattr(rfm_engine, 'init', lambda **kwargs: None)
    captured = {}

    class FakeKumoRFM:
        def __init__(self, graph):
            pass

        def predict(self, query, **kwargs):
            captured['explain'] = kwargs['explain']
            return Explanation(
                prediction=pd.DataFrame({'ENTITY': [1]}),
                summary='', details={}, warning=None)

    monkeypatch.setattr(rfm_engine, 'KumoRFM', FakeKumoRFM)

    cfg = ExplainConfig(skip_summary=True)
    result = KumoRFMAdapter().predict(client, KumoRFMRequest(
        graph='fake-graph', query='PREDICT t FOR e=1', explain=cfg))

    assert isinstance(result, Explanation)
    assert captured['explain'] is cfg

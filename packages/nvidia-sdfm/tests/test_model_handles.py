# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
import pytest

from nvidia_sdfm import (
    RFMModel,
    SDFMClient,
    TabICLModel,
)
from nvidia_sdfm.base import ModelAdapter, ModelCapabilities
from nvidia_sdfm.requests import KumoRFMRequest, TabICLRequest


class _CapturingAdapter(ModelAdapter):
    """Records the request it receives and returns a fixed DataFrame, so a
    handle can be driven end-to-end through the real SDFMClient
    dispatch without a live NIM."""

    def __init__(self, name, request_type, result):
        self.name = name
        self.request_type = request_type
        self.result = result
        self.captured = None

    def capabilities(self):
        return ModelCapabilities(
            model=self.name, request_type=self.request_type.__name__)

    def predict(self, transport, request):
        self.captured = request
        return self.result


def _client_with(adapter) -> SDFMClient:
    client = SDFMClient(url='http://nim.test')
    client._register(adapter)
    return client


def test_rfm_returns_bound_handle():
    client = SDFMClient(url='http://nim.test')
    handle = client.kumorfm('my-graph')
    assert isinstance(handle, RFMModel)


def test_rfm_handle_end_to_end_through_client():
    result = pd.DataFrame({'ENTITY': [1], 'PREDICTION': [0.5]})
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMRequest, result)
    client = _client_with(adapter)

    out = client.kumorfm('my-graph').predict(
        'PREDICT x FOR EACH t.id', [1, 2, 3],
        run_mode='best', anchor_time='2025-01-01')

    assert out is result
    req = adapter.captured
    assert isinstance(req, KumoRFMRequest)
    assert req.model == 'kumo-rfm'
    assert req.graph == 'my-graph'
    assert req.query == 'PREDICT x FOR EACH t.id'
    assert req.indices == [1, 2, 3]
    assert req.run_mode == 'best'
    assert req.options == {'anchor_time': '2025-01-01'}


def test_rfm_handle_defaults_are_minimal():
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMRequest, pd.DataFrame())
    client = _client_with(adapter)

    client.kumorfm('g').predict('PREDICT x')

    req = adapter.captured
    assert req.indices is None
    assert req.run_mode == 'fast'
    assert req.options == {}


def test_rfm_handle_only_forwards_set_options():
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMRequest, pd.DataFrame())
    client = _client_with(adapter)

    client.kumorfm('g').predict(
        'PREDICT x',
        inference_config={'num_estimators': 2},
        return_embeddings=True)

    assert adapter.captured.options == {
        'inference_config': {'num_estimators': 2},
        'return_embeddings': True,
    }


def test_rfm_handle_matches_typed_request():
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMRequest, pd.DataFrame())
    client = _client_with(adapter)

    client.kumorfm('g').predict('PREDICT x', [1], run_mode='normal',
                            anchor_time='2025-06-01')
    via_handle = adapter.captured

    typed = KumoRFMRequest(graph='g', query='PREDICT x', indices=[1],
                           run_mode='normal',
                           options={'anchor_time': '2025-06-01'})

    assert via_handle == typed


def test_tabicl_returns_bound_handle():
    client = SDFMClient(url='http://nim.test')
    handle = client.tabicl(pd.DataFrame({'y': [0, 1]}),
                           target='y', task='classification')
    assert isinstance(handle, TabICLModel)


def test_tabicl_handle_end_to_end_through_client():
    result = pd.DataFrame({'prediction': [1]})
    adapter = _CapturingAdapter('tabicl', TabICLRequest, result)
    client = _client_with(adapter)

    ctx = pd.DataFrame({'x': [1, 2], 'y': [0, 1]})
    rows = pd.DataFrame({'x': [3]})
    out = client.tabicl(ctx, target='y', task='classification').predict(
        rows, positive_class='1')

    assert out is result
    req = adapter.captured
    assert isinstance(req, TabICLRequest)
    assert req.model == 'tabicl'
    assert req.context.equals(ctx)
    assert req.predict.equals(rows)
    assert req.task == 'classification'
    assert req.target == 'y'
    assert req.positive_class == '1'


def test_tabicl_handle_defaults_are_minimal():
    adapter = _CapturingAdapter('tabicl', TabICLRequest, pd.DataFrame())
    client = _client_with(adapter)

    ctx = pd.DataFrame({'x': [1], 'y': [0]})
    client.tabicl(ctx, target='y', task='regression').predict(
        pd.DataFrame({'x': [2]}))

    req = adapter.captured
    assert req.outputs == ['prediction']
    assert req.positive_class is None
    assert req.max_results is None


def test_handle_still_dispatches_by_request_type():
    adapter = _CapturingAdapter('kumo-rfm', TabICLRequest, pd.DataFrame())
    client = _client_with(adapter)

    from nvidia_sdfm import SdfmError
    with pytest.raises(SdfmError):
        client.kumorfm('g').predict('PREDICT x')


def test_public_predict_is_not_exposed():
    client = SDFMClient(url='http://nim.test')
    assert not hasattr(client, 'predict')
    assert hasattr(client, 'kumorfm')
    assert hasattr(client, 'tabicl')


def test_rfm_handle_forwards_explain():
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMRequest, pd.DataFrame())
    client = _client_with(adapter)

    client.kumorfm('g').predict('PREDICT x FOR t.id=1', explain=True)
    assert adapter.captured.explain is True

    client.kumorfm('g').predict('PREDICT x FOR t.id=1')
    assert adapter.captured.explain is False


def test_rfm_handle_forwards_num_neighbors_and_num_hops():
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMRequest, pd.DataFrame())
    client = _client_with(adapter)

    client.kumorfm('g').predict('PREDICT x', num_neighbors=[8, 8], num_hops=3)

    assert adapter.captured.options == {
        'num_neighbors': [8, 8],
        'num_hops': 3,
    }


def test_rfm_handle_forwards_explain_config():
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMRequest, pd.DataFrame())
    client = _client_with(adapter)

    cfg = {'skip_summary': True}
    client.kumorfm('g').predict('PREDICT x FOR t.id=1', explain=cfg)
    assert adapter.captured.explain == cfg


def test_tabicl_handle_forwards_request_id():
    adapter = _CapturingAdapter('tabicl', TabICLRequest, pd.DataFrame())
    client = _client_with(adapter)

    ctx = pd.DataFrame({'x': [1], 'y': [0]})
    client.tabicl(ctx, target='y', task='classification').predict(
        pd.DataFrame({'x': [2]}), request_id='trace-123')

    assert adapter.captured.request_id == 'trace-123'


def test_rfm_handle_forwards_batch_size():
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMRequest, pd.DataFrame())
    client = _client_with(adapter)

    client.kumorfm('g').predict('PREDICT x FOR EACH t.id',
                                list(range(1500)),
                                batch_size='max', num_retries=2)

    req = adapter.captured
    assert req.batch_size == 'max'
    assert req.num_retries == 2


def test_rfm_handle_batch_defaults_off():
    adapter = _CapturingAdapter('kumo-rfm', KumoRFMRequest, pd.DataFrame())
    client = _client_with(adapter)

    client.kumorfm('g').predict('PREDICT x')

    assert adapter.captured.batch_size is None
    assert adapter.captured.num_retries == 1

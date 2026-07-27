# nvidia-sdfm

Client SDK for NVIDIA structured-data foundation model NIMs served behind the
Universal TFM API. A thin, model-agnostic client dispatches typed requests to
per-model adapters; heavy model drivers are optional extras.

## Install

```bash
pip install nvidia-sdfm              # client + every lightweight model (TabICL)
pip install nvidia-sdfm[kumorfm]     # adds KumoRFM (native driver)
pip install nvidia-sdfm[sqlite]      # data-source reads ([duckdb]/[snowflake]/[databricks])
pip install nvidia-sdfm[all]        # everything
```

## Quickstart

A `SDFMClient` owns one connection to a NIM. Requests are typed per model and
validated against the model's capabilities before sending.

TabICL (single table):

```python
from nvidia_sdfm import SDFMClient

with SDFMClient(url="http://localhost:8000") as client:
    model = client.tabicl(context_df, target="label", task="classification")
    df = model.predict(predict_df, outputs=["prediction", "probabilities"])
```

KumoRFM (relational) — needs `nvidia-sdfm[kumorfm]`:

```python
from nvidia_sdfm import SDFMClient, kumorfm

graph = kumorfm.Graph.from_data({"users": df1, "items": df2, "orders": df3})

with SDFMClient(url="http://localhost:8000") as client:
    df = client.kumorfm(graph).predict(
        "PREDICT SUM(orders.price, 0, 30, days) FOR items.item_id=1",
        indices=[...],
        run_mode="fast",
    )
```

Set `explain=True` to get a `kumorfm` `Explanation` instead of a bare
DataFrame. The predicted rows stay on `result.prediction`; `result.details`
carries the driver's structured attribution (feature cohorts and subgraphs).
`result.summary` is a natural-language string only when the backend provides
one. If the backend returns structured attribution only, `result.summary` is
empty.

```python
with SDFMClient(url="http://localhost:8000") as client:
    result = client.kumorfm(graph).predict(
        "PREDICT SUM(orders.price, 0, 30, days) FOR items.item_id=1",
        explain=True,
    )
    predictions = result.prediction
    attribution = result.details
```

Discover what a NIM serves with `client.models()` and
`client.capabilities("tabicl")`. The transport pools connections and retries
transient failures (429/5xx) with backoff; tune it per client with
`SDFMClient(url, timeout=30, max_retries=3)`. Each client holds its own
transport and registry, so multiple clients can target different endpoints at
once.

`from nvidia_sdfm import kumorfm` is the supported surface onto the KumoRFM
driver (`Graph`, `Table`, `KumoRFM`, ...); you never import the driver package
directly.

## Adding a model

Implement `ModelAdapter` in `nvidia_sdfm/adapters/<model>.py` with a typed
request, and register it in the package `__init__`. Models needing a heavy
runtime ship it as a separate driver distribution behind an extra; the adapter
lazy-imports the driver so base installs stay light.

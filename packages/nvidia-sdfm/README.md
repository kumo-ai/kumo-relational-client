# nvidia-sdfm

Client SDK for NVIDIA structured-data foundation model NIMs served behind the
Universal TFM API. A thin, model-agnostic client dispatches typed requests to
per-model adapters; heavy model drivers are optional extras.

## Install

```bash
pip install nvidia-sdfm              # client + every lightweight model (TabICL)
pip install nvidia-sdfm[kumorfm]     # adds KumoRFM (native driver)
pip install nvidia-sdfm[sqlite]      # data-source reads ([duckdb]/[snowflake]/[databricks]/[s3])
pip install nvidia-sdfm[all]         # KumoRFM, every data-source backend,
                                     # and [databricks-serving]
pip install nvidia-sdfm[explain]     # KumoRFM plus the explanation-summary LLM
                                     # client. Deliberately NOT part of [all],
                                     # because it enables the third-party data
                                     # egress described below.
```

## Quickstart

A `SDFMClient` owns one connection to a NIM. Requests are typed per model: each
model handle builds its own request type, and the client rejects a request the
target model's adapter does not accept. Each adapter also checks what it knows
it cannot serve — an unsupported task kind or output field for TabICL, an
unknown `task_type` or a missing entity table for KumoRFM — and raises
`SdfmError(code="INVALID_REQUEST")` before anything is sent. Everything else is
validated by the NIM.

TabICL (single table):

```python
from nvidia_sdfm import SDFMClient

with SDFMClient(url='http://localhost:8000') as client:
    model = client.tabicl(context_df, target='label', task='classification')
    df = model.predict(predict_df, outputs=['prediction', 'probabilities'])
```

KumoRFM (relational) — needs `nvidia-sdfm[kumorfm]`:

```python
from nvidia_sdfm import SDFMClient, kumorfm

graph = kumorfm.Graph.from_data({'users': df1, 'items': df2, 'orders': df3})

with SDFMClient(url='http://localhost:8000') as client:
    df = client.kumorfm(graph).predict(
        'PREDICT SUM(orders.price, 0, 30, days) FOR items.item_id=1',
        indices=[...],
        run_mode='fast',
    )
```

Set `explain=True` to get a `kumorfm` `Explanation` instead of a bare
DataFrame. The predicted rows stay on `result.prediction`; `result.details`
carries the driver's structured attribution (feature cohorts and subgraphs).
`result.summary` is a natural-language string, either from the backend or, when
the backend returns structured attribution only, generated locally by the SDK
(see the warning below).

```python
with SDFMClient(url='http://localhost:8000') as client:
    result = client.kumorfm(graph).predict(
        'PREDICT SUM(orders.price, 0, 30, days) FOR items.item_id=1',
        explain=True,
    )
    predictions = result.prediction
    attribution = result.details
```

> **Data egress warning — the explanation summary calls a third-party LLM.**
> When the backend returns structured attribution without a summary (which is
> what the NIM does today), the SDK generates `result.summary` itself by POSTing
> the predictive query, the returned predictions, the cohort analysis and the
> subgraph attribution — **including the raw cell values of the explained
> entity's subgraph** — to an OpenAI-compatible chat-completions endpoint.
> Unless `KUMORFM_EXPLAIN_LLM_BASE_URL` points elsewhere, that endpoint is
> OpenAI's `https://api.openai.com/v1/` (model `gpt-4.1-mini-2025-04-14`, or
> `KUMORFM_EXPLAIN_LLM_MODEL`), a non-NVIDIA service. The API key is read from
> `KUMORFM_EXPLAIN_LLM_API_KEY` and **falls back to the ambient
> `OPENAI_API_KEY`**, so a key exported for an unrelated tool is enough to
> enable this.
>
> Nothing is sent if no key is discoverable or the `nvidia-sdfm[explain]` extra
> is not installed — which is why that extra is not part of `[all]`.
> `result.summary` then carries a message saying so, and the structured
> attribution on `result.details` is unaffected. To keep the data on the machine
> while still getting that attribution, disable the summary explicitly:
>
> ```python
> result = client.kumorfm(graph).predict(query, explain=dict(skip_summary=True))
> ```

Bring your own train table with `predict_task` when you want to supply the
in-context (train) labels directly instead of deriving them from a PQL query.
`context` holds the labelled rows and `predict` the rows to score, both
referencing entities of `entity_table` in the graph; columns default to
`ENTITY` / `TARGET` / `ANCHOR_TIMESTAMP`:

```python
with SDFMClient(url='http://localhost:8000') as client:
    df = client.kumorfm(graph).predict_task(
        context=train_df,  # ENTITY, TARGET, [ANCHOR_TIMESTAMP]
        predict=predict_df,  # ENTITY, [ANCHOR_TIMESTAMP]
        task_type='multiclass_classification',
        entity_table='users',
    )
```

`client.models()` lists the models this client has adapters for and
`client.capabilities("kumo-rfm")` describes one of them. Both read the
client-side registry, not the endpoint: a NIM serving only one of these models
still reports both, and a mismatch surfaces as an error from the NIM on the
first prediction. The transport pools connections and retries
transient failures (429/5xx) with backoff; tune it per client with
`SDFMClient(url, timeout=30, max_retries=3)`. `timeout` bounds each attempt
rather than the call as a whole, so a retried call can take up to
`(max_retries + 1) * timeout` plus backoff. Each client holds its own transport
and registry, so multiple clients can target different endpoints at once,
including concurrently: a prediction always goes to the endpoint and credential
of the client that started it. The KumoRFM driver underneath still keeps a
process-wide configuration that each prediction reconfigures, so drive it
through `SDFMClient` rather than mixing in direct `kumorfm.init()` calls.
`close()` (or leaving the `with` block) releases the pooled connections and
retires the client; use a new `SDFMClient` afterwards.

`from nvidia_sdfm import kumorfm` is the supported surface onto the KumoRFM
driver (`Graph`, `LocalTable`, `Stype`, `Dtype`, `ExplainConfig`, ...); you
never import the driver package directly. The model itself is not on that
surface — you reach it through `client.kumorfm(graph)`.

## Adding a model

Implement `ModelAdapter` in `nvidia_sdfm/adapters/<model>.py` with a typed
request, export it from `nvidia_sdfm/adapters/__init__.py`, and add it to
`nvidia_sdfm.client._default_registry`. Out of tree, build an `AdapterRegistry`
yourself and pass it as `SDFMClient(url, registry=...)`. Models needing a heavy
runtime ship it as a separate driver distribution behind an extra; the adapter
lazy-imports the driver so base installs stay light.

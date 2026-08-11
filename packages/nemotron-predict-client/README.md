# nemotron-predict-client

Client SDK for NVIDIA structured-data foundation model NIMs served behind the
Universal TFM API. A thin, model-agnostic client dispatches typed requests to
per-model adapters; heavy model drivers are optional extras.

## Install

```bash
pip install nemotron-predict-client              # client + every lightweight model (Nemotron Tabular)
pip install nemotron-predict-client[relational]     # adds NemotronRelational (native driver)
pip install nemotron-predict-client[sqlite]      # data-source reads ([duckdb]/[snowflake]/[databricks]/[s3])
pip install nemotron-predict-client[all]         # NemotronRelational, every data-source backend,
                                     # and [databricks-serving]
pip install nemotron-predict-client[explain]     # Nemotron Relational plus the explanation-summary LLM
                                     # client. Deliberately NOT part of [all],
                                     # because it enables the third-party data
                                     # egress described below.
```

## Quickstart

A `PredictClient` owns one connection to a NIM. Requests are typed per model: each
model handle builds its own request type, and the client rejects a request the
target model's adapter does not accept. Each adapter also checks what it knows
it cannot serve — an unsupported task kind or output field for Nemotron Tabular, an
unknown `task_type` or a missing entity table for NemotronRelational — and raises
`PredictError(code="INVALID_REQUEST")` before anything is sent. Everything else is
validated by the NIM.

Nemotron Tabular (single table):

```python
from nemotron_predict import PredictClient

with PredictClient(url='http://localhost:8000') as client:
    model = client.tabular(context_df, target='label', task='classification')
    df = model.predict(predict_df, outputs=['prediction', 'probabilities'])
```

NemotronRelational (relational) — needs `nemotron-predict-client[relational]`:

```python
from nemotron_predict import PredictClient, relational

graph = relational.Graph.from_data({'users': df1, 'items': df2, 'orders': df3})

with PredictClient(url='http://localhost:8000') as client:
    df = client.relational(graph).predict(
        'PREDICT SUM(orders.price, 0, 30, days) FOR items.item_id=1',
        indices=[...],
        run_mode='fast',
    )
```

Set `explain=True` to get a `relational` `Explanation` instead of a bare
DataFrame. The predicted rows stay on `result.prediction`; `result.details`
carries the driver's structured attribution (feature cohorts and subgraphs).
`result.summary` is a natural-language string, either from the backend or, when
the backend returns structured attribution only, generated locally by the SDK
(see the warning below).

```python
with PredictClient(url='http://localhost:8000') as client:
    result = client.relational(graph).predict(
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
> Unless `NEMOTRON_PREDICT_EXPLAIN_LLM_BASE_URL` points elsewhere, that endpoint is
> OpenAI's `https://api.openai.com/v1/` (model `gpt-4.1-mini-2025-04-14`, or
> `NEMOTRON_PREDICT_EXPLAIN_LLM_MODEL`), a non-NVIDIA service. The API key is read from
> `NEMOTRON_PREDICT_EXPLAIN_LLM_API_KEY` and **falls back to the ambient
> `OPENAI_API_KEY`**, so a key exported for an unrelated tool is enough to
> enable this.
>
> Nothing is sent if no key is discoverable or the `nemotron-predict-client[explain]` extra
> is not installed — which is why that extra is not part of `[all]`.
> `result.summary` then carries a message saying so, and the structured
> attribution on `result.details` is unaffected. To keep the data on the machine
> while still getting that attribution, disable the summary explicitly:
>
> ```python
> result = client.relational(graph).predict(query, explain=dict(skip_summary=True))
> ```

Bring your own train table with `predict_task` when you want to supply the
in-context (train) labels directly instead of deriving them from a PQL query.
`context` holds the labelled rows and `predict` the rows to score, both
referencing entities of `entity_table` in the graph; columns default to
`ENTITY` / `TARGET` / `ANCHOR_TIMESTAMP`:

```python
with PredictClient(url='http://localhost:8000') as client:
    df = client.relational(graph).predict_task(
        context=train_df,  # ENTITY, TARGET, [ANCHOR_TIMESTAMP]
        predict=predict_df,  # ENTITY, [ANCHOR_TIMESTAMP]
        task_type='multiclass_classification',
        entity_table='users',
    )
```

`client.models()` lists the models this client has adapters for and
`client.capabilities("nemotron-relational")` describes one of them. Both read the
client-side registry, not the endpoint: a NIM serving only one of these models
still reports both, and a mismatch surfaces as an error from the NIM on the
first prediction. The transport pools connections and retries
transient failures (429/5xx) with backoff; tune it per client with
`PredictClient(url, timeout=30, max_retries=3)`. `timeout` bounds each attempt
rather than the call as a whole, so a retried call can take up to
`(max_retries + 1) * timeout` plus backoff. Each client holds its own transport
and registry, so multiple clients can target different endpoints at once,
including concurrently: a prediction always goes to the endpoint and credential
of the client that started it. The Nemotron Relational driver underneath still keeps a
process-wide configuration that each prediction reconfigures, so drive it
through `PredictClient` rather than mixing in direct `nemotron_relational.init()` calls.
`close()` (or leaving the `with` block) releases the pooled connections and
retires the client; use a new `PredictClient` afterwards.

`from nemotron_predict import relational` is the supported surface onto the NemotronRelational
driver (`Graph`, `LocalTable`, `Stype`, `Dtype`, `ExplainConfig`, ...); you
never import the driver package directly. The model itself is not on that
surface — you reach it through `client.relational(graph)`.

## Adding a model

Implement `ModelAdapter` in `nemotron_predict/adapters/<model>.py` with a typed
request, export it from `nemotron_predict/adapters/__init__.py`, and add it to
`nemotron_predict.client._default_registry`. Out of tree, build an `AdapterRegistry`
yourself and pass it as `PredictClient(url, registry=...)`. Models needing a heavy
runtime ship it as a separate driver distribution behind an extra; the adapter
lazy-imports the driver so base installs stay light.

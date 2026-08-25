# kumo-relational-client

Client for NVIDIA structured-data foundation model NIMs served behind the
Universal TFM API. A thin, model-agnostic client dispatches typed requests to
per-model adapters; heavy model drivers are optional extras.

## Install

```bash
pip install kumo-relational-client              # client + every lightweight model (Kumo Tabular)
pip install kumo-relational-client[relational]     # adds Nemotron Relational (native driver)
pip install kumo-relational-client[sqlite]      # data-source reads ([duckdb]/[snowflake]/[databricks]/[s3])
pip install kumo-relational-client[all]         # Kumo Relational, every data-source backend,
                                     # and [databricks-serving]
pip install kumo-relational-client[explain]     # Kumo Relational plus the explanation-summary LLM
                                     # client. Deliberately NOT part of [all],
                                     # because it enables the third-party data
                                     # egress described below.
```

## Quickstart

A `RelationalClient` owns one connection to a NIM. Requests are typed per model: each
model handle builds its own request type, and the client rejects a request the
target model's adapter does not accept. Each adapter also checks what it knows
it cannot serve, an unsupported task kind or output field for Kumo Tabular, an
unknown `task_type` or a missing entity table for Kumo Relational, and raises
`RelationalError(code="INVALID_REQUEST")` before anything is sent. Everything else is
validated by the NIM.

Kumo Tabular (single table):

```python
from kumo_relational_client import RelationalClient

with RelationalClient(url='http://localhost:8000') as client:
    model = client.tabular(context_df, target='label', task='classification')
    df = model.predict(predict_df, outputs=['prediction', 'probabilities'])
```

Kumo Relational (relational), needs `kumo-relational-client[relational]`:

```python
from kumo_relational_client import RelationalClient, relational

graph = relational.Graph.from_data({'users': df1, 'items': df2, 'orders': df3})

with RelationalClient(url='http://localhost:8000') as client:
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
the backend returns structured attribution only, generated locally by the client
(see the warning below).

```python
with RelationalClient(url='http://localhost:8000') as client:
    result = client.relational(graph).predict(
        'PREDICT SUM(orders.price, 0, 30, days) FOR items.item_id=1',
        explain=True,
    )
    predictions = result.prediction
    attribution = result.details
```

> **Data egress warning, the explanation summary calls a third-party LLM.**
> When the backend returns structured attribution without a summary (which is
> what the NIM does today), the client generates `result.summary` itself by POSTing
> the predictive query, the returned predictions, the cohort analysis and the
> subgraph attribution, **including the raw cell values of the explained
> entity's subgraph**, to an OpenAI-compatible chat-completions endpoint.
> Unless `KUMO_RELATIONAL_EXPLAIN_LLM_BASE_URL` points elsewhere, that endpoint is
> OpenAI's `https://api.openai.com/v1/` (model `gpt-4.1-mini-2025-04-14`, or
> `KUMO_RELATIONAL_EXPLAIN_LLM_MODEL`), a non-NVIDIA service. The API key is read from
> `KUMO_RELATIONAL_EXPLAIN_LLM_API_KEY` and nothing else. Setting that one
> variable is what turns the call on; a key exported for another tool, such as
> `OPENAI_API_KEY`, does not enable it.
>
> Nothing is sent if no key is discoverable or the `kumo-relational-client[explain]` extra
> is not installed, which is why that extra is not part of `[all]`.
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
with RelationalClient(url='http://localhost:8000') as client:
    df = client.relational(graph).predict_task(
        context=train_df,  # ENTITY, TARGET, [ANCHOR_TIMESTAMP]
        predict=predict_df,  # ENTITY, [ANCHOR_TIMESTAMP]
        task_type='multiclass_classification',
        entity_table='users',
    )
```

`client.models()` lists the models this client has adapters for and
`client.capabilities("kumo-relational")` describes one of them. Both read the
client-side registry, not the endpoint: a NIM serving only one of these models
still reports both, and a mismatch surfaces as an error from the NIM on the
first prediction. The transport pools connections and retries
transient failures (429/5xx) with backoff; tune it per client with
`RelationalClient(url, timeout=30, max_retries=3)`. `timeout` bounds each attempt
rather than the call as a whole, so a retried call can take up to
`(max_retries + 1) * timeout` plus backoff. Each client holds its own transport
and registry, so multiple clients can target different endpoints at once,
including concurrently: a prediction always goes to the endpoint and credential
of the client that started it. The Nemotron Relational driver underneath still keeps a
process-wide configuration that each prediction reconfigures, so drive it
through `RelationalClient` rather than mixing in direct `nemotron_relational.init()` calls.
`close()` (or leaving the `with` block) releases the pooled connections and
retires the client; use a new `RelationalClient` afterwards.

`from kumo_relational_client import relational` is the supported surface onto the Kumo Relational
driver (`Graph`, `LocalTable`, `Stype`, `Dtype`, `ExplainConfig`, ...); you
never import the driver package directly. The model itself is not on that
surface, you reach it through `client.relational(graph)`.

## Adding a model

Implement `ModelAdapter` in `kumo_relational_client/adapters/<model>.py` with a typed
request, export it from `kumo_relational_client/adapters/__init__.py`, and add it to
`kumo_relational_client.client._default_registry`. Out of tree, build an `AdapterRegistry`
yourself and pass it as `RelationalClient(url, registry=...)`. Models needing a heavy
runtime ship it as a separate driver distribution behind an extra; the adapter
lazy-imports the driver so base installs stay light.

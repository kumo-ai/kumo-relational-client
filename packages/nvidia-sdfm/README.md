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
`result.summary` is a natural-language string, either from the backend or, when
the backend returns structured attribution only, generated locally by the SDK
(see the warning below).

```python
with SDFMClient(url="http://localhost:8000") as client:
    result = client.kumorfm(graph).predict(
        "PREDICT SUM(orders.price, 0, 30, days) FOR items.item_id=1",
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
> OpenAI's `https://api.openai.com/v1/` (model `gpt-4.1-mini`), a non-NVIDIA
> service. The API key is read from `KUMORFM_EXPLAIN_LLM_API_KEY` and **falls
> back to the ambient `OPENAI_API_KEY`**, so a key exported for an unrelated
> tool is enough to enable this.
>
> Nothing is sent if no key is discoverable or the `kumorfm[explain]` extra is
> not installed. To keep the data on the machine while still getting the
> structured attribution, disable the summary:
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
with SDFMClient(url="http://localhost:8000") as client:
    df = client.kumorfm(graph).predict_task(
        context=train_df,      # ENTITY, TARGET, [ANCHOR_TIMESTAMP]
        predict=predict_df,    # ENTITY, [ANCHOR_TIMESTAMP]
        task_type="multiclass_classification",
        entity_table="users",
    )
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

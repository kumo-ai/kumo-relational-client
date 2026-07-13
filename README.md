# nvidia-sdfm

One client SDK for NVIDIA structured-data foundation model NIMs, served behind the
Universal TFM API. A thin, model-agnostic client dispatches to per-model adapters;
heavy model drivers are optional, installed only when you ask for them.

## Install

| Command | You get |
| --- | --- |
| `pip install nvidia-sdfm` | The client + every lightweight model (TabICL today). Works out of the box. |
| `pip install nvidia-sdfm[kumorfm]` | Adds KumoRFM (pulls the native `kumorfm` driver). |
| `pip install nvidia-sdfm[sqlite]` | Read tables from a data source (`[sqlite]` / `[duckdb]` / `[snowflake]` / `[databricks]`). |
| `pip install nvidia-sdfm[all]` | Everything. |

The rule is dependency weight, not favoritism: a model that does no client-side work
(like TabICL, which just shapes a request the NIM runs) ships in the base wheel; a model
that does heavy client-side work (like KumoRFM: graph building, native neighbor-sampling,
PQL) is an opt-in extra. Data-source drivers are opt-in the same way, via the shared
`sdfm-connectors` package.

## Quickstart

A `SDFMClient` owns one connection to a NIM. Requests are typed per model, so your editor
autocompletes the fields and the client validates them before sending.

TabICL (single table):

```python
from nvidia_sdfm import SDFMClient, TabICLRequest

with SDFMClient(url="http://localhost:8000") as client:
    df = client.predict(TabICLRequest(
        context=context_df,
        predict=predict_df,
        task="classification",
        target="label",
        outputs=["prediction", "probabilities"],
    ))
```

KumoRFM (relational) — needs `nvidia-sdfm[kumorfm]`:

```python
from nvidia_sdfm import SDFMClient, KumoRFMRequest, kumorfm

graph = kumorfm.LocalGraph.from_data({"users": df1, "items": df2, "orders": df3})

with SDFMClient(url="http://localhost:8000") as client:
    df = client.predict(KumoRFMRequest(
        graph=graph,
        query="PREDICT SUM(orders.price, 0, 30, days) FOR items.item_id=1",
        indices=[...],
        run_mode="fast",
    ))
```

Each `SDFMClient` holds its own transport and registry, so multiple clients can target
different endpoints or tenants at once. Discover what a NIM serves with
`client.models()` and `client.capabilities("tabicl")`. The transport pools connections
and retries transient failures (429/5xx) with backoff; tune it per client with
`SDFMClient(url, timeout=30, max_retries=3)`.

`from nvidia_sdfm import kumorfm` is a neutral, explicitly-exported surface for the
driver's `Graph`, `Table`, `KumoRFM`, etc.; you never import the driver package directly.

## Repository layout

A monorepo workspace; every independently released distribution lives under `packages/`
with the same `src/` + `tests/` convention:

```text
rfm-sdk/
├── pyproject.toml              # workspace root: shared tooling only, builds nothing
├── e2e/                        # cross-distribution live harnesses
└── packages/
    ├── nvidia-sdfm/            # the client SDK (pure-python, universal wheel)
    │   └── src/nvidia_sdfm/
    │       ├── core/           #   HTTP transport, response parsing, connectors, dtypes
    │       ├── base.py         #   ModelAdapter interface + AdapterRegistry
    │       ├── adapters/       #   one peer module per model
    │       │   ├── tabicl.py     #   single-table (no driver)
    │       │   └── kumorfm.py    #   relational (lazy-wraps the KumoRFM driver)
    │       └── kumorfm.py      #   explicit, lazily-resolved surface onto the driver
    ├── sdfm-connectors/        # shared data-source connectors (pure-python)
    │   └── src/sdfm_connectors/  #   connect(), read(), quote_ident, resolve_sql; DB drivers via extras
    └── kumorfm/                # the KumoRFM driver (native build)
        └── src/kumorfm/          #   graph, samplers, native kumolib, PQL, HTTP client
```

Two orthogonal axes: the **adapter layer** is symmetric (every model is a peer module
implementing `ModelAdapter`, registered in the `SDFMClient`'s `AdapterRegistry`); a **driver**
package holds a model's heavy runtime, and a model wraps zero or one of them. Each model has
a typed request (`TabICLRequest`, `KumoRFMRequest`) that declares which model it targets;
`client.predict(request)` dispatches on that, validates it against the model's
`capabilities()`, and calls the adapter.

Both the client (flat table reads) and the KumoRFM driver (warehouse connections for its
graph samplers) sit on the shared **`sdfm-connectors`** package, so each warehouse is
reached through one place. The `sqlite`, `snowflake`, and `databricks` connection factories
are shared directly; `duckdb` is provided for flat reads (the driver's duckdb graph sampler
needs the ADBC driver's `adbc_ingest`, which is a separate concern).

## Adding a model

1. Add `packages/nvidia-sdfm/src/nvidia_sdfm/adapters/<model>.py` implementing
   `ModelAdapter`, and register it in `packages/nvidia-sdfm/src/nvidia_sdfm/__init__.py`.
   The core never changes.
2. If the model needs a heavy runtime, add it under `packages/<driver>/` as its own
   distribution and add a `[<model>]` extra; the adapter lazy-imports the driver so base
   installs stay light.
3. A dependency-free model (like TabICL) needs no driver and no extra.

## Why the KumoRFM adapter isn't symmetric with TabICL

`TabICLRequest` (`context` / `predict` / `task` / `target`) maps cleanly onto the Universal
wire envelope, but `KumoRFMRequest` (`graph` / `query` / `indices`) does not: `KumoRFM.predict()`
takes a PQL query plus an entity-graph and builds/samples/sends the request as one fused
operation — there is no standalone "build a payload from two flat DataFrames" step to call
into. Reimplementing that outside the driver would duplicate PQL parsing, subgraph sampling,
and point-in-time correctness logic that already lives (and is tested) there. So
`adapters/kumorfm.py` takes the shape KumoRFM actually needs and normalizes the result into
the same DataFrame shape `core.response` produces for TabICL, so callers get one consistent
return type regardless of adapter.

## Sessions

Not wired yet. `core/transport.py` implements `predict()` only; `create_session` /
`session_predict` / `delete_session` are deferred until the NIM session path is exercised.

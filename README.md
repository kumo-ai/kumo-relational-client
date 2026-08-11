# nemotron-predict-client

One client SDK for NVIDIA structured-data foundation model NIMs, served behind the
Universal TFM API. A thin, model-agnostic client dispatches to per-model adapters;
heavy model drivers are optional, installed only when you ask for them.

## Documentation

Full documentation lives under [`docs/`](docs/index.md):

- [Overview](docs/about/overview.md) — what the SDK is and when to use it
- [Architecture](docs/about/architecture.md) — how the client, adapters, drivers, and connectors fit together
- [Prerequisites](docs/get-started/prerequisites.md) and [Installation](docs/get-started/installation.md)
- [Quickstart](docs/get-started/quickstart.md) — your first Nemotron Tabular and Nemotron Relational predictions
- [Environment Variables](docs/reference/environment-variables.md)

## Install

| Command | You get |
| --- | --- |
| `pip install nemotron-predict-client` | The client + every lightweight model (Nemotron Tabular today). Works out of the box. |
| `pip install nemotron-predict-client[relational]` | Adds NemotronRelational (pulls the native `nemotron_relational` driver). |
| `pip install nemotron-predict-client[sqlite]` | Read tables from a data source (`[sqlite]` / `[duckdb]` / `[snowflake]` / `[databricks]` / `[s3]`). |
| `pip install nemotron-predict-client[all]` | NemotronRelational, every data-source backend, and `[databricks-serving]`. Not `[explain]` or `[relbench]` — see below. |

The rule is dependency weight, not favoritism: a model that does no client-side work
(like Nemotron Tabular, which just shapes a request the NIM runs) ships in the base wheel; a model
that does heavy client-side work (like NemotronRelational: graph building, native neighbor-sampling,
PQL) is an opt-in extra. Data-source drivers are opt-in the same way, via the shared
`nemotron-predict-connectors` package.

Two extras stay outside `[all]` and have to be asked for by name. `[explain]` fills in
`Explanation.summary`, which POSTs row data to a third-party LLM endpoint, so installing it
is a deliberate act; `[relbench]` pulls the RelBench datasets in for `Graph.from_relbench()`.

`[relational]` is a native build. Prebuilt wheels are published for Linux x86-64
(`manylinux_2_28`) on CPython 3.10-3.13 only, and no source distribution is published, so
`pip install "nemotron-predict-client[relational]"` resolves on that platform alone. The base client and
the connectors are pure Python and install anywhere.

## Quickstart

A `PredictClient` owns one connection to a NIM. You run inference through a model handle:
`client.relational(graph)` or `client.tabular(context, target=, task=)`, then `.predict(...)`.

Nemotron Tabular (single table):

```python
from nemotron_predict import PredictClient

with PredictClient(url="http://localhost:8000") as client:
    model = client.tabular(context_df, target="label", task="classification")
    df = model.predict(predict_df, outputs=["prediction", "probabilities"])
```

NemotronRelational (relational) — needs `nemotron-predict-client[relational]`:

```python
from nemotron_predict import PredictClient, relational

graph = relational.Graph.from_data({"users": df1, "items": df2, "orders": df3})

with PredictClient(url="http://localhost:8000") as client:
    df = client.relational(graph).predict(
        "PREDICT SUM(orders.price, 0, 30, days) FOR items.item_id=1",
        indices=[...],
        run_mode="fast",
    )
```

Each `PredictClient` holds its own transport and registry, so multiple clients can target
different endpoints or tenants at once, including concurrently from several threads:
a prediction always goes to the endpoint and credential of the client that started it.
The Nemotron Relational driver underneath still keeps a process-wide configuration that each
prediction reconfigures, so drive it through `PredictClient` rather than mixing in direct
`nemotron_relational.init()` calls. `client.models()` and `client.capabilities("nemotron-tabular")` describe
the client's own adapter registry, not the connected endpoint: a NIM serving only one of
these models still reports both, and the mismatch surfaces as an error from the NIM on
the first prediction. The transport pools connections and retries transient failures
(429, 500, 502, 503, 504) with backoff; tune it per client with
`PredictClient(url, timeout=30, max_retries=3)`.

`from nemotron_predict import relational` is a neutral, explicitly-exported surface for the
driver's `Graph`, `Table`, etc. for building graphs; you never import the driver package directly.

## Repository layout

A monorepo workspace; every independently released distribution lives under `packages/`
with the same `src/` + `tests/` convention:

```text
nemotron-predict-client/
├── pyproject.toml              # workspace root: shared tooling only, builds nothing
├── e2e/                        # cross-distribution live harnesses
├── docs/                       # user-facing documentation
├── examples/                   # runnable notebooks and scripts
├── scripts/                    # release and maintenance tooling
└── packages/
    ├── nemotron-predict-client/            # the client SDK (pure-python, universal wheel)
    │   └── src/nemotron_predict/
    │       ├── client.py       #   PredictClient: the entry point and its registry
    │       ├── models.py       #   the per-model handles the client hands back
    │       ├── requests.py     #   the internal typed requests handles build
    │       ├── errors.py       #   the exception hierarchy
    │       ├── core/           #   HTTP transport, response parsing, connectors, dtypes
    │       ├── base.py         #   ModelAdapter interface + AdapterRegistry
    │       ├── adapters/       #   one peer module per model
    │       │   ├── tabicl.py     #   single-table (no driver)
    │       │   └── nemotron_relational.py    #   relational (lazy-wraps the Nemotron Relational driver)
    │       └── nemotron_relational.py      #   explicit, lazily-resolved surface onto the driver
    ├── nemotron-predict-connectors/        # shared data-source connectors (pure-python)
    │   └── src/nemotron_predict_connectors/  #   connect(), read(), quote_ident, resolve_sql; DB drivers via extras
    └── nemotron_relational/                # the Nemotron Relational driver (native build)
        └── src/nemotron_relational/          #   graph, samplers, native relationallib, PQL, HTTP client
```

Two orthogonal axes: the **adapter layer** is symmetric (every model is a peer module
implementing `ModelAdapter`, registered in the `PredictClient`'s `AdapterRegistry`); a **driver**
package holds a model's heavy runtime, and a model wraps zero or one of them. Each model has
an internal typed request that declares which model it targets; the client dispatches on
that, checks it is the type that model's adapter accepts, and calls the adapter. Nothing is
checked against `capabilities()`, which is a discovery accessor for callers, not a gate on
the dispatch path. You reach all of this through the handles (`client.relational(...)` /
`client.tabular(...)`); the request types are not exported from `nemotron_predict`.

Both the client (flat table reads) and the Nemotron Relational driver (warehouse connections for its
graph samplers) sit on the shared **`nemotron-predict-connectors`** package, so each warehouse is
reached through one place. The `sqlite`, `duckdb`, `snowflake`, and `databricks` connection
factories are shared directly.

## Adding a model

1. Add `packages/nemotron-predict-client/src/nemotron_predict/adapters/<model>.py` implementing
   `ModelAdapter`, and register it in `_default_registry()` in
   `packages/nemotron-predict-client/src/nemotron_predict/client.py`. The core never changes.
2. If the model needs a heavy runtime, add it under `packages/<driver>/` as its own
   distribution and add a `[<model>]` extra; the adapter lazy-imports the driver so base
   installs stay light.
3. A dependency-free model (like Nemotron Tabular) needs no driver and no extra.

## Why the Nemotron Relational adapter isn't symmetric with Nemotron Tabular

The internal `NemotronTabularRequest` (`context` / `predict` / `task` / `target`) maps cleanly onto the
Universal wire envelope, but `NemotronRelationalRequest` (`graph` / `query` / `indices`) does not — these
are the shapes the handles build for the adapters, not a user-facing API. `NemotronRelational.predict()`
takes a PQL query plus an entity-graph and builds/samples/sends the request as one fused
operation — there is no standalone "build a payload from two flat DataFrames" step to call
into. Reimplementing that outside the driver would duplicate PQL parsing, subgraph sampling,
and point-in-time correctness logic that already lives (and is tested) there. So
`adapters/relational.py` takes the shape Nemotron Relational actually needs and normalizes the result into
the same DataFrame shape `core.response` produces for Nemotron Tabular, so callers get one consistent
return type regardless of adapter.

## Sessions

A session pins `model` / `task` / `schema` / `context` on the NIM so later calls send only the
rows to score. Both model paths use them, and neither exposes them: they are an internal
transport optimisation and never change a prediction.

- **Nemotron Tabular** — `client.tabular(context, ...)` reuses one session for the life of the handle.
  It is opened on the second `predict()` against the same context, so scoring a single table
  costs exactly one request as before, and every call after that carries the rows alone.
- **NemotronRelational** — a multi-batch `predict()` opens one session for the run and deletes it at the
  end. Set `KUMORFM_DISABLE_SESSIONS=1` to force the stateless path.

Both fall back to `POST /v1/predictions` when the NIM answers 404/405/501 on session creation,
and both re-pin the context transparently if a session expires.

## License & Contributing

This project is released under the [Apache License 2.0](./LICENSE); third-party
components are listed in [`NOTICE`](./NOTICE). External contributions are
welcome under the Developer Certificate of Origin, see
[`CONTRIBUTING.md`](./CONTRIBUTING.md).

### Licensing

Each package under `packages/` carries its own `LICENSE` file so that its
`pyproject.toml` can reference it via `license-files`. Those copies are the same
Apache-2.0 terms as the root [`LICENSE`](./LICENSE); they are not separate or
additional terms.

The one third-party exception is `mermaid.js`, which is vendored into this
repository as source at
`packages/nemotron-relational/src/nemotron_relational/rfm/assets/mermaid.min.js` for offline graph
visualization. It is MIT-licensed and is itself a bundle: the components
embedded inside it, including DOMPurify (Apache-2.0 and Mozilla Public License
2.0), are enumerated with their copyrights in the third-party section at the end
of the root [`LICENSE`](./LICENSE). Every other dependency is resolved by the
package manager at install time rather than shipped here.

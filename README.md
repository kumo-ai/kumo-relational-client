# NVIDIA Kumo Relational Client

Make predictions on tables and relational data with NVIDIA Kumo
structured-data foundation models, without training a model for each dataset.

```bash
pip install kumo-relational-client[relational]
```

## Overview

Structured-data foundation models predict in context. You give the model
labelled rows and the rows you want scored, and it answers in one pass. There
is no training loop, no hyperparameter search, and no per-dataset artifact to
manage.

This client talks to those models. It builds the request, sends it to a NIM
inference endpoint, and gives you back a pandas DataFrame. Inference runs in
the NIM, so this package downloads no weights and needs no GPU.

One model is available through the client today:

| Model | For | Reached by |
| --- | --- | --- |
| `kumo-relational` | several related tables, joined as a graph | `client.relational(...)` |

## Getting Started

You need a running NIM to predict against. Point the client at it and score a
set of related tables:

```python
import pandas as pd
from kumo_relational_client import RelationalClient, relational

users = pd.DataFrame({"user_id": [1, 2, 3]})
orders = pd.DataFrame({
    "order_id": [1, 2, 3],
    "user_id": [1, 2, 1],
    "price": [10.0, 20.0, 30.0],
    "ts": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
})

graph = relational.Graph.from_data({"users": users, "orders": orders})

with RelationalClient(url="http://localhost:8000") as client:
    model = client.relational(graph)
    print(model.predict(
        "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id",
        indices=[1, 2],
    ))
```

`client.models()` lists what the client can dispatch and
`client.capabilities("kumo-relational")` describes it. Neither needs a
live endpoint.

## Requirements

- **Python** 3.10 to 3.13.
- **A reachable NIM** serving `kumo-relational`. See
  [Getting a NIM](#getting-a-nim).
- **No GPU** on the client. The NIM owns that.
- **OS and architecture.** The client and connectors are pure Python and
  install anywhere. The `[relational]` extra is a native build, published as
  `manylinux_2_28` wheels for Linux x86-64 on CPython 3.10 to 3.13, with no
  source distribution.

## Getting a NIM

The client sends requests; the model runs in a NIM you point it at. There are
two ways to have one.

**A hosted endpoint.** Find the model in the
[NVIDIA API catalog](https://build.nvidia.com), generate an API key, and pass
the endpoint and key to `RelationalClient`. Nothing to deploy.

**The container, run yourself.** Pull the NIM from the
[NGC catalog](https://catalog.ngc.nvidia.com) and run it on your own GPU host,
then point the client at it. Use this when the data cannot leave your
environment.

Either way the client needs only the URL and, for a hosted endpoint, the key:

```python
client = RelationalClient(url="https://...", api_key="nvapi-...")
```

`RelationalClient` takes both explicitly; it reads nothing from the
environment. For what the models do and how to size a deployment, see the
[product documentation](https://docs.nvidia.com/sdgm/rfm/overview).

## Installation

| Command | You get |
| --- | --- |
| `pip install kumo-relational-client` | The client on its own. |
| `pip install kumo-relational-client[relational]` | Adds Kumo Relational, pulling in the native driver. |
| `pip install kumo-relational-client[sqlite]` | Reads source tables from a warehouse. Also `[duckdb]`, `[snowflake]`, `[databricks]`, `[postgres]`, `[s3]`. |
| `pip install kumo-relational-client[all]` | Kumo Relational, every warehouse backend, and `[databricks-serving]`. |

What ships in the base wheel is decided by dependency weight, not by
preference. A model that does heavy client-side work, like Kumo Relational
with its graph building, native neighbor sampling and PQL, is an opt-in extra. Warehouse drivers are opt-in the
same way, through the shared `kumo-connectors` package.

Two extras stay outside `[all]` and must be asked for by name. `[explain]`
fills in `Explanation.summary`, which posts row data to a third-party LLM
endpoint, so installing it is a deliberate act. `[relbench]` pulls in the
RelBench datasets for `Graph.from_relbench()`.

## Usage

A `RelationalClient` owns one connection to a NIM. You run inference through a
model handle, then call `.predict(...)` on it.

### Related tables

You describe the data as a graph and ask for a prediction in PQL, a small query
language for predictive questions. This runs as written, against a ready-made
dataset it downloads on first use:

```bash
pip install "kumo-relational-client[relational,relbench]"
```

```python
from kumo_relational_client import RelationalClient, relational

graph = relational.Graph.from_relbench("f1")

with RelationalClient(url="http://localhost:8000") as client:
    df = client.relational(graph).predict(
        "PREDICT COUNT(results.*, 0, 90, days) > 3 FOR EACH drivers.driverId",
        indices=[814, 0, 842, 831, 3, 829],
    )
```

For your own tables, use `relational.Graph.from_data({"users": users_df,
"orders": orders_df})`. Pick an aggregation window your data can support: a
query over `0, 90, days` needs 90 days of history before the anchor time.

`from kumo_relational_client import relational` is the supported surface for `Graph`,
`Table` and friends. You never import the driver package directly.

### Several endpoints at once

Each `RelationalClient` holds its own transport and adapter, so clients can
target different endpoints or tenants concurrently, including from several
threads. A prediction always goes to the endpoint and credential of the client
that started it.

The transport pools connections and retries transient failures (429, 500, 502,
503, 504) with backoff. Tune it per client with
`RelationalClient(url, timeout=30, max_retries=3)`.

Two things are worth knowing. `client.models()` describes the client itself
rather than the connected endpoint, so it answers without a live connection and
a NIM serving something else surfaces as an error on the first prediction. And
the relational driver keeps a process-wide configuration that
each prediction reconfigures, so drive it through `RelationalClient` rather than
mixing in direct `kumo_relational_engine.init()` calls.

### Errors

Every failure raises `RelationalError` or a subclass carrying a stable `code`, so
branch on the code rather than the message:

```python
from kumo_relational_client.errors import RelationalError

try:
    df = model.predict(rows)
except RelationalError as exc:
    print(exc.code)
```

`NimRequestError` also carries `status_code`, which distinguishes a retriable
5xx or 429 from a 4xx that needs a different request.

## Documentation

Full documentation is under [`docs/`](docs/index.md):

- [Overview](docs/about/overview.md), what the client is and when to use it
- [Architecture](docs/about/architecture.md), how the client, adapter, driver
  and connectors fit together
- [Prerequisites](docs/get-started/prerequisites.md) and
  [Installation](docs/get-started/installation.md)
- [Quickstart](docs/get-started/quickstart.md)
- [Environment variables](docs/reference/environment-variables.md)
- [Prediction output](docs/reference/prediction-output.md), including how many
  rows each task returns

## Releases

Three packages are released together and share one version, so 1.0.0 works
with 1.0.0:

| Package | Import | What it is |
| --- | --- | --- |
| `kumo-relational-client` | `kumo_relational_client` | the client and its model handles |
| `kumo-connectors` | `kumo_connectors` | reads source tables from SQLite, DuckDB, Snowflake, Databricks, PostgreSQL, and S3 |
| `kumo-relational-engine` | `kumo_relational_engine` | the relational driver: graph building, PQL, and a native neighbor sampler |

Release history is in [`CHANGELOG.md`](CHANGELOG.md).

## Contributing

Contributions are welcome under the Developer Certificate of Origin. Start with
[`CONTRIBUTING.md`](CONTRIBUTING.md) and please read
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

```bash
git clone https://github.com/NVIDIA/kumo-relational-client.git
cd kumo-relational-client
python -m pip install -e './packages/kumo-relational-client[test]'
python -m pytest packages/kumo-relational-client/tests -m 'not live_nim' -q
```

## Governance and support

- Who decides what, and how a change gets merged:
  [`GOVERNANCE.md`](GOVERNANCE.md)
- Maintainers and the escalation path: [`MAINTAINERS.md`](MAINTAINERS.md)
- Support level and where to ask: [`SUPPORT.md`](SUPPORT.md)

## Security

Please do not open a public issue for a security problem. Vulnerability
reporting goes to NVIDIA PSIRT. See [`SECURITY.md`](SECURITY.md), which also
records what is a deliberate design decision rather than a vulnerability.

## License

Apache-2.0. See [`LICENSE`](LICENSE). Third-party components are listed in
[`NOTICE`](NOTICE).

Each package under `packages/` carries its own copy of `LICENSE` so its
`pyproject.toml` can reference it. Those copies are the same terms as the root
file, not additional ones.

One third-party component is vendored rather than resolved at install time:
`mermaid.js`, bundled for offline graph visualization. It is MIT licensed and
is itself a bundle, so the components inside it, including DOMPurify under
Apache-2.0 and MPL-2.0, are enumerated with their copyrights in the third-party
section of [`LICENSE`](LICENSE).

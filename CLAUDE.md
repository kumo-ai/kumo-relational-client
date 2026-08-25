# Working in this repository

The NVIDIA Kumo Relational Client is the Python client for NVIDIA's structured-data
foundation models, served behind the Universal TFM API by a NIM. It does not
serve models itself: every prediction is an HTTP call to a NIM you point it at.

## The three packages

| Package | Import | What it is |
| --- | --- | --- |
| `kumo-relational-client` | `kumo_relational_client` | The client. Pure Python, no heavy dependencies. |
| `nemotron-structured-connectors` | `nemotron_structured_connectors` | Reads source tables from sqlite, duckdb, Snowflake, Databricks, S3. |
| `nemotron-relational` | `nemotron_relational` | The relational driver: graph building, PQL, and a compiled neighbor sampler. |

`nemotron-relational` ships prebuilt wheels for Linux x86_64 (`manylinux_2_28`)
on CPython 3.10-3.13 only, and no source distribution. It does not install on
macOS, Windows, or arm64. The client and connectors are pure Python and install
anywhere.

## Two models, one client

`RelationalClient` dispatches to whichever model you ask for:

```python
from kumo_relational_client import RelationalClient

with RelationalClient(url='http://localhost:8000') as client:
    model = client.tabular(context_df, target='label', task='classification')
    df = model.predict(rows)
```

`client.tabular(...)` returns a `TabularModel` and sends `kumo-tabular`.
`client.relational(graph)` returns a `RelationalModel` and sends
`nemotron-relational`; it needs the `[relational]` extra. The method names the
capability, so a future model of the same shape is a `model=` argument rather
than a new method.

`client.models()` lists what the client can dispatch, and
`client.capabilities(model)` describes one model. Neither needs a live endpoint.

## Things that are easy to get wrong

**Join predictions on the entity, never on row position.** The number of rows a
prediction returns depends on the task: classification returns one row per
entity, multiclass one row per entity *per class*, temporal link prediction `k`
rows per entity under `RANK TOP k`, and forecasting one row per entity per
forecast step. Row order is not a contract. See
`docs/reference/prediction-output.md`.

**Drive the relational model only through `RelationalClient`.** The driver keeps a
process-wide configuration that each prediction reconfigures. Predictions are
pinned to their own client and are unaffected, but calling
`nemotron_relational.init()` directly in the same process changes what that
global points at. Do not mix the two.

**Branch on the error code, not the message.** Every failure raises
`RelationalError` or a subclass carrying a stable `code`: `MISSING_EXTRA`,
`TRANSPORT_ERROR`, `INVALID_REQUEST`, `UNKNOWN_MODEL`, and the connector codes
`CONNECT_FAILED`, `QUERY_FAILED`, `READ_FAILED`, `DRIVER_LOAD_FAILED`.
`NimRequestError` also carries `status_code`, so a retriable 5xx or 429 is
distinguishable from a 4xx that needs a different request. `MissingExtraError`
names the exact `pip install` that fixes it.

## Where to read next

- `docs/get-started/`: install, prerequisites, quickstart.
- `docs/about/`: what the client is and how a request flows.
- `docs/reference/`: environment variables and prediction output shapes.
- `examples/kumo_relational_quickstart.ipynb`: a worked end-to-end notebook.

## Running the tests

Each package is installed editable and tested from its own directory:

```bash
python -m pip install -e './packages/kumo-relational-client[test]'
python -m pytest packages/kumo-relational-client/tests -m 'not live_nim' -q
```

Tests marked `live_nim` need a running NIM and are excluded by default. The
relational package's tests run from `packages/nemotron-relational`, because some
of them invoke `scripts/` and read paths relative to the working directory.

Lint and format with `ruff` before pushing; the formatter owns line length.

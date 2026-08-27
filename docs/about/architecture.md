---
title: "NVIDIA Kumo Relational Client Architecture"
description: "How the NVIDIA Kumo Relational Client is structured: a thin client, one model adapter, an optional model driver, and shared data-source connectors."
template-library-version: "1.0.0"
---

# NVIDIA Kumo Relational Client Architecture

The NVIDIA Kumo Relational Client separates a small, universal client from the
heavy runtime the model needs. The client stays thin because the model
is a peer adapter, while a model's optional driver holds its client-side compute.

## High-Level Architecture Diagram

```mermaid
flowchart TD
    app["Your application"] --> client["RelationalClient (kumo-relational-client)"]
    client --> rfm["Kumo Relational adapter"]
    rfm --> driver["kumo_relational_engine driver: graph, sampler, PQL"]
    client --> transport["Transport (HTTP)"]
    driver --> nimclient["NimClient (HTTP)"]
    driver -.-> serving["DatabricksServingClient"]
    transport --> nim["Universal TFM NIM"]
    nimclient --> nim
    serving --> endpoint["Databricks Model Serving endpoint"]
    client --> connectors["kumo-connectors"]
    driver --> connectors
```

## Functional Layers

### Client Layer

`RelationalClient` owns the endpoint address and one adapter. Because each
client owns its own adapter and configuration, several clients can target
different endpoints or tenants in the same process, concurrently: every
prediction is issued against the endpoint and credential of the client that
started it.

Requests do not all leave through the same object. The client holds a
`Transport`, a pooled HTTP session with retry and backoff, which serves the one
call the client makes itself, `/v1/health/ready`. Predictions do not go through
it: the client hands its address and credential to the driver, which opens its
own pooled session (`NimClient`) and sends from there. The two are separate implementations of the same HTTP contract, because
`kumo_relational_engine` cannot depend on `kumo-relational-client`; the dependency runs the other way.

The Kumo Relational driver underneath does keep a process-wide configuration, which
each prediction reconfigures. The adapter applies that configuration and
resolves the resulting client as one atomic step, then binds it to that
prediction, so a concurrent prediction from a differently configured client
cannot re-point it. What remains shared is the driver global itself: direct
`kumo_relational_engine.init()` callers, and anything else reading it, see whichever client
configured it last. Drive the driver through `RelationalClient` only.

### Adapter Layer

The model is served by one adapter module. It advertises its
capabilities, shapes an internal typed request (`KumoRelationalRequest`,
`KumoRelationalTaskRequest`,
built by the model handles, and not importable from `kumo_relational_client`) into the
Universal TFM API envelope, and normalizes the NIM's response into a consistent
pandas DataFrame.

### Driver Layer

A driver is a model's heavy client-side runtime, packaged as its own
distribution. The Kumo Relational driver (`kumo_relational_engine`) performs graph building, native
neighbor sampling through a compiled extension, and PQL parsing. The Kumo Relational
adapter lazy-imports this driver, so base installs stay lightweight and
platform-independent.

## Data Flow

1. Your application builds a model handle with `client.relational(...)` and
   calls `predict` on it.
2. The handle builds that model's internal typed request; the client checks it is
   the type the model's adapter accepts and dispatches to it. Nothing is checked
   against `capabilities()`, which describes the client-side adapter for callers
   who ask and is not consulted on this path.
3. The adapter builds the Universal TFM API request: the driver parses the PQL
   query, samples the relevant subgraph, and materializes the request.
4. The request goes to the NIM over HTTP, with retry on transient failures,
   through the driver's own `NimClient`. A client built with
   `RelationalClient.for_databricks_serving`
   sends through `DatabricksServingClient` instead, which invokes a named
   Model Serving endpoint through the Databricks SDK rather than speaking
   HTTP.
5. The adapter normalizes the response into a pandas DataFrame and returns it.

## Deployment Topologies

The client is a client library; it connects to a NIM you deploy and operate.

- **Local NIM.** Point `RelationalClient(url=...)` at a NIM running on `localhost`.
- **Networked NIM.** Point the client at any reachable NIM endpoint. If the
  deployment fronts the NIM with an authenticating gateway, pass an `api_key`.
- **Databricks Model Serving.** Build the client with
  `RelationalClient.for_databricks_serving(endpoint_name)` to reach a Kumo Relational model
  served inside a Databricks workspace.

## Service Interactions

The client speaks the Universal TFM API. Every prediction is a
`POST /v1/predictions`, except that Kumo Relational switches to the session routes
under `/v1/sessions` when a prediction splits into more than one batch and is
reproducible: sessions upload the context once and reuse it across the batches,
so they need a `random_seed`, and they are skipped when an explanation is
requested. Standard NIM management endpoints are provided by the NIM runtime,
not by the client.

Two endpoints are read rather than predicted against, and not by the same
caller. `RelationalClient.health_ready()` issues `GET /v1/health/ready` and reports
whether it answered 200. The Kumo Relational driver checks more before its first
prediction: it reads `/v1/health/ready` for a ready status and then
`/v1/models`, and fails if the endpoint does not advertise
`kumo-relational`. Nothing
on the client path reads `/v1/models`.

## External Integration Points

- **Data sources.** Both the client and the Kumo Relational driver read tables through
  the shared `kumo-connectors` package (SQLite, DuckDB, Snowflake, Databricks),
  so each warehouse is reached through one place.
- **NIM endpoint.** Any NIM that implements the Universal TFM API. The Kumo Relational
  path additionally requires the endpoint to advertise `kumo-relational` in
  `/v1/models`.
- **Databricks Model Serving.** `RelationalClient.for_databricks_serving(name)`
  targets a named serving endpoint through the Databricks SDK. There is no base
  URL and no HTTP session on this path, and it serves Kumo Relational only.

## Repository Layout

A monorepo workspace. Every independently released distribution lives under
`packages/` with the same `src/` and `tests/` convention:

```text
kumo-relational-client/
├── pyproject.toml              # workspace root: shared tooling only, builds nothing
├── e2e/                        # cross-distribution live harnesses
├── docs/                       # user-facing documentation
├── examples/                   # runnable notebooks and scripts
├── scripts/                    # release and maintenance tooling
└── packages/
    ├── kumo-relational-client/            # the client (pure python, universal wheel)
    │   └── src/kumo_relational_client/
    │       ├── client.py       #   RelationalClient: the entry point
    │       ├── models.py       #   the handles the client hands back
    │       ├── requests.py     #   the internal typed requests handles build
    │       ├── errors.py       #   the exception hierarchy
    │       ├── core/           #   HTTP transport, connectors, serving targets
    │       ├── base.py         #   ModelCapabilities + the result type
    │       ├── adapters/       #   one peer module per model
    │       │   └── relational.py #   relational (lazy-wraps the driver)
    │       └── relational.py  #   explicit, lazily-resolved surface onto the driver
    ├── kumo-connectors/        # shared data-source connectors (pure python)
    │   └── src/kumo_connectors/
    └── kumo-relational-engine/                # the relational driver (native build)
        └── src/kumo_relational_engine/
```

## Serving Another Model

There is no registry and no adapter interface: the client supports
`kumo-relational` and dispatches to one adapter. A second model would mean a
second adapter module and a branch in `RelationalClient` -- a deliberate change,
not an extension point. A model needing a heavy runtime would ship it under
`packages/<driver>/` as its own distribution behind an extra, the way the
relational driver does, so base installs stay light.

## Why the Relational Adapter Wraps the Driver

`KumoRelationalRequest` (`graph`, `query`, `indices`) does not map cleanly onto
the wire envelope. It is the shape the handle builds for the adapter, not a
user-facing API.

`KumoRelational.predict()` takes a PQL query plus an entity graph and
builds, samples and sends the request as one fused operation. There is no
standalone "build a payload from two flat DataFrames" step to call into.
Reimplementing that outside the driver would duplicate PQL parsing, subgraph
sampling and point-in-time correctness logic that already lives, and is tested,
there.

So `adapters/relational.py` takes the shape the driver actually needs and
normalizes the driver's result into the DataFrame shape callers get back.

## Sessions

A session pins `model`, `task`, `schema` and `context` on the NIM so later
calls send only the rows to score. The client uses them without exposing them:
they are a transport optimisation and never change a prediction.

A multi-batch `predict()` opens one session for the run and deletes it at the
end. Set `KUMO_RELATIONAL_DISABLE_SESSIONS=1` to force the stateless path.

It falls back to `POST /v1/predictions` when the NIM answers 404, 405 or 501 on
session creation, and re-pins the context transparently if a session expires.

## Related Topics

- [NVIDIA Kumo Relational Client Documentation](overview.md)
- [Quickstart for the NVIDIA Kumo Relational Client](../get-started/quickstart.md)
- [NVIDIA Kumo Relational Client Environment Variables](../reference/environment-variables.md)

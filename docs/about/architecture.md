---
title: "NVIDIA SDFM SDK Architecture"
description: "How the NVIDIA SDFM SDK is structured: a model-agnostic client, per-model adapters, optional model drivers, and shared data-source connectors."
template-library-version: "1.0.0"
---

# NVIDIA SDFM SDK Architecture

The NVIDIA SDFM SDK separates a small, universal client from the heavy runtimes
that individual models need. The client is symmetric across models — every model
is a peer adapter — while a model's optional driver holds its client-side compute.

## High-Level Architecture Diagram

```mermaid
flowchart TD
    app["Your application"] --> client["PredictClient (nemotron-predict-client)"]
    client --> registry["AdapterRegistry"]
    registry --> tabicl["TabICL adapter"]
    registry --> rfm["NemotronRelational adapter"]
    rfm --> driver["nemotron_relational driver: graph, sampler, PQL"]
    tabicl --> transport["Transport (HTTP)"]
    driver --> kumoclient["RelationalClient (HTTP)"]
    driver -.-> serving["DatabricksServingClient"]
    transport --> nim["Universal TFM NIM"]
    kumoclient --> nim
    serving --> endpoint["Databricks Model Serving endpoint"]
    client --> connectors["nemotron-predict-connectors"]
    driver --> connectors
```

## Functional Layers

### Client Layer

`PredictClient` owns the endpoint address and an `AdapterRegistry`. Because each
client owns its own registry and configuration, several clients can target
different endpoints or tenants in the same process, concurrently: every
prediction is issued against the endpoint and credential of the client that
started it.

Requests do not all leave through the same object. The client holds a
`Transport`, a pooled HTTP session with retry and backoff, and TabICL predicts
through it. NemotronRelational does not: the client hands its address and credential to
the driver, which opens its own pooled session (`RelationalClient`) and sends from
there. The two are separate implementations of the same HTTP contract, because
`nemotron_relational` cannot depend on `nemotron-predict-client` — the dependency runs the other way.

The NemotronRelational driver underneath does keep a process-wide configuration, which
each prediction reconfigures. The adapter applies that configuration and
resolves the resulting client as one atomic step, then binds it to that
prediction, so a concurrent prediction from a differently configured client
cannot re-point it. What remains shared is the driver global itself: direct
`nemotron_relational.init()` callers, and anything else reading it, see whichever client
configured it last. Drive the driver through `PredictClient` only.

### Adapter Layer

Every model is a peer module implementing the `ModelAdapter` interface and
registered in the client's `AdapterRegistry`. An adapter advertises its
capabilities, shapes an internal typed request (`TabICLRequest`, `NemotronRelationalRequest`
— built by the model handles, and not importable from `nemotron_predict`) into the
Universal TFM API envelope, and normalizes the NIM's response into a consistent
pandas DataFrame. Adding a model means adding one adapter module — the client
core does not change.

### Driver Layer

A driver is a model's heavy client-side runtime, packaged as its own
distribution. The NemotronRelational driver (`nemotron_relational`) performs graph building, native
neighbor sampling through a compiled extension, and PQL parsing. The NemotronRelational
adapter lazy-imports this driver, so base installs stay lightweight and
platform-independent. TabICL requires no driver.

## Data Flow

1. Your application builds a model handle with `client.tabicl(...)` or
   `client.relational(...)` and calls `predict` on it.
2. The handle builds that model's internal typed request; the client checks it is
   the type the model's adapter accepts and dispatches to it. Nothing is checked
   against `capabilities()`, which describes the client-side adapter for callers
   who ask and is not consulted on this path.
3. The adapter builds the Universal TFM API request. For NemotronRelational, the driver
   parses the PQL query, samples the relevant subgraph, and materializes the
   request; for TabICL, the adapter serializes the context and predict tables
   directly.
4. The request goes to the NIM over HTTP, with retry on transient failures:
   TabICL sends through the client's `Transport`, NemotronRelational through the driver's
   own `RelationalClient`. A client built with `PredictClient.for_databricks_serving`
   sends through `DatabricksServingClient` instead, which invokes a named
   Model Serving endpoint through the Databricks SDK rather than speaking
   HTTP.
5. The adapter normalizes the response into a pandas DataFrame and returns it.

## Deployment Topologies

The SDK is a client library; it connects to a NIM you deploy and operate.

- **Local NIM.** Point `PredictClient(url=...)` at a NIM running on `localhost`.
- **Networked NIM.** Point the client at any reachable NIM endpoint. If the
  deployment fronts the NIM with an authenticating gateway, pass an `api_key`.
- **Databricks Model Serving.** Build the client with
  `PredictClient.for_databricks_serving(endpoint_name)` to reach a NemotronRelational model
  served inside a Databricks workspace.

## Service Interactions

The SDK speaks the Universal TFM API. Every prediction is a
`POST /v1/predictions`, except that NemotronRelational switches to the session routes
under `/v1/sessions` when a prediction splits into more than one batch and is
reproducible: sessions upload the context once and reuse it across the batches,
so they need a `random_seed`, and they are skipped when an explanation is
requested. Standard NIM management endpoints are provided by the NIM runtime,
not by the SDK.

Two endpoints are read rather than predicted against, and not by the same
caller. `PredictClient.health_ready()` issues `GET /v1/health/ready` and reports
whether it answered 200. The NemotronRelational driver checks more before its first
prediction: it reads `/v1/health/ready` for a ready status and then
`/v1/models`, and fails if the endpoint does not advertise
`nemotron-relational-v1`. Nothing
on the TabICL path reads `/v1/models`.

## External Integration Points

- **Data sources.** Both the client and the NemotronRelational driver read tables through
  the shared `nemotron-predict-connectors` package (SQLite, DuckDB, Snowflake, Databricks),
  so each warehouse is reached through one place.
- **NIM endpoint.** Any NIM that implements the Universal TFM API. The NemotronRelational
  path additionally requires the endpoint to advertise
  `nemotron-relational-v1` in
  `/v1/models`.
- **Databricks Model Serving.** `PredictClient.for_databricks_serving(name)`
  targets a named serving endpoint through the Databricks SDK. There is no base
  URL and no HTTP session on this path, and it serves NemotronRelational only.

## Related Topics

- [NVIDIA SDFM SDK Documentation](overview.md)
- [Quickstart for the NVIDIA SDFM SDK](../get-started/quickstart.md)
- [NVIDIA SDFM SDK Environment Variables](../reference/environment-variables.md)

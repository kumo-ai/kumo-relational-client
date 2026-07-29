---
title: "NVIDIA SDFM SDK Documentation"
description: "Overview of the NVIDIA SDFM SDK, the Python client for structured-data foundation model NIMs served behind the Universal TFM API."
template-library-version: "1.0.0"
---

# NVIDIA SDFM SDK Documentation

The NVIDIA SDFM (Structured Data Foundation Models) SDK is one Python client for
NVIDIA's structured-data foundation model NIMs, served behind the Universal TFM
API. A thin, model-agnostic client dispatches typed requests to per-model
adapters; heavy model drivers are optional and installed only when you ask for
them. Two models are available today: **TabICL** (single-table, in-context
learning) and **KumoRFM** (relational, graph-aware in-context learning).

## Benefits

- **One client for every model.** A single `SDFMClient` connects to a NIM and
  serves any registered model through one typed `predict` call.
- **Pay only for what you use.** The base install is pure Python and works on
  every platform. Heavy drivers, such as KumoRFM's native graph sampler, are
  opt-in extras.
- **NIM-first and secure by default.** The client talks to a NIM you control;
  your data stays on your infrastructure, and authentication is owned by the
  deployment rather than the API contract.
- **Typed, discoverable requests.** Each model has a typed request object, so
  your editor autocompletes the fields and the client validates them before
  sending.

## Skip Ahead

- To install the SDK, refer to [Installation Guide for the NVIDIA SDFM SDK](../get-started/installation.md).
- To run your first prediction, refer to [Quickstart for the NVIDIA SDFM SDK](../get-started/quickstart.md).
- To understand how the SDK is put together, refer to [NVIDIA SDFM SDK Architecture](architecture.md).
- To look up configuration, refer to [NVIDIA SDFM SDK Environment Variables](../reference/environment-variables.md).

## Use Cases

### Score a Single Table with TabICL

Provide a table of labeled context rows and a table of rows to predict, and
TabICL returns predictions in one forward pass — no per-dataset training.

### Predict Over Relational Data with KumoRFM

Build a graph from related tables (for example, users, items, and orders), then
express a prediction target in Predictive Query Language (PQL). KumoRFM samples
the relevant subgraph and returns predictions for the entities you name.

## Core Concepts

- **Universal TFM API.** The shared HTTP contract every SDFM NIM implements. The
  SDK builds requests against this contract so one client serves both models.
- **Model adapter.** A per-model module that shapes a typed request into the
  wire envelope and normalizes the response into a pandas DataFrame.
- **Driver.** A model's heavy client-side runtime. KumoRFM ships one (graph
  building, native neighbor sampling, PQL); TabICL needs none.
- **In-context learning (ICL).** Both models predict from labeled context rows
  in a single forward pass rather than training per dataset.

## Core Components

| Component | Package | Role |
| --- | --- | --- |
| Client SDK | `nvidia-sdfm` | The `SDFMClient`, typed requests, and per-model adapters. Pure Python. |
| KumoRFM driver | `kumorfm` | Graph, samplers, native `kumolib`, and PQL for the relational model. Installed via the `[kumorfm]` extra. |
| Connectors | `sdfm-connectors` | Shared data-source connectors (SQLite, DuckDB, Snowflake, Databricks) used by the client and the driver. |

## Learn More

- [Installation Guide for the NVIDIA SDFM SDK](../get-started/installation.md)
- [Quickstart for the NVIDIA SDFM SDK](../get-started/quickstart.md)
- [NVIDIA SDFM SDK Architecture](architecture.md)
- [NVIDIA SDFM SDK Environment Variables](../reference/environment-variables.md)

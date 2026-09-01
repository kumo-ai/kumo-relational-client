---
title: "NVIDIA Kumo Relational Client Environment Variables"
description: "Reference for the environment variables that configure the NVIDIA Kumo Relational Client, the Kumo Relational driver, and the data-source connectors."
template-library-version: "1.0.0"
---

# NVIDIA Kumo Relational Client Environment Variables

This page lists the environment variables the NVIDIA Kumo Relational Client reads. Most
configuration is passed directly to `RelationalClient` in code; the variables below
cover the Kumo Relational driver's connection defaults, logging, and the data-source
connectors.

## Kumo Relational Driver

The Kumo Relational driver reads these variables when you use the `kumo_relational_engine` model
without passing the values explicitly.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `KUMO_RELATIONAL_API_ENDPOINT` | Conditional | None | URL of the Universal TFM API NIM. Used when a NIM URL is not provided in code. Read when the driver initializes; importing the package never connects. |
| `KUMO_RELATIONAL_API_KEY` | No | None | API key sent to the NIM as `X-API-Key`. NIMs are unauthenticated by contract, so this is only needed when the deployment fronts the NIM with an authenticating gateway. It is refused on a plaintext `http://` endpoint other than localhost. |
| `KUMO_RELATIONAL_LOG` | No | `INFO` | Log level for the Kumo Relational driver, for example `DEBUG`, `INFO`, or `WARNING`. |
| `KUMO_RELATIONAL_DISABLE_SESSIONS` | No | Unset | Set to `1`/`true` to stop a multi-batch prediction from sharing one uploaded context through a NIM session. Each batch then re-uploads the full context, which the progress output reports. Only affects transport cost; predictions are unchanged. Passing `random_seed=None` has the same effect, because unseeded runs re-sample neighborhoods per batch. |

## Explanation Summary (Third-Party LLM)

When you call `predict(..., explain=True)` and the NIM returns structured
attribution without a natural-language summary, the client generates that summary
itself by calling an OpenAI-compatible chat-completions endpoint. The request
carries the predictive query, the returned predictions, the cohort analysis and
the subgraph attribution, which includes the raw cell values of the explained
entity's subgraph. Unless `KUMO_RELATIONAL_EXPLAIN_LLM_BASE_URL` is set, the destination
is OpenAI's `https://api.openai.com/v1/`, a non-NVIDIA service.

Nothing is sent when no API key is discoverable or the `kumo-relational-client[explain]`
extra is not installed. To disable the call while keeping the structured explanation,
pass `explain=dict(skip_summary=True)`.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `KUMO_RELATIONAL_EXPLAIN_LLM_API_KEY` | No | None | API key for the summary endpoint. Setting it is what enables the call, which sends the rows behind a prediction to that endpoint. There is no fallback to any other key. |
| `KUMO_RELATIONAL_EXPLAIN_LLM_BASE_URL` | No | OpenAI (`https://api.openai.com/v1/`) | Base URL of any OpenAI-compatible endpoint, including a self-hosted one. Set this to keep the data inside your own network. |
| `KUMO_RELATIONAL_EXPLAIN_LLM_MODEL` | Conditional | `gpt-4.1-mini-2025-04-14` | Model name. Required when `KUMO_RELATIONAL_EXPLAIN_LLM_BASE_URL` is set. |
| `KUMO_RELATIONAL_EXPLAIN_LLM_TIMEOUT` | No | `20` | Request timeout in seconds. |

## Databricks Connector

When you use the Databricks connector without passing connection arguments, it
falls back to these variables.

| Variable | Required | Description |
| --- | --- | --- |
| `DATABRICKS_SERVER_HOSTNAME` | Yes | Databricks workspace host name. |
| `DATABRICKS_HTTP_PATH` | Yes | SQL warehouse HTTP path. |
| `DATABRICKS_TOKEN` | Yes | Databricks access token. |
| `DATABRICKS_CATALOG` | No | Default catalog. |
| `DATABRICKS_SCHEMA` | No | Default schema. |

### Partner telemetry attribution

SDK-created Databricks clients attribute usage to NVIDIA Kumo Relational
Client. SQL and Model Serving use the same shared SDK release version.
Caller-owned clients are unchanged.

## Snowflake Connector

The Snowflake connector is configured through connection arguments rather than
environment variables. When called with no arguments inside a Snowpark session,
it borrows the active session's connection.

## Related Topics

- [Installation Guide for the NVIDIA Kumo Relational Client](../get-started/installation.md)
- [Quickstart for the NVIDIA Kumo Relational Client](../get-started/quickstart.md)

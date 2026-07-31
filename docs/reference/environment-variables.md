---
title: "NVIDIA SDFM SDK Environment Variables"
description: "Reference for the environment variables that configure the NVIDIA SDFM SDK client, the KumoRFM driver, and the data-source connectors."
template-library-version: "1.0.0"
---

# NVIDIA SDFM SDK Environment Variables

This page lists the environment variables the NVIDIA SDFM SDK reads. Most
configuration is passed directly to `SDFMClient` in code; the variables below
cover the KumoRFM driver's connection defaults, logging, and the data-source
connectors.

## KumoRFM Driver

The KumoRFM driver reads these variables when you use the `kumorfm` model
without passing the values explicitly.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `KUMO_API_ENDPOINT` | Conditional | None | URL of the Universal TFM API NIM. Used when a NIM URL is not provided in code. Read when the driver initializes; importing the package never connects. |
| `KUMO_API_KEY` | No | None | API key sent to the NIM as `X-API-Key`. NIMs are unauthenticated by contract, so this is only needed when the deployment fronts the NIM with an authenticating gateway. It is refused on a plaintext `http://` endpoint other than localhost. |
| `KUMO_LOG` | No | `INFO` | Log level for the KumoRFM driver, for example `DEBUG`, `INFO`, or `WARNING`. |

## Explanation Summary (Third-Party LLM)

When you call `predict(..., explain=True)` and the NIM returns structured
attribution without a natural-language summary, the SDK generates that summary
itself by calling an OpenAI-compatible chat-completions endpoint. The request
carries the predictive query, the returned predictions, the cohort analysis and
the subgraph attribution, which includes the raw cell values of the explained
entity's subgraph. Unless `KUMORFM_EXPLAIN_LLM_BASE_URL` is set, the destination
is OpenAI's `https://api.openai.com/v1/`, a non-NVIDIA service.

Nothing is sent when no API key is discoverable or the `nvidia-sdfm[explain]`
extra is not installed. To disable the call while keeping the structured explanation,
pass `explain=dict(skip_summary=True)`.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `KUMORFM_EXPLAIN_LLM_API_KEY` | No | None | API key for the summary endpoint. **Falls back to `OPENAI_API_KEY`**, so a key exported for another tool enables the call. |
| `OPENAI_API_KEY` | No | None | Fallback API key, read when `KUMORFM_EXPLAIN_LLM_API_KEY` is unset. |
| `KUMORFM_EXPLAIN_LLM_BASE_URL` | No | OpenAI (`https://api.openai.com/v1/`) | Base URL of any OpenAI-compatible endpoint, including a self-hosted one. Set this to keep the data inside your own network. |
| `KUMORFM_EXPLAIN_LLM_MODEL` | Conditional | `gpt-4.1-mini-2025-04-14` | Model name. Required when `KUMORFM_EXPLAIN_LLM_BASE_URL` is set. |
| `KUMORFM_EXPLAIN_LLM_TIMEOUT` | No | `20` | Request timeout in seconds. |

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

## Snowflake Connector

The Snowflake connector is configured through connection arguments rather than
environment variables. When called with no arguments inside a Snowpark session,
it borrows the active session's connection.

## Related Topics

- [Installation Guide for the NVIDIA SDFM SDK](../get-started/installation.md)
- [Quickstart for the NVIDIA SDFM SDK](../get-started/quickstart.md)

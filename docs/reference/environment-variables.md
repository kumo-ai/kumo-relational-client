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
| `KUMO_API_ENDPOINT` | Conditional | None | URL of the Universal TFM API NIM. Used when a NIM URL is not provided in code. If set, the driver initializes against this endpoint automatically at import time (outside test runs). |
| `KUMO_API_KEY` | No | None | API key sent to the NIM. NIMs are unauthenticated by contract, so this is only needed when the deployment fronts the NIM with an authenticating gateway. |
| `KUMO_LOG` | No | `INFO` | Log level for the KumoRFM driver, for example `DEBUG`, `INFO`, or `WARNING`. |

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

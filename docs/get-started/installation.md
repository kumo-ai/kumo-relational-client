---
title: "Installation Guide for the NVIDIA Kumo Relational Client"
description: "Install the NVIDIA Kumo Relational Client and its optional model drivers and data-source connectors from the NVIDIA internal package index."
template-library-version: "1.0.0"
---

# Installation Guide for the NVIDIA Kumo Relational Client

Install the NVIDIA Kumo Relational Client with pip. The base package installs the client and
every lightweight model; optional extras add the Nemotron Relational driver and data-source
connectors so you install only what you need.

## Prerequisites

Before you start, you must complete the following prerequisites:

1. Confirm your environment meets the [NVIDIA Kumo Relational Client Prerequisites](prerequisites.md).

## Installation Methods

```bash
pip install "kumo-relational-client"
```

This installs the client and every lightweight model, including Nemotron Tabular.

## Package Extras

Add extras in brackets to install additional capabilities.

| Command | You get |
| --- | --- |
| `pip install "kumo-relational-client"` | The client plus every lightweight model (Nemotron Tabular). |
| `pip install "kumo-relational-client[relational]"` | Adds the Nemotron Relational driver (native graph sampler and PQL). |
| `pip install "kumo-relational-client[sqlite]"` | Adds the SQLite data-source connector. Also `[duckdb]`, `[snowflake]`, `[databricks]`, `[s3]`. |
| `pip install "kumo-relational-client[databricks-serving]"` | Reaches a Nemotron Relational model served by name on Databricks Model Serving, rather than a NIM addressed by URL. |
| `pip install "kumo-relational-client[all]"` | Nemotron Relational, every data-source connector, and `[databricks-serving]`. |

Two extras are deliberately outside `[all]` and must be named:

| Command | You get |
| --- | --- |
| `pip install "kumo-relational-client[explain]"` | Fills in `Explanation.summary`, which POSTs row data to a third-party LLM endpoint. Kept opt-in for that reason. |
| `pip install "kumo-relational-client[relbench]"` | The RelBench datasets used by `Graph.from_relbench()`. |

For example, to install the client with the Nemotron Relational driver and Snowflake
connector:

```bash
pip install "kumo-relational-client[relational,snowflake]"
```

## Additional Setup

If you use `uv`, add the dependency with:

```bash
uv add "kumo-relational-client[relational]"
```

## Installation Verification

Confirm the client imports and reports its models. This reads the client's own
adapters and does not contact a NIM, so it works before you have one running:

```bash
python -c "from kumo_relational_client import RelationalClient; print(RelationalClient(url='http://localhost:8000').models())"
```

Expected output:

```text
['kumo-relational', 'kumo-tabular']
```

If you installed the `[relational]` extra, confirm the driver's native extension
loads:

```bash
python -c "import nemotron_relational.relationallib as k; print(hasattr(k.NeighborSampler, 'seed'))"
```

Expected output:

```text
True
```

## Troubleshoot the Installation

- **`No matching distribution found for kumo-relational-client`.** Confirm your Python is
  3.10 or newer and that `pip` can reach your configured package index.
- **`No matching distribution found for nemotron_relational`.** The `[relational]` extra has
  prebuilt wheels only for Linux x86_64 (`manylinux_2_28`) on Python 3.10–3.13,
  and no source distribution is published, so there is nothing to fall back to
  on another platform, macOS included. Install the base `kumo-relational-client` and run
  Nemotron Relational from a Linux x86_64 host, or build the driver from source.
- **The native extension fails to import after installing `[relational]`.**
  Reinstall the `nemotron_relational` wheel for your exact Python version, and confirm your
  platform matches the supported wheel matrix.

## Next Steps

- Run your first prediction with the [Quickstart for the NVIDIA Kumo Relational Client](quickstart.md).
- Review configurable settings in the [NVIDIA Kumo Relational Client Environment Variables](../reference/environment-variables.md).

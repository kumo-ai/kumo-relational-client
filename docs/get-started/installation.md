---
title: "Installation Guide for the NVIDIA SDFM SDK"
description: "Install the NVIDIA SDFM SDK and its optional model drivers and data-source connectors from the NVIDIA internal package index."
template-library-version: "1.0.0"
---

# Installation Guide for the NVIDIA SDFM SDK

Install the NVIDIA SDFM SDK with pip. The base package installs the client and
every lightweight model; optional extras add the NemotronRelational driver and data-source
connectors so you install only what you need.

## Prerequisites

Before you start, you must complete the following prerequisites:

1. Confirm your environment meets the [NVIDIA SDFM SDK Prerequisites](prerequisites.md).

## Installation Methods

```bash
pip install "nemotron-predict-client"
```

This installs the client and every lightweight model, including TabICL.

## Package Extras

Add extras in brackets to install additional capabilities.

| Command | You get |
| --- | --- |
| `pip install "nemotron-predict-client"` | The client plus every lightweight model (TabICL). |
| `pip install "nemotron-predict-client[nemotron_relational]"` | Adds the NemotronRelational driver (native graph sampler and PQL). |
| `pip install "nemotron-predict-client[sqlite]"` | Adds the SQLite data-source connector. Also `[duckdb]`, `[snowflake]`, `[databricks]`, `[s3]`. |
| `pip install "nemotron-predict-client[databricks-serving]"` | Reaches a NemotronRelational model served by name on Databricks Model Serving, rather than a NIM addressed by URL. |
| `pip install "nemotron-predict-client[all]"` | NemotronRelational, every data-source connector, and `[databricks-serving]`. |

Two extras are deliberately outside `[all]` and must be named:

| Command | You get |
| --- | --- |
| `pip install "nemotron-predict-client[explain]"` | Fills in `Explanation.summary`, which POSTs row data to a third-party LLM endpoint. Kept opt-in for that reason. |
| `pip install "nemotron-predict-client[relbench]"` | The RelBench datasets used by `Graph.from_relbench()`. |

For example, to install the client with the NemotronRelational driver and Snowflake
connector:

```bash
pip install "nemotron-predict-client[relational,snowflake]"
```

## Additional Setup

If you use `uv`, add the dependency with:

```bash
uv add "nemotron-predict-client[nemotron_relational]"
```

## Installation Verification

Confirm the client imports and reports its models. This reads the client's own
adapters and does not contact a NIM, so it works before you have one running:

```bash
python -c "from nemotron_predict import PredictClient; print(PredictClient(url='http://localhost:8000').models())"
```

Expected output:

```text
['nemotron-relational-v1', 'tabicl']
```

If you installed the `[nemotron_relational]` extra, confirm the driver's native extension
loads:

```bash
python -c "import nemotron_relational.relationallib as k; print(hasattr(k.NeighborSampler, 'seed'))"
```

Expected output:

```text
True
```

## Troubleshoot the Installation

- **`No matching distribution found for nemotron-predict-client`.** Confirm your Python is
  3.10 or newer and that `pip` can reach your configured package index.
- **`No matching distribution found for nemotron_relational`.** The `[nemotron_relational]` extra has
  prebuilt wheels only for Linux x86_64 (`manylinux_2_28`) on Python 3.10–3.13,
  and no source distribution is published, so there is nothing to fall back to
  on another platform — macOS included. Install the base `nemotron-predict-client` and run
  NemotronRelational from a Linux x86_64 host, or build the driver from source.
- **The native extension fails to import after installing `[nemotron_relational]`.**
  Reinstall the `nemotron_relational` wheel for your exact Python version, and confirm your
  platform matches the supported wheel matrix.

## Next Steps

- Run your first prediction with the [Quickstart for the NVIDIA SDFM SDK](quickstart.md).
- Review configurable settings in the [NVIDIA SDFM SDK Environment Variables](../reference/environment-variables.md).

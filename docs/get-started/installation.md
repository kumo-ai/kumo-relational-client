---
title: "Installation Guide for the NVIDIA Kumo Relational Client"
description: "Install the NVIDIA Kumo Relational Client and its optional model drivers and data-source connectors with pip."
template-library-version: "1.0.1"
---

# Installation Guide for the NVIDIA Kumo Relational Client

Install the NVIDIA Kumo Relational Client with pip. The base package installs the client and
every lightweight model; optional extras add the Kumo Relational driver and data-source
connectors so you install only what you need.

## Prerequisites

Before you start, you must complete the following prerequisites:

1. Confirm your environment meets the [NVIDIA Kumo Relational Client Prerequisites](prerequisites.md).

## Installation Methods

```bash
pip install "kumo-relational-client"
```

This installs the client on its own.

## Package Extras

Add extras in brackets to install additional capabilities.

| Command | You get |
| --- | --- |
| `pip install "kumo-relational-client"` | The client on its own. |
| `pip install "kumo-relational-client[relational]"` | Adds the Kumo Relational driver (native graph sampler and PQL). |
| `pip install "kumo-relational-client[sqlite]"` | Adds the SQLite data-source connector. Also `[duckdb]`, `[snowflake]`, `[databricks]`, `[postgres]`, `[s3]`. |
| `pip install "kumo-relational-client[databricks-serving]"` | Reaches a Kumo Relational model served by name on Databricks Model Serving, rather than a NIM addressed by URL. |
| `pip install "kumo-relational-client[all]"` | Kumo Relational, every data-source connector, and `[databricks-serving]`. |

Two extras are deliberately outside `[all]` and must be named:

| Command | You get |
| --- | --- |
| `pip install "kumo-relational-client[explain]"` | Fills in `Explanation.summary`, which POSTs row data to a third-party LLM endpoint. Kept opt-in for that reason. |
| `pip install "kumo-relational-client[relbench]"` | The RelBench datasets used by `Graph.from_relbench()`. |

For example, to install the client with the Kumo Relational driver and Snowflake
connector:

```bash
pip install "kumo-relational-client[relational,snowflake]"
```

For a Databricks serverless Python 3.12 environment, pin the coordinated SDK
release and require binary artifacts so the native engine is never compiled on
the serverless host:

```bash
python -m pip install --only-binary=:all: \
  "kumo-relational-client[databricks,databricks-serving]==1.0.1" \
  "kumo-relational-engine==1.0.1" \
  "kumo-connectors==1.0.1"
```

Use `[databricks]` for a Databricks SQL Warehouse and `[postgres]` for a
direct PostgreSQL connection, including a direct connection to Lakebase.

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
['kumo-relational']
```

If you installed the `[relational]` extra, confirm the driver's native extension
loads:

```bash
python -c "import kumo_relational_engine.relationallib as k; print(hasattr(k.NeighborSampler, 'seed'))"
```

Expected output:

```text
True
```

## Troubleshoot the Installation

- **`No matching distribution found for kumo-relational-client`.** Confirm your Python is
  3.10 or newer and that `pip` can reach your configured package index.
- **`No matching distribution found for kumo_relational_engine`.** The `[relational]` extra
  ships prebuilt wheels for Python 3.10–3.13 on Linux x86_64 and ARM64
  (`manylinux_2_28`), macOS Intel and Apple silicon, and Windows x64. Windows
  on ARM64 has none, and no source distribution is published, so there is
  nothing to fall back to there. Use a supported host, or build the driver
  from source.
- **A `pyarrow` build failure on Windows ARM64.** `pyarrow` publishes no
  Windows ARM64 wheel, so `pip` falls back to its source distribution and tries
  to compile Apache Arrow's C++ library. The connectors require `pyarrow`,
  which makes this the first thing that fails on that platform, before any
  Kumo package is reached. There is no workaround short of a supported host.
- **The native extension fails to import after installing `[relational]`.**
  Reinstall the `kumo_relational_engine` wheel for your exact Python version, and confirm your
  platform matches the supported wheel matrix.

## Next Steps

- Run your first prediction with the [Quickstart for the NVIDIA Kumo Relational Client](quickstart.md).
- Review configurable settings in the [NVIDIA Kumo Relational Client Environment Variables](../reference/environment-variables.md).

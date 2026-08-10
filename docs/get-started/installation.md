---
title: "Installation Guide for the NVIDIA SDFM SDK"
description: "Install the NVIDIA SDFM SDK and its optional model drivers and data-source connectors from the NVIDIA internal package index."
template-library-version: "1.0.0"
---

# Installation Guide for the NVIDIA SDFM SDK

Install the NVIDIA SDFM SDK with pip. The base package installs the client and
every lightweight model; optional extras add the KumoRFM driver and data-source
connectors so you install only what you need.

## Prerequisites

Before you start, you must complete the following prerequisites:

1. Confirm your environment meets the [NVIDIA SDFM SDK Prerequisites](prerequisites.md).

## Installation Methods

```bash
pip install "nvidia-sdfm"
```

This installs the client and every lightweight model, including TabICL.

## Package Extras

Add extras in brackets to install additional capabilities.

| Command | You get |
| --- | --- |
| `pip install "nvidia-sdfm"` | The client plus every lightweight model (TabICL). |
| `pip install "nvidia-sdfm[kumorfm]"` | Adds the KumoRFM driver (native graph sampler and PQL). |
| `pip install "nvidia-sdfm[sqlite]"` | Adds the SQLite data-source connector. Also `[duckdb]`, `[snowflake]`, `[databricks]`, `[s3]`. |
| `pip install "nvidia-sdfm[databricks-serving]"` | Reaches a KumoRFM model served by name on Databricks Model Serving, rather than a NIM addressed by URL. |
| `pip install "nvidia-sdfm[all]"` | KumoRFM, every data-source connector, and `[databricks-serving]`. |

Two extras are deliberately outside `[all]` and must be named:

| Command | You get |
| --- | --- |
| `pip install "nvidia-sdfm[explain]"` | Fills in `Explanation.summary`, which POSTs row data to a third-party LLM endpoint. Kept opt-in for that reason. |
| `pip install "nvidia-sdfm[relbench]"` | The RelBench datasets used by `Graph.from_relbench()`. |

For example, to install the client with the KumoRFM driver and Snowflake
connector:

```bash
pip install "nvidia-sdfm[kumorfm,snowflake]"
```

## Additional Setup

If you use `uv`, add the dependency with:

```bash
uv add "nvidia-sdfm[kumorfm]"
```

## Installation Verification

Confirm the client imports and reports its models. This reads the client's own
adapters and does not contact a NIM, so it works before you have one running:

```bash
python -c "from nvidia_sdfm import SDFMClient; print(SDFMClient(url='http://localhost:8000').models())"
```

Expected output:

```text
['kumo-rfm', 'tabicl']
```

If you installed the `[kumorfm]` extra, confirm the driver's native extension
loads:

```bash
python -c "import kumorfm.kumolib as k; print(hasattr(k.NeighborSampler, 'seed'))"
```

Expected output:

```text
True
```

## Troubleshoot the Installation

- **`No matching distribution found for nvidia-sdfm`.** Confirm your Python is
  3.10 or newer and that `pip` can reach your configured package index.
- **`No matching distribution found for kumorfm`.** The `[kumorfm]` extra has
  prebuilt wheels only for Linux x86_64 (`manylinux_2_28`) on Python 3.10–3.13,
  and no source distribution is published, so there is nothing to fall back to
  on another platform — macOS included. Install the base `nvidia-sdfm` and run
  KumoRFM from a Linux x86_64 host, or build the driver from source.
- **The native extension fails to import after installing `[kumorfm]`.**
  Reinstall the `kumorfm` wheel for your exact Python version, and confirm your
  platform matches the supported wheel matrix.

## Next Steps

- Run your first prediction with the [Quickstart for the NVIDIA SDFM SDK](quickstart.md).
- Review configurable settings in the [NVIDIA SDFM SDK Environment Variables](../reference/environment-variables.md).

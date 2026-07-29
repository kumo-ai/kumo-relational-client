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
2. Connect to the NVIDIA network so the internal package index is reachable.

## Installation Methods

Install from the NVIDIA internal PyPI index. The `--index-url` resolves the SDK
packages and their dependencies from one place:

```bash
pip install "nvidia-sdfm" \
    --index-url https://pypi.org/simple
```

This installs the client and every lightweight model, including TabICL.

## Package Extras

Add extras in brackets to install additional capabilities. Extras resolve from
the same index.

| Command | You get |
| --- | --- |
| `pip install "nvidia-sdfm"` | The client plus every lightweight model (TabICL). |
| `pip install "nvidia-sdfm[kumorfm]"` | Adds the KumoRFM driver (native graph sampler and PQL). |
| `pip install "nvidia-sdfm[sqlite]"` | Adds the SQLite data-source connector. Also `[duckdb]`, `[snowflake]`, `[databricks]`. |
| `pip install "nvidia-sdfm[all]"` | Everything. |

For example, to install the client with the KumoRFM driver and Snowflake
connector:

```bash
pip install "nvidia-sdfm[kumorfm,snowflake]" \
    --index-url https://pypi.org/simple
```

## Additional Setup

If you use `uv`, add the index to your project configuration:

```toml
[[tool.uv.index]]
name = "nv-shared"
url = "https://pypi.org/simple"
```

Then add the dependency:

```bash
uv add "nvidia-sdfm[kumorfm]"
```

## Installation Verification

Confirm the client imports and reports its models:

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

- **`No matching distribution found for nvidia-sdfm`.** Confirm you passed the
  `--index-url` for the NVIDIA internal index and that you are on the NVIDIA
  network.
- **`No matching distribution found for kumorfm`.** The `[kumorfm]` extra has
  prebuilt wheels only for Linux x86_64 and macOS arm64 on Python 3.10–3.12.
  Use a supported platform and interpreter, or install the base `nvidia-sdfm`
  without the extra.
- **The native extension fails to import after installing `[kumorfm]`.**
  Reinstall the `kumorfm` wheel for your exact Python version, and confirm your
  platform matches the supported wheel matrix.

## Next Steps

- Run your first prediction with the [Quickstart for the NVIDIA SDFM SDK](quickstart.md).
- Review configurable settings in the [NVIDIA SDFM SDK Environment Variables](../reference/environment-variables.md).

---
title: "NVIDIA SDFM SDK Prerequisites"
description: "Hardware, software, and network prerequisites for installing and using the NVIDIA SDFM SDK."
template-library-version: "1.0.0"
---

# NVIDIA SDFM SDK Prerequisites

Before you install the NVIDIA SDFM SDK, make sure your environment meets the
following requirements.

## Hardware

### Compute

The base client (`nvidia-sdfm`) is pure Python and has no special hardware
requirements. The KumoRFM driver performs graph sampling on the CPU of the
machine that runs the client; the model itself runs on the NIM's GPU, not on
your client machine.

## Software

### Operating System

The KumoRFM driver ships prebuilt binary wheels for the following platforms:

| Platform | Supported |
| --- | --- |
| Linux x86_64 (glibc 2.28+, `manylinux_2_28`) | Yes |
| Everything else, including macOS | No — build from source |

No source distribution is published either, so `pip install
"nvidia-sdfm[kumorfm]"` resolves on Linux x86_64 only. On any other platform,
install the base `nvidia-sdfm` — it is pure Python, as are the connectors, and
installs on anything running Python 3.10 or later — and run KumoRFM from a Linux
x86_64 host.

### Runtime Dependencies

| Dependency | Version |
| --- | --- |
| Python | 3.10, 3.11, or 3.12 |
| pip | 23.0 or later recommended |

Data-source connectors (SQLite, DuckDB, Snowflake, Databricks) are optional and
install only when you request the matching extra.

## Verified Configurations

- Python 3.12 on Linux x86_64, installing `nvidia-sdfm[kumorfm]`.
- Python 3.12 on macOS arm64, installing the base `nvidia-sdfm`.

## Network Access

### Package Index Access

While the SDK is distributed internally, installation resolves from NVIDIA's
internal PyPI index on `the internal package index`. You must be on the NVIDIA network to
install.

### NIM Endpoint Access

At runtime, the client needs network access to a Universal TFM API NIM. Set the
endpoint when you create the client, or through the `KUMO_API_ENDPOINT`
environment variable for the KumoRFM path.

## Pre-Installation Checklist

Before you continue to installation, confirm the following:

1. Verify that Python 3.10, 3.11, or 3.12 is installed: `python --version`.
2. Confirm you are on the NVIDIA network so the internal index is reachable.
3. Identify the URL of the NIM you will connect to.

## Troubleshoot Prerequisites

- **`No matching distribution found for kumorfm`.** Your platform or Python
  version is outside the wheel matrix (Linux x86_64, Python 3.10–3.12), and no
  source distribution is published to fall back to. Install the base
  `nvidia-sdfm` without the `[kumorfm]` extra, or use a supported host and
  interpreter.
- **Cannot reach the package index.** Confirm you are connected to the NVIDIA
  network, then retry the install.

## Next Steps

- Continue to the [Installation Guide for the NVIDIA SDFM SDK](installation.md).

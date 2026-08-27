---
title: "NVIDIA Kumo Relational Client Prerequisites"
description: "Hardware, software, and network prerequisites for installing and using the NVIDIA Kumo Relational Client."
template-library-version: "1.0.0"
---

# NVIDIA Kumo Relational Client Prerequisites

Before you install the NVIDIA Kumo Relational Client, make sure your environment meets the
following requirements.

## Hardware

### Compute

The base client (`kumo-relational-client`) is pure Python and has no special hardware
requirements. The Kumo Relational driver performs graph sampling on the CPU of the
machine that runs the client; the model itself runs on the NIM's GPU, not on
your client machine.

## Software

### Operating System

The Kumo Relational driver ships prebuilt binary wheels for the following platforms:

| Platform | Supported |
| --- | --- |
| Linux x86_64 (glibc 2.28+, `manylinux_2_28`) | Yes |
| Everything else, including macOS | No, build from source |

No source distribution is published either, so `pip install
"kumo-relational-client[relational]"` resolves on Linux x86_64 only. On any other platform,
install the base `kumo-relational-client` and run Kumo Relational from a Linux
x86_64 host. The base client is pure Python, as are the connectors, so it installs
on anything running Python 3.10 or later.

### Runtime Dependencies

| Dependency | Version |
| --- | --- |
| Python | 3.10, 3.11, 3.12, or 3.13 |
| pip | 23.0 or later recommended |

Data-source connectors (SQLite, DuckDB, Snowflake, Databricks) are optional and
install only when you request the matching extra.

## Verified Configurations

- Python 3.12 on Linux x86_64, installing `kumo-relational-client[relational]`.
- Python 3.12 on macOS arm64, installing the base `kumo-relational-client`.

## Network Access

### NIM Endpoint Access

At runtime, the client needs network access to a Universal TFM API NIM. Set the
endpoint when you create the client, or through the `KUMO_RELATIONAL_API_ENDPOINT`
environment variable for the Kumo Relational path.

## Pre-Installation Checklist

Before you continue to installation, confirm the following:

1. Verify that Python 3.10, 3.11, 3.12, or 3.13 is installed: `python --version`.
2. Identify the URL of the NIM you will connect to.

## Troubleshoot Prerequisites

- **`No matching distribution found for kumo_relational_engine`.** Your platform or Python
  version is outside the wheel matrix (Linux x86_64, Python 3.10–3.13), and no
  source distribution is published to fall back to. Install the base
  `kumo-relational-client` without the `[relational]` extra, or use a supported host and
  interpreter.
- **Cannot reach the package index.** Confirm you are connected to the NVIDIA
  network, then retry the install.

## Next Steps

- Continue to the [Installation Guide for the NVIDIA Kumo Relational Client](installation.md).

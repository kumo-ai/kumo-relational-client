---
title: "NVIDIA Kumo Relational Client Prerequisites"
description: "Hardware, software, and network prerequisites for installing and using the NVIDIA Kumo Relational Client."
template-library-version: "1.0.1"
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
| Linux ARM64 (glibc 2.28+, `manylinux_2_28`) | Yes |
| macOS 11+ Intel (x86_64) | Yes |
| macOS 11+ Apple silicon (arm64) | Yes |
| Windows x64 | Yes |
| Windows ARM64 | No wheel; builds from a source checkout |

Every supported platform carries a wheel for each of CPython 3.10, 3.11, 3.12
and 3.13. No source distribution is published, so on Windows ARM64 `pip install
"kumo-relational-client[relational]"` has nothing to resolve; install the base
`kumo-relational-client` there and run Kumo Relational from a supported host.
The base client is pure Python, as are the connectors, so it installs on
anything running Python 3.10 or later.

### Runtime Dependencies

| Dependency | Version |
| --- | --- |
| Python | 3.10, 3.11, 3.12, or 3.13 |
| pip | 23.0 or later recommended |

Data-source connectors (SQLite, DuckDB, Snowflake, Databricks) are optional and
install only when you request the matching extra.

## Verified Configurations

Each release installs `kumo-relational-client[relational]` and imports the
native driver on every platform in the table above, on each of Python 3.10,
3.11, 3.12 and 3.13. A failure on any one of them blocks the release.

Verified in addition:

- Python 3.12 on Linux ARM64, installing
  `kumo-relational-client[databricks,databricks-serving]` for Databricks
  serverless.

## Network Access

### NIM Endpoint Access

At runtime, the client needs network access to a Universal TFM API NIM. Set the
endpoint when you create the client, or through the `KUMO_RELATIONAL_API_ENDPOINT`
environment variable for the Kumo Relational path.

You can use a hosted endpoint from the [NVIDIA API
catalog](https://build.nvidia.com), which needs an API key and nothing deployed,
or pull the NIM container from the [NGC
catalog](https://catalog.ngc.nvidia.com) and run it on your own GPU host when
the data cannot leave your environment.

## Pre-Installation Checklist

Before you continue to installation, confirm the following:

1. Verify that Python 3.10, 3.11, 3.12, or 3.13 is installed: `python --version`.
2. Identify the URL of the NIM you will connect to.

## Troubleshoot Prerequisites

- **`No matching distribution found for kumo_relational_engine`.** Your platform or Python
  version is outside the wheel matrix in the table above, and no source
  distribution is published to fall back to. Install the base
  `kumo-relational-client` without the `[relational]` extra, or use a supported
  host and interpreter.
- **Cannot reach the package index.** Confirm `pip` can reach PyPI, including
  through any proxy or mirror your environment requires, then retry.

## Next Steps

- Continue to the [Installation Guide for the NVIDIA Kumo Relational Client](installation.md).

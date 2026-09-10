# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    import kumo_relational_engine.relationallib  # noqa: F401
except Exception as e:
    import platform

    _msg = f"""RFM is not supported in your environment.

💻 Your Environment:
Python version: {platform.python_version()}
Operating system: {platform.system()}
CPU architecture: {platform.machine()}
glibc version: {platform.libc_ver()[1]}

✅ Prebuilt wheels are published for Python 3.10, 3.11, 3.12 and 3.13 on:
* Linux (x86_64), glibc >=2.28
* Linux (arm64), glibc >=2.28
* macOS 11+ (x86_64 and arm64)
* Windows (x86_64)

No source distribution is published, so `pip install kumo-relational-engine` resolves on that
matrix alone.

❌ Not supported:
* Python versions below 3.10, or 3.14 and later
* Windows (arm64). The extension does build from a source checkout there, but
  that path is not tested and not supported.
* macOS below 11
* glibc versions: <2.28

Please create a feature request at 'https://github.com/kumo-ai/kumo-relational-client'."""

    raise RuntimeError(_msg) from e

from .table import LocalTable
from .graph_store import LocalGraphStore
from .sampler import LocalSampler

__all__ = [
    'LocalGraphStore',
    'LocalSampler',
    'LocalTable',
]

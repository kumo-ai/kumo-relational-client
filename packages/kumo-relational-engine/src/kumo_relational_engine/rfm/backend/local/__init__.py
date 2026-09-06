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

✅ Prebuilt wheels are published for:
* Linux (x86_64), glibc >=2.28: Python 3.10, 3.11, 3.12, 3.13
* Linux (arm64), glibc >=2.28: Python 3.12

No source distribution is published, so `pip install kumo-relational-engine` resolves on that
matrix alone. The extension also builds from a source checkout on macOS
(arm64) and Windows (x86_64).

❌ Not supported:
* Python versions below 3.10, or 3.14 and later
* Linux (arm64) on Python versions other than 3.12
* macOS (x86_64), Windows (arm64)
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

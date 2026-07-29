# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    import kumorfm.kumolib  # noqa: F401
except Exception as e:
    import platform

    _msg = f"""RFM is not supported in your environment.

💻 Your Environment:
Python version: {platform.python_version()}
Operating system: {platform.system()}
CPU architecture: {platform.machine()}
glibc version: {platform.libc_ver()[1]}

✅ Supported Environments:
* Python versions: 3.10, 3.11, 3.12, 3.13, 3.14
* Operating systems and CPU architectures:
  * Linux (x86_64)
  * macOS (arm64)
  * Windows (x86_64)
* glibc versions: >=2.28

❌ Unsupported Environments:
* Python versions: 3.8, 3.9, 3.15
* Operating systems and CPU architectures:
  * Linux (arm64)
  * macOS (x86_64)
  * Windows (arm64)
* glibc versions: <2.28

Please create a feature request at 'https://github.com/NVIDIA/nvidia-sdfm-sdk'."""

    raise RuntimeError(_msg) from e

from .table import LocalTable
from .graph_store import LocalGraphStore
from .sampler import LocalSampler

__all__ = [
    'LocalTable',
    'LocalGraphStore',
    'LocalSampler',
]

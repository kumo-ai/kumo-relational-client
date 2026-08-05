# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sdfm_connectors.backends.duckdb import Connection, connect


from .table import DuckDBTable  # noqa: E402
from .sampler import DuckDBSampler  # noqa: E402

__all__ = [
    'connect',
    'Connection',
    'DuckDBTable',
    'DuckDBSampler',
]

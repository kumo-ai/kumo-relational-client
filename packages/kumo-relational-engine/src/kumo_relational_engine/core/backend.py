# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_relational_engine.api.common import StrEnum


class DataBackend(StrEnum):
    LOCAL = 'local'
    SQLITE = 'sqlite'
    DUCKDB = 'duckdb'
    SNOWFLAKE = 'snowflake'
    DATABRICKS = 'databricks'

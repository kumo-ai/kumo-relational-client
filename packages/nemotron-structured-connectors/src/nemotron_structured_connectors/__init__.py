# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemotron_structured_connectors.backends import connect
from nemotron_structured_connectors.reader import read, read_table
from nemotron_structured_connectors.sql import (
    ConnectorError,
    MissingBackendError,
    quote_ident,
    resolve_sql,
)

__version__ = '1.0.0'

__all__ = [
    'ConnectorError',
    'MissingBackendError',
    '__version__',
    'connect',
    'quote_ident',
    'read',
    'read_table',
    'resolve_sql',
]

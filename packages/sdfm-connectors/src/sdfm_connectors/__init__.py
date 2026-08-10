# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sdfm_connectors.backends import connect
from sdfm_connectors.reader import read, read_table
from sdfm_connectors.sql import (
    ConnectorError,
    MissingBackendError,
    quote_ident,
    resolve_sql,
)

__version__ = '0.4.0'

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

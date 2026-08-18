# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pandas as pd
from nemotron_structured_connectors import read as _read
from nemotron_structured_connectors.sql import (
    ConnectorError,
    MissingBackendError,
)

from nemotron_structured.errors import MissingExtraError, StructuredError


def read(source: str, **kwargs: Any) -> pd.DataFrame:
    try:
        return _read(source, **kwargs)
    except MissingBackendError as error:
        raise MissingExtraError(error.backend, error.driver) from error
    except ConnectorError as error:
        raise StructuredError(
            error.message,
            code=error.code,
            details=error.details,
        ) from error
    except ImportError as error:
        raise StructuredError(
            f'the {source!r} connector is installed but failed to load its '
            f'driver: {error}',
            code='DRIVER_LOAD_FAILED',
        ) from error

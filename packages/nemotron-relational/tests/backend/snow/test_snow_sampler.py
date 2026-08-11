# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import pytest

try:
    from nemotron_relational.rfm.backend.snow import Connection
    from nemotron_relational.rfm.backend.snow.sampler import paramstyle
except ImportError:
    pytest.skip("'snowflake' extension not installed", allow_module_level=True)


class _FakeConnection:
    def __init__(self) -> None:
        self._paramstyle = 'pyformat'


def test_paramstyle_restored_on_error() -> None:
    # Regression test for `sampler-snowflake-paramstyle-not-restored.md`: the
    # connection is usually owned by the caller, so a failed query must not
    # leave it stuck in 'qmark' mode.
    connection = cast(Connection, cast(Any, _FakeConnection()))

    with pytest.raises(RuntimeError), paramstyle(connection):
        assert connection._paramstyle == 'qmark'
        raise RuntimeError('query failed')

    assert connection._paramstyle == 'pyformat'


def test_paramstyle_restored_on_success() -> None:
    connection = cast(Connection, cast(Any, _FakeConnection()))

    with paramstyle(connection):
        assert connection._paramstyle == 'qmark'

    assert connection._paramstyle == 'pyformat'

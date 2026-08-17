# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routing a serving target to the platform that serves it.

``test_relational_serving_branch.py`` covers the Databricks branch end to end.
What is covered here is the table it dispatches through, which is the part a
third platform will touch: that each platform is described in one place, and
that a target which names no platform is refused rather than served by whichever
one happens to be listed first.
"""

from __future__ import annotations

import pytest

from nemotron_predict.adapters.relational import (
    _SERVING_PLATFORMS,
    _init_serving,
)
from nemotron_predict.core.serving import (
    DatabricksServingTarget,
    ServingTarget,
    SnowflakeServingTarget,
)
from nemotron_predict.errors import MissingExtraError, PredictError


class _Engine:
    """Records which initializer ran and with what."""

    _CLIENT_TOKEN = object()

    def __init__(self) -> None:
        self.calls: dict[str, dict] = {}

    def init_databricks_serving(self, endpoint, **kwargs):
        self.calls['databricks'] = {'endpoint': endpoint, **kwargs}

    def init_snowflake_serving(self, endpoint, **kwargs):
        self.calls['snowflake'] = {'endpoint': endpoint, **kwargs}


def test_the_base_target_names_no_platform() -> None:
    """A subclass that forgets ``kind`` must not inherit a working default."""
    assert ServingTarget.kind == ''


@pytest.mark.parametrize(
    ('target_type', 'expected'),
    [
        (DatabricksServingTarget, 'databricks'),
        (SnowflakeServingTarget, 'snowflake'),
    ],
)
def test_each_target_routes_to_its_own_platform(target_type, expected) -> None:
    engine = _Engine()
    _init_serving(engine, target_type('a-name', 'a-client'))
    assert list(engine.calls) == [expected]
    assert engine.calls[expected]['endpoint'] == 'a-name'


def test_the_client_is_passed_under_the_name_each_platform_uses() -> None:
    """The two engines disagree on the keyword, which is why the table holds a
    call rather than a keyword name.
    """
    engine = _Engine()
    _init_serving(engine, DatabricksServingTarget('a-name', 'WS'))
    _init_serving(engine, SnowflakeServingTarget('DB.SC.SVC', 'SESSION'))
    assert engine.calls['databricks']['workspace_client'] == 'WS'
    assert engine.calls['snowflake']['session'] == 'SESSION'


def test_a_target_naming_no_platform_is_refused() -> None:
    with pytest.raises(PredictError) as caught:
        _init_serving(_Engine(), ServingTarget('a-name'))
    assert caught.value.code == 'INVALID_CONFIGURATION'


def test_the_refusal_names_the_unknown_platform() -> None:
    class _Elsewhere(ServingTarget):
        kind = 'bigquery'

    with pytest.raises(PredictError) as caught:
        _init_serving(_Engine(), _Elsewhere('a-name'))
    assert 'bigquery' in str(caught.value)


@pytest.mark.parametrize('kind', sorted(_SERVING_PLATFORMS))
def test_a_missing_dependency_names_that_platforms_own_extra(kind) -> None:
    """The failure this guards is a platform routed correctly but reported
    under another platform's extra, which sends the user to install the wrong
    package.
    """
    platform = _SERVING_PLATFORMS[kind]

    class _Bare:
        _CLIENT_TOKEN = object()

        def __getattr__(self, name):
            def raise_import_error(*args, **kwargs):
                raise ImportError('no vendor client here')

            return raise_import_error

    target = ServingTarget('a-name')
    object.__setattr__(target, 'kind', kind)
    with pytest.raises(MissingExtraError) as caught:
        _init_serving(_Bare(), target)
    assert platform.extra in str(caught.value)
    assert platform.dependency in str(caught.value)


def test_every_platform_is_described_exactly_once() -> None:
    extras = [p.extra for p in _SERVING_PLATFORMS.values()]
    dependencies = [p.dependency for p in _SERVING_PLATFORMS.values()]
    assert len(set(extras)) == len(extras)
    assert len(set(dependencies)) == len(dependencies)


def test_every_shipped_target_has_a_platform_entry() -> None:
    """A target class without a table entry is unreachable, and the failure
    would only appear the first time someone constructed one.
    """
    for target_type in (DatabricksServingTarget, SnowflakeServingTarget):
        assert target_type.kind in _SERVING_PLATFORMS

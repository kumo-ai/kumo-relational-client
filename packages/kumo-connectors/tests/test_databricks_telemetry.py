# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from kumo_connectors._databricks_telemetry import (
    DatabricksTelemetryVersionError,
    DatabricksUserAgentEntryError,
    databricks_product_version,
    databricks_user_agent,
)
from kumo_connectors._version import __version__


@pytest.mark.parametrize(
    'sdk_version,expected',
    [
        ('1.2.3', '1.2.3'),
        ('1.2.3a1', '1.2.3-alpha.1'),
        ('1.2.3b2', '1.2.3-beta.2'),
        ('1.2.3rc4', '1.2.3-rc.4'),
        ('1.2.3.dev5', '1.2.3-dev.5'),
        ('1.2.3rc4.dev5', '1.2.3-rc.4.dev.5'),
        ('1.2.3.post6', '1.2.3+post.6'),
    ],
)
def test_product_version_is_valid_semver(sdk_version, expected):
    assert databricks_product_version(sdk_version) == expected


@pytest.mark.parametrize(
    'sdk_version',
    [
        '1.2',
        'v1.2.3',
        '1!1.2.3',
        '01.2.3',
        '1.2.3+local',
    ],
)
def test_unsupported_sdk_version_is_not_misattributed(sdk_version):
    with pytest.raises(
        DatabricksTelemetryVersionError, match='cannot be represented'
    ):
        databricks_product_version(sdk_version)


def test_user_agent_identifies_the_sdk_not_a_model_release():
    assert databricks_user_agent('1.2.3') == (
        'nvidia_kumo-relational-client/1.2.3'
    )


def test_user_agent_preserves_an_additional_caller_entry():
    assert databricks_user_agent('1.2.3', 'customer_product/4.5.6') == (
        'nvidia_kumo-relational-client/1.2.3 customer_product/4.5.6'
    )


def test_user_agent_preserves_visible_spaces_in_a_caller_entry():
    assert databricks_user_agent(
        '1.2.3', 'customer_product/4.5.6 another_product/7.8.9'
    ).endswith('customer_product/4.5.6 another_product/7.8.9')


@pytest.mark.parametrize(
    'entry', ['', '   ', 'product/1\nInjected: yes', 'x\x7f']
)
def test_user_agent_rejects_empty_or_control_character_entries(entry):
    with pytest.raises(DatabricksUserAgentEntryError):
        databricks_user_agent('1.2.3', entry)


def test_user_agent_rejects_a_non_string_entry():
    with pytest.raises(TypeError, match='must be a string'):
        databricks_user_agent('1.2.3', 123)


def test_the_declared_sdk_version_is_accepted():
    assert databricks_product_version(__version__)

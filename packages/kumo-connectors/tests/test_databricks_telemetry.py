# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from kumo_connectors import __version__
from kumo_connectors._databricks_telemetry import (
    databricks_product_version,
    databricks_user_agent,
)


@pytest.mark.parametrize(
    'sdk_version,expected',
    [
        ('1.2.3', '1.2.3'),
        ('1.2.3a1', '1.2.3-alpha.1'),
        ('1.2.3b2', '1.2.3-beta.2'),
        ('1.2.3rc4', '1.2.3-rc.4'),
    ],
)
def test_product_version_is_valid_semver(sdk_version, expected):
    assert databricks_product_version(sdk_version) == expected


@pytest.mark.parametrize(
    'sdk_version',
    [
        '1.2',
        'v1.2.3',
        '1.2.3.dev1',
        '1.2.3.post1',
        '1!1.2.3',
        '01.2.3',
    ],
)
def test_unsupported_sdk_version_is_not_misattributed(sdk_version):
    with pytest.raises(ValueError, match='cannot be represented'):
        databricks_product_version(sdk_version)


def test_user_agent_identifies_the_sdk_not_a_model_release():
    assert databricks_user_agent('1.2.3') == (
        'nvidia_kumo-relational-client/1.2.3'
    )


def test_user_agent_preserves_an_additional_caller_entry():
    assert databricks_user_agent('1.2.3', 'customer_product/4.5.6') == (
        'nvidia_kumo-relational-client/1.2.3 customer_product/4.5.6'
    )


def test_the_declared_sdk_version_is_accepted():
    assert databricks_product_version(__version__)

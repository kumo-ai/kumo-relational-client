# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from kumo_relational_engine._version import __version__
from kumo_relational_engine.client._databricks_telemetry import (
    DATABRICKS_PRODUCT,
    databricks_product_version,
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
    ['1.2', 'v1.2.3', '1.2.3.dev1', '1.2.3.post1', '1!1.2.3'],
)
def test_unsupported_sdk_version_is_not_misattributed(sdk_version):
    with pytest.raises(ValueError, match='cannot be represented'):
        databricks_product_version(sdk_version)


def test_the_declared_engine_version_is_accepted():
    assert databricks_product_version(__version__)


def test_product_identifies_the_sdk_not_the_relational_model():
    assert DATABRICKS_PRODUCT == 'kumo-relational-client'

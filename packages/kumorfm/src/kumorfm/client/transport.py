# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The two transports the RFM path can be pointed at.

:class:`~kumorfm.client.client.KumoClient` speaks HTTP to a Universal TFM NIM
and hands back a ``requests.Response``.
:class:`~kumorfm.client.databricks_serving.DatabricksServingClient` invokes a
named Model Serving endpoint through the Databricks SDK -- no HTTP session, no
base URL, no ``requests.Response`` -- and hands back a
:class:`~kumorfm.client.databricks_serving.ServingResponse`. Neither class
derives from the other, so the shared call sites (``RFMAPI`` and
``raise_on_error``) have to name both.
"""

from typing import TypeAlias

import requests

from kumorfm.client.client import KumoClient
from kumorfm.client.databricks_serving import (
    DatabricksServingClient,
    ServingResponse,
)

__all__ = [
    'RFMTransport',
    'TransportResponse',
]

RFMTransport: TypeAlias = KumoClient | DatabricksServingClient
TransportResponse: TypeAlias = requests.Response | ServingResponse

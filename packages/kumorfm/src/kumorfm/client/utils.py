# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumorfm.client.transport import TransportResponse
from kumorfm.exceptions import HTTPException


def raise_on_error(response: TransportResponse) -> None:
    r"""Raises an :class:`~kumorfm.exceptions.HTTPException` if a response does
    not return with an OK status code.
    """
    if not response.ok:
        assert response.status_code is not None
        raise HTTPException(response.status_code, response.text)

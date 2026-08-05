# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Types the SDK owns that the vendored ``kumorfm.api`` tree used to hold.

``kumorfm/api`` is re-copied wholesale from upstream ``kumo-api`` by
``scripts/sync_internal_packages.py``, so anything defined there is lost on the
next sync. These two are used across the runtime, so they live here instead.
"""

from kumorfm.api.common import StrEnum


class RunMode(StrEnum):
    r"""How much in-context data a prediction is given.

    The mode caps the number of in-context examples the SDK collects and also
    selects the per-hop neighbour fanout, trading accuracy against latency and
    payload size. It is sent as ``inference.run_mode``.

    The caps are 100 (``debug``), 1,000 (``fast``), 5,000 (``normal``) and
    10,000 (``best``); the exact counts live in ``kumorfm.rfm.rfm``. Temporal
    link prediction always uses the ``fast`` budget regardless of the mode.
    """
    FAST = 'fast'
    NORMAL = 'normal'
    BEST = 'best'
    DEBUG = 'debug'


class MissingType(StrEnum):
    r"""Sentinel for a table attribute the caller did not supply.

    Distinguishes "not given, so infer it" from an explicit ``None``, which
    means "this table has no such column".
    """
    VALUE = '???'

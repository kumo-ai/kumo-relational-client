# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The substrate both foundation models sit on.

:class:`~kumo_relational_engine.rfm.KumoRelational` and
:class:`~kumo_relational_engine.tfm.KumoTabular` take the same graph and the
same PQL, so the code that turns a query into a task -- and the limits that
task is sampled under -- belongs to neither of them. It lives here, above
both.

Nothing in this package may import from :mod:`kumo_relational_engine.rfm` or
:mod:`kumo_relational_engine.tfm` at runtime: that is what keeps it a substrate
rather than a place things drift into, and importing ``rfm.base`` from here
would be a cycle, since ``rfm/__init__`` builds the relational model.

The rest of ``rfm/base`` -- :class:`Sampler`, :class:`Table`, the column and
expression types -- is substrate too, filed under ``rfm`` only for historical
reasons. Moving it here is the next step.
"""

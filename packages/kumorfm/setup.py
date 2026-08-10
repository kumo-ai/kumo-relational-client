# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from setuptools import setup

# ``kumo-api`` and ``kumo-pql`` are internalized under ``kumorfm.api`` and
# ``kumorfm.pql`` (see their SOURCE.md files), so the SDK carries no external
# ``kumo-api`` dependency. ``pydantic`` used to arrive transitively via
# ``kumo-api``; it is now declared directly.

if bool(int(os.getenv('WITH_KUMOLIB', '1'))):
    kwargs = dict(cmake_source_dir='.')
else:
    kwargs = dict()

setup(
    install_requires=[
        'numpy',
        'pandas',
        'pyarrow>=8.0.0',
        'python-dateutil',
        'requests>=2.28.2',
        'urllib3',
        'typing_extensions>=4.5.0',
        'pydantic>=2.7',
        'sdfm-connectors>=0.2,<1',
        'antlr4-python3-runtime==4.9.3',
        'rich>=9.0.0',
        'jinja2',
        'tabulate',
    ],
    **kwargs,
)

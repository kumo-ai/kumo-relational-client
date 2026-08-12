# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

# These tests replay the request/response examples that ship beside the
# Universal TFM OpenAPI contract, to check this client encodes what the
# contract says. The examples are not vendored here, so the tests skip unless
# a checkout is pointed at explicitly. The default is the sibling-directory
# layout the maintainers use; anyone else sets NEMOTRON_PREDICT_CONTRACT_DIR.
_ENV_CONTRACT_DIR = 'NEMOTRON_PREDICT_CONTRACT_DIR'
CANONICAL_SPEC_DIR = Path(
    os.environ.get(_ENV_CONTRACT_DIR, '../structured-data-api')
)
CANONICAL_EXAMPLES_DIR = CANONICAL_SPEC_DIR / 'examples'

canonical_examples_available = pytest.mark.skipif(
    not CANONICAL_EXAMPLES_DIR.exists(),
    reason=f'contract examples not found at {CANONICAL_EXAMPLES_DIR}; set '
    f'{_ENV_CONTRACT_DIR} to a checkout to run these',
)


def load_canonical_example(*parts: str) -> dict:
    path = CANONICAL_EXAMPLES_DIR.joinpath(*parts)
    return json.loads(path.read_text())


@pytest.fixture
def context_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'row_id': ['ctx-0', 'ctx-1', 'ctx-2', 'ctx-3', 'ctx-4', 'ctx-5'],
            'age': [22, 31, 47, 54, 36, 28],
            'score': [0.20, 0.78, 0.35, 0.91, 0.66, 0.29],
            'target_col': ['no', 'yes', 'no', 'yes', 'yes', 'no'],
        }
    )


@pytest.fixture
def predict_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'row_id': ['q-0', 'q-1'],
            'age': [33, 49],
            'score': [0.72, 0.30],
        }
    )

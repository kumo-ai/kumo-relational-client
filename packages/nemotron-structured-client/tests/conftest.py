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
# contract says.
#
# The examples are vendored under `contract_examples/`, with the revision they
# came from recorded in its PROVENANCE.json. They used to be read from a
# sibling checkout of the contract repository, which meant they were absent
# everywhere except a maintainer's laptop: the tests skipped in CI, and a
# rename on the contract side went unnoticed for as long as nobody ran them by
# hand. Vendoring costs a copy that can fall behind, which `PROVENANCE.json`
# and `test_contract_examples_provenance.py` exist to make visible, and buys a
# check that actually runs.
#
# Point NEMOTRON_STRUCTURED_CONTRACT_DIR at a checkout of the contract
# repository to replay the examples as they are there instead, which is how to
# find out whether the vendored copy has gone stale.
_ENV_CONTRACT_DIR = 'NEMOTRON_STRUCTURED_CONTRACT_DIR'
_VENDORED_EXAMPLES_DIR = Path(__file__).parent / 'contract_examples'
_contract_dir = os.environ.get(_ENV_CONTRACT_DIR)
CANONICAL_EXAMPLES_DIR = (
    Path(_contract_dir) / 'examples'
    if _contract_dir
    else _VENDORED_EXAMPLES_DIR
)

canonical_examples_available = pytest.mark.skipif(
    not CANONICAL_EXAMPLES_DIR.exists(),
    reason=f'contract examples not found at {CANONICAL_EXAMPLES_DIR}; unset '
    f'{_ENV_CONTRACT_DIR} to use the vendored copy',
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

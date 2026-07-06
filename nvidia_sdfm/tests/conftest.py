from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

CANONICAL_SPEC_DIR = Path('../structured-data-api')
CANONICAL_EXAMPLES_DIR = CANONICAL_SPEC_DIR / 'examples'

canonical_examples_available = pytest.mark.skipif(
    not CANONICAL_EXAMPLES_DIR.exists(),
    reason='canonical structured-data-api checkout is not available beside '
    'this repo',
)


def load_canonical_example(*parts: str) -> dict:
    path = CANONICAL_EXAMPLES_DIR.joinpath(*parts)
    return json.loads(path.read_text())


@pytest.fixture
def context_df() -> pd.DataFrame:
    return pd.DataFrame({
        'row_id': ['ctx-0', 'ctx-1', 'ctx-2', 'ctx-3', 'ctx-4', 'ctx-5'],
        'age': [22, 31, 47, 54, 36, 28],
        'score': [0.20, 0.78, 0.35, 0.91, 0.66, 0.29],
        'target_col': ['no', 'yes', 'no', 'yes', 'yes', 'no'],
    })


@pytest.fixture
def predict_df() -> pd.DataFrame:
    return pd.DataFrame({
        'row_id': ['q-0', 'q-1'],
        'age': [33, 49],
        'score': [0.72, 0.30],
    })

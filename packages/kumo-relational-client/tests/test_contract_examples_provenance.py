# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest

VENDORED = Path(__file__).parent / 'contract_examples'
PROVENANCE = VENDORED / 'PROVENANCE.json'


@pytest.fixture
def provenance() -> dict:
    return json.loads(PROVENANCE.read_text())


def test_the_vendored_examples_record_where_they_came_from(
    provenance: dict,
) -> None:
    r"""A copy without a source is a copy nobody can check, and these exist to
    be checked against the contract they were taken from."""
    assert provenance['source_repository'].endswith('structured-data-api.git')
    assert len(provenance['source_revision']) == 40
    assert provenance['source_path'] == 'examples'


def test_every_vendored_example_matches_its_recorded_hash(
    provenance: dict,
) -> None:
    r"""Catches an example edited in place here rather than updated from the
    contract, which would quietly turn a conformance check into a check that
    the client agrees with itself."""
    for name, expected in provenance['files'].items():
        actual = hashlib.sha256((VENDORED / name).read_bytes()).hexdigest()
        assert actual == expected, name


def test_the_record_lists_exactly_what_is_vendored(provenance: dict) -> None:
    on_disk = {
        str(p.relative_to(VENDORED))
        for p in VENDORED.rglob('*.json')
        if p.name != 'PROVENANCE.json'
    }

    assert on_disk == set(provenance['files'])

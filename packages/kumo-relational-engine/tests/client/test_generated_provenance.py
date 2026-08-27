# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Checks that hold without a checkout of the contract.

The drift tests in ``test_tfm_codegen.py`` compare the generated bindings
against the canonical OpenAPI contract, which lives outside this repository, so
they skip wherever it is absent -- which is everywhere except a maintainer's
machine. These checks cover the failure that does not need the contract to
detect: the generated file being edited by hand instead of regenerated.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

GENERATED_DIR = (
    Path(__file__).resolve().parents[2]
    / 'src'
    / 'kumo_relational_engine'
    / 'client'
    / 'generated'
)
PROVENANCE = GENERATED_DIR / 'PROVENANCE.json'


@pytest.fixture
def provenance() -> dict:
    return json.loads(PROVENANCE.read_text())


def test_the_generated_bindings_match_their_recorded_hash(
    provenance: dict,
) -> None:
    r"""A hand-edit of generated code is the failure this catches.

    ``DO NOT EDIT`` is a request; this is the check. Regenerating updates both
    the file and this record together, so only an edit that skipped the
    generator moves one without the other.
    """
    generated = GENERATED_DIR / provenance['generated_file']
    actual = hashlib.sha256(generated.read_bytes()).hexdigest()

    assert actual == provenance['generated_sha256'], (
        f'{generated.name} does not match the hash recorded in '
        f'{PROVENANCE.name}. Regenerate it with '
        f'{provenance["generator"]} rather than editing it, and update the '
        f'record.'
    )


def test_the_record_names_the_contract_revision_it_came_from(
    provenance: dict,
) -> None:
    r"""Without this, "which contract is this built against" has no answer in a
    checkout that cannot reach the contract.
    """
    assert provenance['source_path'] == 'nim-sd.openapi.yaml'
    assert len(provenance['source_revision']) == 40
    assert len(provenance['source_sha256']) == 64


def test_the_generated_file_agrees_with_the_recorded_source(
    provenance: dict,
) -> None:
    r"""The generator stamps the spec's hash into the file it writes. If that
    disagrees with the record, the two were produced from different contracts.
    """
    generated = GENERATED_DIR / provenance['generated_file']
    stamped = re.search(
        r'# Source SHA256: ([0-9a-f]{64})', generated.read_text()
    )

    assert stamped is not None, 'the generated file carries no source hash'
    assert stamped.group(1) == provenance['source_sha256']

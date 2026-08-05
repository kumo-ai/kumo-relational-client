# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Tests for ``scripts/sync_internal_packages.py``.

``kumorfm/api`` and ``kumorfm/pql`` are vendored: the sync script deletes each
tree and re-copies it from upstream. Anything the SDK removed from those trees
comes back, and anything it added there is lost, unless the script is told
otherwise. Nothing else in the suite exercises that, so a mistake here is
invisible until someone re-syncs and the package stops importing.

These run the real ``sync_api`` against a synthetic upstream that carries every
module the SDK drops, so a regression shows up as a restored file rather than
as a broken import weeks later.
"""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'scripts' / 'sync_internal_packages.py'
VENDORED_API = REPO / 'src' / 'kumorfm' / 'api'


def load_sync() -> Any:
    spec = importlib.util.spec_from_file_location('_sync', SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def sync() -> Any:
    return load_sync()


@pytest.fixture
def upstream(tmp_path: Path, sync: Any) -> Path:
    r"""A stand-in for kumo-api: today's tree plus every module we drop.

    Built from the vendored copy so it stays in step with the real package, and
    each excluded module is written back with a marker, which is what a real
    re-sync would deliver.
    """
    root = tmp_path / 'kumoapi'
    shutil.copytree(VENDORED_API, root,
                    ignore=shutil.ignore_patterns('__pycache__'))
    for name in sync.API_EXCLUDE:
        target = root / name
        if target.suffix != '.py':
            target.mkdir(parents=True, exist_ok=True)
            target = target / '__init__.py'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('RESTORED_FROM_UPSTREAM = True\n')
    # Upstream's own copies of the files the SDK owns.
    for name in sync.API_OWNED:
        (root / name).write_text('UPSTREAM_VERSION = True\n')
    return root


@pytest.fixture
def resynced(upstream: Path, sync: Any, tmp_path: Path) -> Path:
    r"""Run the real sync, then restore the working tree."""
    backup = tmp_path / 'backup'
    shutil.copytree(VENDORED_API, backup,
                    ignore=shutil.ignore_patterns('__pycache__'))
    try:
        sync.sync_api(upstream.parent)
        yield VENDORED_API
    finally:
        shutil.rmtree(VENDORED_API, ignore_errors=True)
        shutil.copytree(backup, VENDORED_API)


def test_excluded_modules_do_not_come_back(resynced: Path, sync: Any) -> None:
    restored = [name for name in sync.API_EXCLUDE
                if (resynced / name).exists()]
    assert not restored, (
        f'a re-sync restored modules the SDK dropped: {restored}. Add them to '
        f'API_EXCLUDE in {SCRIPT.name}.')


def test_owned_files_survive_a_resync(resynced: Path, sync: Any) -> None:
    r"""API_OWNED files are the SDK's; upstream must not overwrite them."""
    for name in sync.API_OWNED:
        text = (resynced / name).read_text()
        assert 'UPSTREAM_VERSION' not in text, (
            f'{name} was replaced by upstream despite being in API_OWNED')


def test_exclusions_are_matched_by_path_not_just_basename(
        resynced: Path) -> None:
    r"""``explain/common.py`` is dropped while the top-level ``common.py`` stays.

    ``shutil.ignore_patterns`` matches basenames only, so a bare ``common.py``
    entry would silently delete both. ``common.py`` holds ``StrEnum``, the base
    of every enum in the tree.
    """
    assert not (resynced / 'explain' / 'common.py').exists()
    assert (resynced / 'common.py').exists()
    assert 'class StrEnum' in (resynced / 'common.py').read_text()


def test_modules_the_sdk_imports_are_kept(resynced: Path) -> None:
    for name in ('typing.py', 'task.py', 'json_serde.py', 'graph.py',
                 'table.py', 'explain/gradient.py', 'rfm/context.py',
                 'rfm/inference.py', 'pquery/validated_predictive_query.py'):
        assert (resynced / name).exists(), f'{name} is imported by the SDK'


def test_package_still_imports_after_a_resync(resynced: Path) -> None:
    r"""The end state a re-sync has to leave behind: a working package."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, '-c',
         'import kumorfm; from kumorfm.api.rfm import RFMPredictRequest; '
         'from kumorfm.api.explain import GraphGradientScore; '
         'from kumorfm.runmode import RunMode; print(RunMode.FAST.value)'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-800:]
    assert result.stdout.strip() == 'fast'


def test_runmode_is_not_defined_in_the_vendored_tree() -> None:
    r"""RunMode and MissingType are used across the runtime, so they cannot
    live under ``api/``: a re-sync would delete them and break every import.
    """
    for module in VENDORED_API.rglob('*.py'):
        text = module.read_text()
        assert 'class RunMode' not in text, (
            f'RunMode is defined in the vendored {module.name}; it belongs in '
            f'kumorfm/runmode.py, which a re-sync does not touch.')
        assert 'class MissingType' not in text


def test_apply_once_rejects_a_moved_anchor(sync: Any, tmp_path: Path) -> None:
    r"""The guard that turns an upstream change into a loud failure."""
    path = tmp_path / 'patched.py'
    path.write_text('anchor\n')
    sync._apply_once(path, 'anchor', 'replacement', 'test')
    assert path.read_text() == 'replacement\n'

    with pytest.raises(SystemExit, match='expected exactly one anchor'):
        sync._apply_once(path, 'anchor', 'replacement', 'test')


def test_apply_once_rejects_an_ambiguous_anchor(sync: Any,
                                                tmp_path: Path) -> None:
    path = tmp_path / 'patched.py'
    path.write_text('anchor\nanchor\n')
    with pytest.raises(SystemExit, match='expected exactly one anchor'):
        sync._apply_once(path, 'anchor', 'replacement', 'test')

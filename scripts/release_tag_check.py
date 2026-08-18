# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard: a release tag must match the tagged package's declared version.

Release tags have the form ``<distribution>/v<version>`` (for example
``nemotron-structured-client/v0.1.0``). The publish jobs upload whatever version the package
files declare, so a tag that disagrees with ``__version__`` would publish a
release under the wrong name. This check fails the pipeline first.

Usage: release_tag_check.py [tag]   (defaults to $CI_COMMIT_TAG)
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

VERSION_FILES = {
    'nemotron-structured-connectors': 'packages/nemotron-structured-connectors/src/nemotron_structured_connectors/__init__.py',
    'nemotron-structured-client': 'packages/nemotron-structured-client/src/nemotron_structured/_version.py',
    'nemotron-relational': 'packages/nemotron-relational/src/nemotron_relational/_version.py',
}

_VERSION_RE = re.compile(
    r"^__version__\s*=\s*['\"]([^'\"]+)['\"]\s*$",
    re.MULTILINE,
)


def declared_version(distribution: str) -> str:
    path = REPO_ROOT / VERSION_FILES[distribution]
    match = _VERSION_RE.search(path.read_text())
    if match is None:
        raise SystemExit(f'no __version__ found in {path}')
    return match.group(1)


def check(tag: str) -> str:
    distribution, separator, version = tag.partition('/v')
    if not separator or not version:
        raise SystemExit(
            f'tag {tag!r} does not match <distribution>/v<version>; known '
            f'distributions: {sorted(VERSION_FILES)}'
        )
    if distribution not in VERSION_FILES:
        raise SystemExit(
            f'unknown distribution {distribution!r} in tag {tag!r}; known '
            f'distributions: {sorted(VERSION_FILES)}'
        )
    declared = declared_version(distribution)
    if version != declared:
        raise SystemExit(
            f'tag {tag!r} says {version!r} but {VERSION_FILES[distribution]} '
            f'declares {declared!r}; bump the version file and re-tag'
        )
    return f'ok: {tag} matches declared version {declared}'


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('CI_COMMIT_TAG')
    if not tag:
        raise SystemExit('no tag given and CI_COMMIT_TAG is not set')
    print(check(tag))


if __name__ == '__main__':
    main()

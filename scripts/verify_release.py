# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wait until a published release is visible on a PyPI simple index.

Artifactory's simple index can lag a successful twine upload, so the verify
stage polls until the release files appear (or a deadline passes) before the
job goes on to ``pip install`` the release. Uses only the standard library:
the verify jobs run before anything is installed.

Usage:
    verify_release.py <distribution> <version> --index <simple-index-url>
        [--require-wheel-only] [--attempts N] [--delay SECONDS]

``--require-wheel-only`` additionally fails if the index lists an sdist for
the version (kumorfm must never publish source distributions).
"""

from __future__ import annotations

import argparse
import html.parser
import re
import sys
import time
import urllib.error
import urllib.request


class _AnchorTextParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.filenames: list[str] = []
        self._in_anchor = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == 'a':
            self._in_anchor = True

    def handle_endtag(self, tag: str) -> None:
        if tag == 'a':
            self._in_anchor = False

    def handle_data(self, data: str) -> None:
        if self._in_anchor and data.strip():
            self.filenames.append(data.strip())


def normalize(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name).lower()


def fetch_filenames(index: str, distribution: str) -> list[str]:
    url = f"{index.rstrip('/')}/{normalize(distribution)}/"
    with urllib.request.urlopen(url, timeout=30) as response:
        parser = _AnchorTextParser()
        parser.feed(response.read().decode('utf-8', errors='replace'))
        return parser.filenames


def release_files(filenames: list[str], distribution: str,
                  version: str) -> list[str]:
    prefix = f"{distribution.replace('-', '_')}-{version}"
    return [
        name for name in filenames
        if name.startswith(f'{prefix}-') or name == f'{prefix}.tar.gz'
    ]


def main() -> None:
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument('distribution')
    arguments.add_argument('version')
    arguments.add_argument('--index', required=True)
    arguments.add_argument('--require-wheel-only', action='store_true')
    arguments.add_argument('--attempts', type=int, default=30)
    arguments.add_argument('--delay', type=float, default=10.0)
    options = arguments.parse_args()

    found: list[str] = []
    for attempt in range(1, options.attempts + 1):
        try:
            filenames = fetch_filenames(options.index, options.distribution)
            found = release_files(
                filenames, options.distribution, options.version)
        except urllib.error.URLError as error:
            print(f'attempt {attempt}: index not readable yet ({error})')
        else:
            if found:
                break
            print(
                f'attempt {attempt}: {options.distribution}=='
                f'{options.version} not on the index yet '
                f'({len(filenames)} other files listed)'
            )
        if attempt < options.attempts:
            time.sleep(options.delay)

    if not found:
        sys.exit(
            f'{options.distribution}=={options.version} did not appear on '
            f'{options.index} after {options.attempts} attempts'
        )

    print(f'found {len(found)} release file(s):')
    for name in sorted(found):
        print(f'  {name}')

    if options.require_wheel_only:
        sdists = [name for name in found if not name.endswith('.whl')]
        if sdists:
            sys.exit(f'wheel-only distribution has non-wheel files: {sdists}')


if __name__ == '__main__':
    main()

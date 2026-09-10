# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the wheel set built for a coordinated SDK release.

Usage:
    validate_wheel_bundle.py <wheel-directory> <version> [--engine-only]

The full candidate consists of twenty-two wheels: an engine wheel for each of
CPython 3.10 through 3.13 on each of the five supported platforms, and one
platform-independent wheel for each of the client and connectors packages.

Windows on ARM64 is deliberately absent. No wheel is built for it, so an
`[relational]` install there still has nothing to resolve.
"""

from __future__ import annotations

import argparse
import email.parser
import re
import sys
import zipfile
from pathlib import Path

from packaging.tags import Tag
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

ENGINE = 'kumo-relational-engine'
CLIENT = 'kumo-relational-client'
CONNECTORS = 'kumo-connectors'

ENGINE_PYTHONS = ('cp310', 'cp311', 'cp312', 'cp313')
# The exact platform tag each wheel must carry. `manylinux_2_28` is the glibc
# baseline the first release shipped, and `macosx_11_0` matches the deployment
# target set in the engine's cibuildwheel configuration; a wheel built against
# a newer floor than either would install on fewer machines than the docs
# promise, so it is rejected rather than published.
ENGINE_PLATFORMS = (
    'manylinux_2_28_x86_64',
    'manylinux_2_28_aarch64',
    'macosx_11_0_x86_64',
    'macosx_11_0_arm64',
    'win_amd64',
)
ENGINE_TARGETS = {
    (python, python, platform)
    for python in ENGINE_PYTHONS
    for platform in ENGINE_PLATFORMS
}
PURE_TAGS = {Tag('py3', 'none', 'any')}
_MANYLINUX = re.compile(r'manylinux_2_(\d+)_(x86_64|aarch64)')
_MACOS = re.compile(r'macosx_(\d+)_(\d+)_(x86_64|arm64)')


def _metadata(wheel: Path) -> tuple[str, str, set[Tag]]:
    with zipfile.ZipFile(wheel) as archive:
        metadata_paths = [
            name
            for name in archive.namelist()
            if name.endswith('.dist-info/METADATA')
        ]
        wheel_paths = [
            name
            for name in archive.namelist()
            if name.endswith('.dist-info/WHEEL')
        ]
        if len(metadata_paths) != 1 or len(wheel_paths) != 1:
            raise ValueError(
                f'{wheel.name}: expected one METADATA and one WHEEL file'
            )
        parser = email.parser.Parser()
        metadata = parser.parsestr(
            archive.read(metadata_paths[0]).decode('utf-8')
        )
        wheel_metadata = parser.parsestr(
            archive.read(wheel_paths[0]).decode('utf-8')
        )

    tags = {
        Tag(*value.split('-', 2)) for value in wheel_metadata.get_all('Tag', [])
    }
    return metadata['Name'], metadata['Version'], tags


def _linux_platform(platforms: set[str]) -> str:
    # auditwheel emits the wheel's own tag alongside every older alias it also
    # satisfies, so a manylinux wheel legitimately carries several.
    parsed = []
    for platform in platforms:
        match = _MANYLINUX.fullmatch(platform)
        if match is None:
            raise ValueError(f'engine wheel mixes {platform} with manylinux')
        parsed.append((int(match.group(1)), match.group(2)))
    architectures = {architecture for _minor, architecture in parsed}
    if len(architectures) != 1:
        raise ValueError(f'engine wheel mixes architectures: {architectures}')
    if any(minor > 28 for minor, _architecture in parsed):
        raise ValueError(
            'engine wheel requires a glibc baseline newer than 2.28'
        )
    (architecture,) = architectures
    required = f'manylinux_2_28_{architecture}'
    if required not in platforms:
        raise ValueError(f'engine wheel is missing tag {required}')
    return required


def _engine_target(tags: frozenset[Tag]) -> tuple[str, str, str]:
    python_abis = {(tag.interpreter, tag.abi) for tag in tags}
    if len(python_abis) != 1:
        raise ValueError(f'engine wheel mixes Python/ABI tags: {python_abis}')
    (python_abi,) = python_abis

    platforms = {tag.platform for tag in tags}
    if any(_MANYLINUX.fullmatch(platform) for platform in platforms):
        return (*python_abi, _linux_platform(platforms))

    if len(platforms) != 1:
        raise ValueError(f'engine wheel carries several platforms: {platforms}')
    (platform,) = platforms

    # A macOS or Windows wheel names exactly one platform, so the tag it
    # carries is the whole claim it makes and is compared literally.
    if _MACOS.fullmatch(platform) or platform == 'win_amd64':
        return (*python_abi, platform)

    raise ValueError(f'engine wheel has unexpected platform {platform}')


def validate(directory: Path, version: str, engine_only: bool) -> None:
    expected_distributions = {ENGINE}
    if not engine_only:
        expected_distributions.update({CLIENT, CONNECTORS})
    found_engine_targets: set[tuple[str, str, str]] = set()
    found_pure: set[str] = set()
    wheels = sorted(directory.glob('*.whl'))

    expected_count = len(ENGINE_TARGETS) + (0 if engine_only else 2)
    if len(wheels) != expected_count:
        raise ValueError(
            f'expected {expected_count} wheels, found {len(wheels)}: '
            f'{[wheel.name for wheel in wheels]}'
        )
    non_wheels = [
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix != '.whl'
    ]
    if non_wheels:
        raise ValueError(
            f'wheel directory contains non-wheel files: {non_wheels}'
        )

    wanted_version = Version(version)
    for wheel in wheels:
        distribution, filename_version, _build, filename_tags = (
            parse_wheel_filename(wheel.name)
        )
        name = canonicalize_name(distribution)
        if name not in expected_distributions:
            raise ValueError(f'{wheel.name}: unexpected distribution {name!r}')
        if filename_version != wanted_version:
            raise ValueError(
                f'{wheel.name}: filename version {filename_version} != {version}'
            )
        tags = frozenset(filename_tags)
        if name == ENGINE:
            try:
                target = _engine_target(tags)
            except ValueError as error:
                raise ValueError(f'{wheel.name}: {error}') from error
            if target not in ENGINE_TARGETS:
                raise ValueError(f'{wheel.name}: unexpected target {target}')
            if target in found_engine_targets:
                raise ValueError(f'{wheel.name}: duplicate target {target}')
            found_engine_targets.add(target)
        else:
            if tags != PURE_TAGS:
                raise ValueError(
                    f'{wheel.name}: expected py3-none-any, found '
                    f'{sorted(map(str, tags))}'
                )
            if name in found_pure:
                raise ValueError(f'{wheel.name}: duplicate pure wheel')
            found_pure.add(name)

        metadata_name, metadata_version, metadata_tags = _metadata(wheel)
        if canonicalize_name(metadata_name) != name:
            raise ValueError(
                f'{wheel.name}: METADATA name {metadata_name!r} does not match'
            )
        if Version(metadata_version) != wanted_version:
            raise ValueError(
                f'{wheel.name}: METADATA version {metadata_version} != {version}'
            )
        if metadata_tags != tags:
            raise ValueError(
                f'{wheel.name}: WHEEL tags {sorted(map(str, metadata_tags))} '
                'do not match filename tags'
            )

    if found_engine_targets != ENGINE_TARGETS:
        raise ValueError(
            f'missing engine targets: {ENGINE_TARGETS - found_engine_targets}'
        )
    expected_pure = set() if engine_only else {CLIENT, CONNECTORS}
    if found_pure != expected_pure:
        raise ValueError(f'missing pure wheels: {expected_pure - found_pure}')

    print(f'validated {len(wheels)} wheels for SDK {version}')
    for wheel in wheels:
        print(f'  {wheel.name}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('version')
    parser.add_argument('--engine-only', action='store_true')
    options = parser.parse_args()
    try:
        validate(options.directory, options.version, options.engine_only)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        sys.exit(f'invalid wheel bundle: {error}')


if __name__ == '__main__':
    main()

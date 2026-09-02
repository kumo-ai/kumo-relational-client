# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test an installed SDK on a Databricks serverless-compatible host."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import kumo_connectors
import kumo_relational_client
import kumo_relational_engine
import kumo_relational_engine.relationallib as relationallib
import pandas as pd
from databricks import sdk as databricks_sdk
from kumo_relational_client import RelationalClient, relational

_DISTRIBUTIONS = {
    'kumo-connectors',
    'kumo-relational-client',
    'kumo-relational-engine',
}


def _normalized(name: str) -> str:
    return name.lower().replace('_', '-').replace('.', '-')


def _verify_install_report(
    report_path: Path, wheel_directory: Path, version: str
) -> None:
    report = json.loads(report_path.read_text())
    installs = {
        _normalized(item['metadata']['name']): item
        for item in report.get('install', [])
        if _normalized(item['metadata']['name']) in _DISTRIBUTIONS
    }
    assert set(installs) == _DISTRIBUTIONS, installs

    wheel_directory = wheel_directory.resolve()
    for name, item in installs.items():
        assert item['metadata']['version'] == version, item['metadata']
        url = urlparse(item['download_info']['url'])
        assert url.scheme == 'file', f'{name} was not installed locally: {url}'
        wheel = Path(unquote(url.path)).resolve()
        assert wheel.parent == wheel_directory, (
            f'{name} came from {wheel}, expected {wheel_directory}'
        )
        assert wheel.suffix == '.whl', f'{name} did not come from a wheel'


class _Endpoints:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def query(
        self, *, name: str, dataframe_records: list[dict[str, str]]
    ) -> dict[str, Any]:
        self.calls.append(
            {'name': name, 'dataframe_records': dataframe_records}
        )
        request = json.loads(dataframe_records[0]['request_json'])
        table = request['predict']['instance_table']
        instance_id_index = table['columns'].index('instance_id')
        instance_ids = [str(row[instance_id_index]) for row in table['rows']]
        response = {
            'id': 'candidate-response',
            'model': 'kumo-relational',
            'predictions': [
                {'id': instance_id, 'row_index': index, 'prediction': True}
                for index, instance_id in enumerate(instance_ids)
            ],
            'metadata': {'task_kind': 'binary_classification'},
        }
        return {'predictions': [{'response_json': json.dumps(response)}]}


class _Workspace:
    def __init__(self) -> None:
        self.serving_endpoints = _Endpoints()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('version')
    parser.add_argument('architecture', choices=('x86_64', 'aarch64'))
    parser.add_argument('--install-report', type=Path)
    parser.add_argument('--wheel-directory', type=Path)
    options = parser.parse_args()

    if (options.install_report is None) != (options.wheel_directory is None):
        parser.error(
            '--install-report and --wheel-directory must be supplied together'
        )
    if options.install_report is not None:
        _verify_install_report(
            options.install_report, options.wheel_directory, options.version
        )

    actual_architecture = platform.machine().lower()
    aliases = {'amd64': 'x86_64', 'arm64': 'aarch64'}
    actual_architecture = aliases.get(actual_architecture, actual_architecture)
    assert actual_architecture == options.architecture, (
        f'running on {actual_architecture}, expected {options.architecture}'
    )
    assert platform.python_version_tuple()[:2] == ('3', '12')
    assert hasattr(databricks_sdk, 'WorkspaceClient')

    versions = {
        'kumo-relational-client': kumo_relational_client.__version__,
        'kumo-relational-engine': kumo_relational_engine.__version__,
        'kumo-connectors': kumo_connectors.__version__,
    }
    assert set(versions.values()) == {options.version}, versions
    assert hasattr(relationallib.NeighborSampler, 'seed')

    graph = relational.Graph.from_data(
        {'users': pd.DataFrame({'user_id': [1, 2], 'churned': [False, True]})},
        verbose=False,
    )
    assert list(graph.tables) == ['users']

    workspace = _Workspace()
    with RelationalClient.for_databricks_serving(
        'kumo-relational-candidate', workspace_client=workspace
    ) as client:
        assert client.models() == ['kumo-relational']
        result = client.relational(graph).predict(
            'PREDICT users.churned FOR EACH users.user_id',
            indices=[1],
            verbose=False,
        )
    assert bool(result.loc[0, 'PREDICTION']) is True
    assert len(workspace.serving_endpoints.calls) == 1
    call = workspace.serving_endpoints.calls[0]
    assert call['name'] == 'kumo-relational-candidate'
    assert len(call['dataframe_records']) == 1
    assert 'request_json' in call['dataframe_records'][0]

    print(
        f'verified SDK {options.version} on Python '
        f'{platform.python_version()} {actual_architecture}'
    )


if __name__ == '__main__':
    main()

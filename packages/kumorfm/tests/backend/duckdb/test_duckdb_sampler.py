# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from pathlib import Path

import pytest

from kumorfm.rfm import Graph

duckdb = pytest.importorskip('duckdb', reason="'duckdb' extension "
                             "not installed")
pytest.importorskip('adbc_driver_duckdb', reason="'duckdb' extension "
                    "not installed")

from kumorfm.rfm.backend.duckdb import DuckDBSampler  # noqa: E402


@pytest.fixture()
def graph(tmp_path: Path) -> Graph:
    path = tmp_path / 'users.duckdb'
    connection = duckdb.connect(path)
    connection.execute("CREATE TABLE users AS "
                       "SELECT i AS user_id, i % 40 AS age "
                       "FROM range(200) t(i)")
    connection.execute("CREATE UNIQUE INDEX users_pkey ON users (user_id)")
    connection.close()

    return Graph.from_duckdb(
        path,
        tables=[dict(name='users', primary_key='user_id')],
        verbose=False,
    )


def _sample(sampler: DuckDBSampler, random_seed: int | None) -> list[int]:
    df = sampler._sample_entity_table('users', {'user_id'}, 5,
                                      random_seed=random_seed)
    return sorted(df['user_id'].tolist())


def test_random_seed_is_reproducible(graph: Graph) -> None:
    # Regression test for `sampler-random-seed-ignored-on-sql-backends.md`:
    # DuckDB supports `USING SAMPLE ... REPEATABLE (seed)`, so reusing a seed
    # must draw the same in-context examples.
    sampler = DuckDBSampler(graph, verbose=False)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert _sample(sampler, 42) == _sample(sampler, 42)
        assert _sample(sampler, 42) != _sample(sampler, 7)


def test_unseeded_sample_is_random(graph: Graph) -> None:
    sampler = DuckDBSampler(graph, verbose=False)

    samples = {tuple(_sample(sampler, None)) for _ in range(5)}
    assert len(samples) > 1


def test_discovery_on_an_empty_database_names_what_it_searched(
        tmp_path: Path,  #
) -> None:
    r"""graph-duckdb-empty-discovery-not-rejected.md

    ``_require_discovered_tables`` was wired into three of the four discovery
    constructors. DuckDB, which needs it most because it creates a database
    rather than failing on a mistyped path, was the one left out.
    """
    path = tmp_path / 'empty.duckdb'
    duckdb.connect(str(path)).close()

    with pytest.raises(ValueError, match='No tables found in the DuckDB'):
        Graph.from_duckdb(str(path), verbose=False)


def test_missing_database_is_reported_rather_than_created(
        tmp_path: Path,  #
) -> None:
    r"""graph-duckdb-empty-discovery-not-rejected.md

    ``Graph.from_duckdb`` reaches DuckDB through its own ``adbc``-backed shim
    rather than the ``sdfm_connectors`` backend, so the existence guard has to
    be applied there too. A read must not write.
    """
    path = tmp_path / 'typo.duckdb'

    with pytest.raises(Exception, match='does not exist'):
        Graph.from_duckdb(str(path), verbose=False)
    assert not path.exists()

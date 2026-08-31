# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

NOTEBOOK = (
    Path(__file__).parents[1]
    / 'examples'
    / 'rfm'
    / 'notebooks'
    / 'kumo_relational_engine_adventureworks_lakebase.py'
)


def test_lakebase_notebook_copies_the_same_adventureworks_tables():
    source = NOTEBOOK.read_text(encoding='utf-8')
    for table in (
        'adventureworks_customers',
        'adventureworks_products',
        'adventureworks_sales_order_headers',
        'adventureworks_sales_order_details',
    ):
        assert table in source
    assert 'source_cursor.fetchmany' in source
    assert 'target_cursor.copy' in source
    assert 'ADD FOREIGN KEY' in source
    assert 'PARITY_QUERIES' in source
    assert 'source_edges != lakebase_edges' in source
    assert 'samples_match' in source


def test_lakebase_notebook_uses_in_memory_oauth_and_direct_connection():
    source = NOTEBOOK.read_text(encoding='utf-8')
    assert 'generate_database_credential' in source
    assert "connect(\n    'postgres'" in source
    assert "sslmode='require'" in source
    assert "widgets.text('password'" not in source


def test_lakebase_notebook_is_a_migration_and_parity_utility():
    source = NOTEBOOK.read_text(encoding='utf-8')
    assert 'Graph.from_databricks' in source
    assert 'Graph.from_postgres' in source
    assert '[relational,databricks,postgres]' in source
    assert 'RelationalClient' not in source
    assert '.predict(' not in source

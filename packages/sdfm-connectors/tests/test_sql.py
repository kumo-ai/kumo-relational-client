from __future__ import annotations

import pytest

from sdfm_connectors.sql import ConnectorError, quote_ident, resolve_sql


@pytest.mark.parametrize('ident, char, expected', [
    ('items', '"', '"items"'),
    ('items', '`', '`items`'),
    ('a"b', '"', '"a""b"'),
    ("a'b", "'", "'a''b'"),
])
def test_quote_ident(ident, char, expected):
    assert quote_ident(ident, char) == expected


@pytest.mark.parametrize('table', [
    'items',
    'main.items',
    'DB.SCHEMA.TABLE_1',
    '_private$tbl',
])
def test_resolve_sql_accepts_valid_identifiers(table):
    assert resolve_sql(table=table, query=None) == f'SELECT * FROM {table}'


def test_resolve_sql_passes_query_through():
    assert resolve_sql(table=None, query='SELECT 1') == 'SELECT 1'


@pytest.mark.parametrize('table', [
    'items; DROP TABLE users',
    'items --',
    'items OR 1=1',
    '1items',
    'items..b',
    '',
])
def test_resolve_sql_rejects_invalid_identifiers(table):
    with pytest.raises(ConnectorError) as excinfo:
        resolve_sql(table=table, query=None)
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


@pytest.mark.parametrize('kwargs', [
    {},
    {'table': 'items', 'query': 'SELECT 1'},
])
def test_resolve_sql_requires_exactly_one_source(kwargs):
    with pytest.raises(ConnectorError) as excinfo:
        resolve_sql(table=kwargs.get('table'), query=kwargs.get('query'))
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'

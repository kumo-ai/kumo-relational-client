from collections import defaultdict

import pytest

from kumoai.rfm import Graph
from kumoai.testing import onlyFullTest, withPackage


@withPackage('pooch')
def test_from_relbench() -> None:
    graph = Graph.from_relbench('f1', verbose=False)
    graph.validate()
    assert len(graph.tables) == 9
    assert len(graph.edges) == 13


@onlyFullTest
@withPackage('relbench')
@pytest.mark.parametrize('dataset', [
    'rel-f1',
    'rel-avito',
    'rel-event',
    'rel-amazon',
    'rel-hm',
    'rel-stack',
    'rel-trial',
])
def test_graph_inference(dataset: str) -> None:
    from relbench.datasets import get_dataset
    db = get_dataset(dataset).get_db(upto_test_timestamp=False)
    df_dict = {
        table_name: table.df
        for table_name, table in db.table_dict.items()
    }

    graph = Graph.from_data(df_dict, edges=[], verbose=False)

    # Necessary corrections - these are not wrong!
    if dataset == 'rel-f1':
        db.table_dict['drivers'].time_col = 'dob'

    graph.infer_links(verbose=False)

    # Necessary corrections - these are not wrong!
    if dataset == 'rel-avito':
        db.table_dict['Category'].fkey_col_to_pkey_table[
            'ParentCategoryID'] = 'Category'
        db.table_dict['Category'].fkey_col_to_pkey_table[
            'SubcategoryID'] = 'Category'

    fkey_to_pkey_dict = defaultdict(dict)
    for edge in graph.edges:
        fkey_to_pkey_dict[edge.src_table][edge.fkey] = edge.dst_table

    for table_name in df_dict.keys():
        table = db.table_dict[table_name]
        assert graph[table_name]._primary_key == table.pkey_col
        assert graph[table_name]._time_column == table.time_col
        assert fkey_to_pkey_dict[table_name] == table.fkey_col_to_pkey_table

import pytest

from kumoai.graph import Edge
from kumoai.rfm import Graph

try:
    from kumoai.rfm.backend.snow import Connection, SnowTable
except ImportError:
    pytest.skip("'snowflake' extension not installed", allow_module_level=True)


def test_graph(connection: Connection) -> None:
    graph = Graph(tables=[
        SnowTable(connection, name='USERS'),
        SnowTable(connection, name='ORDERS'),
        SnowTable(connection, name='ITEMS'),
    ])

    assert len(graph.tables) == 3

    assert graph['USERS'].primary_key is not None
    assert graph['USERS'].primary_key.name == 'USER_ID'
    assert graph['ORDERS'].primary_key is None
    assert graph['ITEMS'].primary_key is not None
    assert graph['ITEMS'].primary_key.name == 'ITEM_ID'

    assert graph['USERS'].time_column is None
    assert graph['ORDERS'].time_column is None
    assert graph['ITEMS'].time_column is None

    assert Edge('ORDERS', 'USER_ID', 'USERS') in graph.edges
    assert Edge('ORDERS', 'ITEM_ID', 'ITEMS') in graph.edges


def test_from_snowflake(connection: Connection) -> None:
    graph = Graph.from_snowflake(connection, verbose=False)

    assert len(graph.tables) == 3

    assert graph['USERS'].primary_key is not None
    assert graph['USERS'].primary_key.name == 'USER_ID'
    assert graph['ORDERS'].primary_key is None
    assert graph['ITEMS'].primary_key is not None
    assert graph['ITEMS'].primary_key.name == 'ITEM_ID'

    assert graph['USERS'].time_column is not None
    assert graph['USERS'].time_column.name == 'DOB'
    assert graph['ORDERS'].time_column is not None
    assert graph['ORDERS'].time_column.name == 'DATE'
    assert graph['ITEMS'].time_column is None

    assert Edge('ORDERS', 'USER_ID', 'USERS') in graph.edges
    assert Edge('ORDERS', 'ITEM_ID', 'ITEMS') in graph.edges


def test_from_semantic_view(connection: Connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()")
        result = cursor.fetchone()
        assert result is not None
        database, schema = result

    yaml = f"""
    name: semantic_view
    tables:
      - name: USER
        base_table:
          database: {database}
          schema: {schema}
          table: USERS
        primary_key:
          columns:
            - USER_ID
        dimensions:
          - name: AGE
            expr: USER.AGE
            data_type: NUMBER
          - name: DESC
            expr: CONCAT(GENDER, '-', USER.GENDER)
            data_type: VARCHAR(134217728)
      - name: TRANSACTION
        base_table:
          database: {database}
          schema: {schema}
          table: ORDERS
        dimensions:
          - name: PRICE
            expr: PRICE
        time_dimensions:
          - name: TIMESTAMP
            expr: DATE
            data_type: DATE
      - name: ITEM
        base_table:
          database: {database}
          schema: {schema}
          table: ITEMS
        primary_key:
          columns:
            - ITEM_ID
        facts:
          - name: ITEM_SALES
            expr: SUM(TRANSACTION.PRICE)
        dimensions:
          - name: CATEGORY
            expr: CATEGORY
    relationships:
      - name: TRANSACTION_TO_USER
        left_table: TRANSACTION
        right_table: USER
        relationship_columns:
          - left_column: USER_ID
            right_column: USER_ID
      - name: TRANSACTION_TO_ITEM
        left_table: TRANSACTION
        right_table: ITEM
        relationship_columns:
          - left_column: ITEM_ID
            right_column: ITEM_ID
    """

    with connection.cursor() as cursor:
        sql = (f"CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML(\n"
               f"    '{database}.{schema}',\n"
               f"    $${yaml}"
               f"    $$\n"
               f")")
        cursor.execute(sql)

    with pytest.warns(UserWarning, match="Failed to add column 'ITEM_SALES'"):
        graph = Graph.from_snowflake_semantic_view(
            semantic_view_name='semantic_view',
            connection=connection,
            verbose=False,
        )

    assert len(graph.tables) == 3

    assert graph['USER'].source_name == f'{database}.{schema}.USERS'
    assert graph['TRANSACTION'].source_name == f'{database}.{schema}.ORDERS'
    assert graph['ITEM'].source_name == f'{database}.{schema}.ITEMS'

    assert graph['USER'].primary_key is not None
    assert graph['USER'].primary_key.name == 'USER_ID'
    assert graph['TRANSACTION'].primary_key is None
    assert graph['ITEM'].primary_key is not None
    assert graph['ITEM'].primary_key.name == 'ITEM_ID'

    assert graph['USER'].time_column is None
    assert graph['TRANSACTION'].time_column is not None
    assert graph['TRANSACTION'].time_column.name == 'TIMESTAMP'
    assert graph['ITEM'].time_column is None

    assert set(c.name for c in graph['USER'].columns) == {  #
        'USER_ID', 'AGE', 'DESC'
    }
    assert set(c.name for c in graph['TRANSACTION'].columns) == {
        'USER_ID', 'ITEM_ID', 'PRICE', 'TIMESTAMP'
    }
    assert set(c.name for c in graph['ITEM'].columns) == {  #
        'ITEM_ID', 'CATEGORY'
    }

    assert Edge('TRANSACTION', 'USER_ID', 'USER') in graph.edges
    assert Edge('TRANSACTION', 'ITEM_ID', 'ITEM') in graph.edges

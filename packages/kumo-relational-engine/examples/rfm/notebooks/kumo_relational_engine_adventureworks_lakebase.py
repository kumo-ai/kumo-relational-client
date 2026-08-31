# Databricks notebook source
# ruff: noqa: E402, ERA001, F821
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# MAGIC %md
# MAGIC # AdventureWorks: copy and validate Lakebase
# MAGIC
# MAGIC This notebook copies the same four AdventureWorks Delta tables used by
# MAGIC `kumo_relational_engine_adventureworks_databricks.ipynb` into Lakebase,
# MAGIC then validates data, metadata, graph, and sampled-row parity. It is a
# MAGIC provider-specific migration/validation utility, not a second prediction
# MAGIC walkthrough. Lakebase is accessed directly over PostgreSQL; the SQL
# MAGIC Warehouse is used only to read the source Delta tables.

# COMMAND ----------

# MAGIC %pip install --find-links /Volumes/main/default/wheels "kumo-relational-client[relational,databricks,postgres]" "databricks-sdk>=0.68.0,<1.0" "numpy<2"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import math
import re
import time
from datetime import datetime

import pandas as pd
from databricks import sql as dbsql
from databricks.sdk import WorkspaceClient
from kumo_connectors import connect
from kumo_connectors.sql import quote_ident
from kumo_relational_client import relational

dbutils = dbutils
display = display

# COMMAND ----------

dbutils.widgets.text('source_catalog', 'main', 'Source catalog')
dbutils.widgets.text('source_schema', 'kumo_rfm', 'Source schema')
dbutils.widgets.text('warehouse_id', '', 'Source SQL warehouse ID')
dbutils.widgets.text(
    'lakebase_endpoint',
    '',
    'Lakebase endpoint resource path',
)
dbutils.widgets.text('lakebase_host', '', 'Lakebase endpoint host')
dbutils.widgets.text(
    'lakebase_database',
    'databricks_postgres',
    'Lakebase database',
)
dbutils.widgets.text(
    'lakebase_schema',
    'adventureworks',
    'Lakebase target schema',
)
dbutils.widgets.dropdown('copy_tables', 'true', ['true', 'false'])

source_catalog = dbutils.widgets.get('source_catalog')
source_schema = dbutils.widgets.get('source_schema')
warehouse_id = dbutils.widgets.get('warehouse_id')
lakebase_endpoint = dbutils.widgets.get('lakebase_endpoint')
lakebase_host = dbutils.widgets.get('lakebase_host')
lakebase_database = dbutils.widgets.get('lakebase_database')
lakebase_schema = dbutils.widgets.get('lakebase_schema')
copy_tables = dbutils.widgets.get('copy_tables').lower() == 'true'

required = {
    'warehouse_id': warehouse_id,
    'lakebase_endpoint': lakebase_endpoint,
    'lakebase_host': lakebase_host,
}
missing = [name for name, value in required.items() if not value]
if missing:
    raise ValueError(f'Set these widgets: {", ".join(missing)}')

# COMMAND ----------

workspace = WorkspaceClient()
odbc = workspace.warehouses.get(warehouse_id).odbc_params
warehouse = dbsql.connect(
    server_hostname=odbc.hostname,
    http_path=odbc.path,
    credentials_provider=lambda: workspace.config.authenticate,
)

# The OAuth token is generated in memory and is never placed in a widget or
# notebook output. The caller needs a Lakebase OAuth role on this endpoint.
credential = workspace.postgres.generate_database_credential(
    endpoint=lakebase_endpoint
)
current_user = workspace.current_user.me().user_name
lakebase = connect(
    'postgres',
    host=lakebase_host,
    dbname=lakebase_database,
    user=current_user,
    password=credential.token,
    sslmode='require',
)


def source_query(sql: str) -> list[tuple]:
    with warehouse.cursor() as cursor:
        cursor.execute(sql)
        return [tuple(row) for row in cursor.fetchall()]


def lakebase_query(sql: str) -> list[tuple]:
    with lakebase.cursor() as cursor:
        cursor.execute(sql)
        return [tuple(row) for row in cursor.fetchall()]


def uc_name(table: str) -> str:
    return '.'.join(
        quote_ident(name, '`')
        for name in (source_catalog, source_schema, table)
    )


def pg_name(table: str) -> str:
    return '.'.join(quote_ident(name) for name in (lakebase_schema, table))


TABLE_SPECS = [
    {'name': 'customers', 'source_name': 'adventureworks_customers'},
    {'name': 'products', 'source_name': 'adventureworks_products'},
    {
        'name': 'sales_order_headers',
        'source_name': 'adventureworks_sales_order_headers',
    },
    {
        'name': 'sales_order_details',
        'source_name': 'adventureworks_sales_order_details',
    },
]
SOURCE_TABLES = [spec['source_name'] for spec in TABLE_SPECS]

# COMMAND ----------


def source_columns(table: str) -> list[tuple[str, str]]:
    rows = source_query(f'DESCRIBE TABLE {uc_name(table)}')
    return [
        (name, dtype)
        for name, dtype, *_ in rows
        if name and not name.startswith('#')
    ]


def postgres_type(databricks_type: str) -> str:
    dtype = databricks_type.strip().lower()
    if dtype in {'tinyint', 'smallint', 'int', 'integer'}:
        return 'integer'
    if dtype in {'bigint', 'long'}:
        return 'bigint'
    if dtype in {'float', 'double', 'real'}:
        return 'double precision'
    if re.fullmatch(r'(decimal|numeric)\(\d+,\s*\d+\)', dtype):
        return dtype.replace('decimal', 'numeric', 1)
    if dtype in {'boolean', 'bool'}:
        return 'boolean'
    if dtype == 'date':
        return 'date'
    if dtype.startswith('timestamp'):
        return 'timestamp'
    if dtype.startswith('binary'):
        return 'bytea'
    if dtype.startswith(('string', 'varchar', 'char')):
        return 'text'
    raise ValueError(f'No PostgreSQL mapping for Databricks type {dtype!r}')


def copy_table(table: str, batch_size: int = 5_000) -> int:
    columns = source_columns(table)
    definitions = ', '.join(
        f'{quote_ident(name)} {postgres_type(dtype)}' for name, dtype in columns
    )
    names = ', '.join(quote_ident(name) for name, _ in columns)
    source_names = ', '.join(quote_ident(name, '`') for name, _ in columns)
    with lakebase.cursor() as cursor:
        cursor.execute(f'CREATE TABLE {pg_name(table)} ({definitions})')

    copied = 0
    with (
        warehouse.cursor() as source_cursor,
        lakebase.cursor() as target_cursor,
    ):
        source_cursor.execute(f'SELECT {source_names} FROM {uc_name(table)}')
        with target_cursor.copy(
            f'COPY {pg_name(table)} ({names}) FROM STDIN'
        ) as copy:
            while rows := source_cursor.fetchmany(batch_size):
                for row in rows:
                    copy.write_row(tuple(row))
                copied += len(rows)
    return copied


def add_keys_and_indexes() -> None:
    customers = pg_name('adventureworks_customers')
    products = pg_name('adventureworks_products')
    headers = pg_name('adventureworks_sales_order_headers')
    details = pg_name('adventureworks_sales_order_details')
    statements = [
        f'ALTER TABLE {customers} ADD PRIMARY KEY ("CustomerID")',
        f'ALTER TABLE {products} ADD PRIMARY KEY ("ProductID")',
        (
            f'ALTER TABLE {headers} ADD PRIMARY KEY ("SalesOrderID"), '
            f'ADD FOREIGN KEY ("CustomerID") REFERENCES {customers} '
            '("CustomerID")'
        ),
        (
            f'ALTER TABLE {details} '
            'ADD PRIMARY KEY ("SalesOrderDetailID"), '
            f'ADD FOREIGN KEY ("SalesOrderID") REFERENCES {headers} '
            '("SalesOrderID"), '
            f'ADD FOREIGN KEY ("ProductID") REFERENCES {products} '
            '("ProductID"), '
            f'ADD FOREIGN KEY ("CustomerID") REFERENCES {customers} '
            '("CustomerID")'
        ),
        f'CREATE INDEX ON {headers} ("CustomerID")',
        f'CREATE INDEX ON {details} ("SalesOrderID")',
        f'CREATE INDEX ON {details} ("ProductID")',
        f'CREATE INDEX ON {details} ("CustomerID")',
    ]
    with lakebase.cursor() as cursor:
        for statement in statements:
            cursor.execute(statement)


if copy_tables:
    started = time.perf_counter()
    try:
        with lakebase.cursor() as cursor:
            cursor.execute(
                f'CREATE SCHEMA IF NOT EXISTS {quote_ident(lakebase_schema)}'
            )
            for table in reversed(SOURCE_TABLES):
                cursor.execute(f'DROP TABLE IF EXISTS {pg_name(table)} CASCADE')
        for table in SOURCE_TABLES:
            count = copy_table(table)
            print(f'{table}: copied {count:,} rows')
        add_keys_and_indexes()
        lakebase.commit()
    except BaseException:
        lakebase.rollback()
        raise
    print(f'copy completed atomically in {time.perf_counter() - started:.1f}s')

# COMMAND ----------


def equivalent(left: object, right: object) -> bool:
    if isinstance(left, float):
        return math.isclose(left, float(right), abs_tol=1e-3)
    if isinstance(left, datetime):
        left_time = pd.Timestamp(left).tz_localize(None)
        right_time = pd.Timestamp(right).tz_localize(None)
        return left_time == right_time
    return left == right


PARITY_QUERIES = {
    'customers': (
        'COUNT(*), MIN(CustomerID), MAX(CustomerID), SUM(CustomerID)',
        'COUNT(*), MIN("CustomerID"), MAX("CustomerID"), SUM("CustomerID")',
        'adventureworks_customers',
    ),
    'products': (
        'COUNT(*), MIN(ProductID), MAX(ProductID), SUM(ProductID)',
        'COUNT(*), MIN("ProductID"), MAX("ProductID"), SUM("ProductID")',
        'adventureworks_products',
    ),
    'headers': (
        'COUNT(*), MIN(SalesOrderID), MAX(SalesOrderID), '
        'SUM(SalesOrderID), ROUND(SUM(TotalDue), 4)',
        'COUNT(*), MIN("SalesOrderID"), MAX("SalesOrderID"), '
        'SUM("SalesOrderID"), ROUND(SUM("TotalDue")::numeric, 4)',
        'adventureworks_sales_order_headers',
    ),
    'details': (
        'COUNT(*), MIN(SalesOrderDetailID), MAX(SalesOrderDetailID), '
        'SUM(SalesOrderDetailID), SUM(OrderQty), ROUND(SUM(LineTotal), 4), '
        'MIN(OrderDate), MAX(OrderDate)',
        'COUNT(*), MIN("SalesOrderDetailID"), MAX("SalesOrderDetailID"), '
        'SUM("SalesOrderDetailID"), SUM("OrderQty"), '
        'ROUND(SUM("LineTotal")::numeric, 4), '
        'MIN("OrderDate"), MAX("OrderDate")',
        'adventureworks_sales_order_details',
    ),
}

parity_rows = []
for label, (
    source_projection,
    target_projection,
    table,
) in PARITY_QUERIES.items():
    source_value = source_query(
        f'SELECT {source_projection} FROM {uc_name(table)}'
    )[0]
    target_value = lakebase_query(
        f'SELECT {target_projection} FROM {pg_name(table)}'
    )[0]
    passed = all(
        equivalent(left, right)
        for left, right in zip(source_value, target_value)
    )
    parity_rows.append((label, passed, source_value, target_value))
    if not passed:
        raise AssertionError((label, source_value, target_value))
display(
    pd.DataFrame(
        parity_rows,
        columns=['table', 'passed', 'source', 'lakebase'],
    )
)

# COMMAND ----------

source_graph = relational.Graph.from_databricks(
    connection=warehouse,
    catalog=source_catalog,
    schema=source_schema,
    tables=TABLE_SPECS,
    verbose=False,
)
lakebase_graph = relational.Graph.from_postgres(
    connection=lakebase,
    schema=lakebase_schema,
    tables=TABLE_SPECS,
    verbose=False,
)

for graph in (source_graph, lakebase_graph):
    graph['sales_order_headers'].time_column = 'OrderDate'
    graph['sales_order_details'].time_column = 'OrderDate'
    graph['customers'].time_column = None
    graph['products'].time_column = None

# Semantic-type inference samples rows and can legitimately see a different
# sample on each engine. Copy the source graph's reviewed metadata so backend
# selection is the only variable in the prediction comparison.
for table_name in source_graph.tables:
    source_table = source_graph[table_name]
    target_table = lakebase_graph[table_name]
    if [column.name for column in source_table.columns] != [
        column.name for column in target_table.columns
    ]:
        raise AssertionError(f'column mismatch for {table_name}')
    for column in source_table.columns:
        target_column = target_table[column.name]
        if target_column.dtype != column.dtype:
            raise AssertionError(
                f'dtype mismatch for {table_name}.{column.name}'
            )
        target_column.stype = column.stype
    source_table.remove_column('rowguid')
    target_table.remove_column('rowguid')

for graph in (source_graph, lakebase_graph):
    graph.validate()

source_edges = {tuple(edge) for edge in source_graph.edges}
lakebase_edges = {tuple(edge) for edge in lakebase_graph.edges}
if source_edges != lakebase_edges:
    raise AssertionError((source_edges, lakebase_edges))
print('column, dtype, semantic-type, primary-key, and graph-link parity passed')

# Compare the exact 50 rows each sampler will consider first for the notebook's
# top-selling product. Both backends order ties by SalesOrderDetailID.
top_product = source_query(
    f'SELECT ProductID, SUM(OrderQty) AS qty '
    f'FROM {uc_name("adventureworks_sales_order_details")} '
    'GROUP BY ProductID ORDER BY qty DESC, ProductID LIMIT 1'
)[0][0]
source_sample = source_query(
    'SELECT SalesOrderID, SalesOrderDetailID, OrderQty, ProductID, '
    f'UnitPrice, LineTotal, CustomerID, OrderDate FROM '
    f'{uc_name("adventureworks_sales_order_details")} '
    f'WHERE ProductID = {int(top_product)} '
    'ORDER BY OrderDate DESC, SalesOrderDetailID LIMIT 50'
)
lakebase_sample = lakebase_query(
    'SELECT "SalesOrderID", "SalesOrderDetailID", "OrderQty", '
    '"ProductID", "UnitPrice", "LineTotal", "CustomerID", '
    f'"OrderDate" FROM {pg_name("adventureworks_sales_order_details")} '
    f'WHERE "ProductID" = {int(top_product)} '
    'ORDER BY "OrderDate" DESC, "SalesOrderDetailID" LIMIT 50'
)
samples_match = len(source_sample) == len(lakebase_sample) and all(
    equivalent(source_value, lakebase_value)
    for source_row, lakebase_row in zip(source_sample, lakebase_sample)
    for source_value, lakebase_value in zip(source_row, lakebase_row)
)
if not samples_match:
    raise AssertionError('deterministic sampled rows differ')
print(f'sampled-row parity passed for ProductID={top_product}')

# COMMAND ----------

print(
    'Lakebase graph is ready. Use lakebase_graph with the prediction workload '
    'from kumo_relational_engine_adventureworks_databricks.ipynb.'
)

# COMMAND ----------

warehouse.close()
lakebase.close()

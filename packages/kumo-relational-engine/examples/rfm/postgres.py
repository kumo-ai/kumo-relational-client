# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from kumo_connectors import connect
from kumo_relational_client import RelationalClient, relational

# PGHOST, PGDATABASE, PGUSER, PGPASSWORD, and PGSSLMODE are read
# automatically. A PostgreSQL URI or explicit connection keywords work too.
connection = connect('postgres')
graph = relational.Graph.from_postgres(
    connection,
    schema=os.getenv('PGSCHEMA', 'public'),
    tables=['customers', 'orders'],
)

with connection.cursor() as cursor:
    cursor.execute('SELECT customer_id FROM customers LIMIT 10')
    indices = [row[0] for row in cursor.fetchall()]

query = 'PREDICT SUM(orders.amount, 0, 30, days) FOR customers.customer_id'
with RelationalClient(url=os.environ['KUMO_RELATIONAL_API_ENDPOINT']) as client:
    prediction = client.relational(graph).predict(query, indices=indices)
print(prediction)

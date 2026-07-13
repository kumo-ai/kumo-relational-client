import time

from kumorfm import rfm
from kumorfm.testing.databricks import connect

rfm.init()

connection = connect(
    catalog='kumo_test_catalogue',
    schema='nvidia_erp',
)

t = time.perf_counter()
graph = rfm.Graph.from_databricks(
    connection,
    tables=['customers', 'order_lines', 'products'],
)
print(f'Created graph in {time.perf_counter() - t:.2f} seconds')

with connection.cursor() as cursor:
    cursor.execute("SELECT customer_id FROM customers LIMIT 10")
    indices = [row[0] for row in cursor.fetchall()]

model = rfm.KumoRFM(graph)

query = ("PREDICT COUNT(order_lines.*, 0, 30)=0 "
         "FOR EACH customers.customer_id")
# `order_lines` is large (~1.4M rows), so cap the per-hop neighbor fan-out to
# keep the in-context tables under the request-size limit:
pred = model.predict(query, indices, run_mode='fast', num_neighbors=[4])
print(pred)

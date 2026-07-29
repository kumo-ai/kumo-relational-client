# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time

from kumorfm import rfm
from kumorfm.testing.databricks import connect

from nvidia_sdfm import KumoRFMRequest, SDFMClient

nim_url = os.environ.get('RFM_API_URL') or os.environ['KUMO_API_ENDPOINT']

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

query = ("PREDICT COUNT(order_lines.*, 0, 30)=0 "
         "FOR EACH customers.customer_id")
# `order_lines` is large (~1.4M rows), so cap the per-hop neighbor fan-out to
# keep the in-context tables under the request-size limit:
request = KumoRFMRequest(
    graph=graph,
    query=query,
    indices=indices,
    run_mode='fast',
    options={'num_neighbors': [4]},
)
with SDFMClient(url=nim_url) as client:
    pred = client.predict(request)
print(pred)

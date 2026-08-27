---
title: "Quickstart for the NVIDIA Kumo Relational Client"
description: "Run your first Kumo Relational prediction against a Universal TFM API NIM using the NVIDIA Kumo Relational Client."
template-library-version: "1.0.0"
---

# Quickstart for the NVIDIA Kumo Relational Client

This quickstart shows you how to connect to a NIM and run a prediction with each
model. A `RelationalClient` owns one connection to a NIM, and each model has its own
typed handle.

## Prerequisites

Before you start, you must complete the following prerequisites:

1. Install the client. For the relational example, install the Kumo Relational extra:
   `pip install "kumo-relational-client[relational]"`.
2. Identify the URL of a running Universal TFM API NIM.

## Quickstart Steps

1. Create a `RelationalClient` pointed at your NIM.
2. Build a model handle with `client.relational(...)`.
3. Call `predict` and read the returned pandas DataFrame.

## Minimal Code Example

### Kumo Relational: Relational Data

Build a graph from related tables, then express the target in PQL.

This example runs as written. It needs the `[relational]` and `[relbench]`
extras, and it downloads a ready-made dataset the first time you run it:

```bash
pip install "kumo-relational-client[relational,relbench]"
```

```python
from kumo_relational_client import RelationalClient, relational

# Nine tables of Formula 1 history. Links between them are inferred.
graph = relational.Graph.from_relbench('f1')

with RelationalClient(url='http://localhost:8000') as client:
    # Will each of these drivers race more than three times in the next 90 days?
    predictions = client.relational(graph).predict(
        'PREDICT COUNT(results.*, 0, 90, days) > 3 FOR EACH drivers.driverId',
        indices=[814, 0, 842, 831, 3, 829],
    )

print(predictions)
```

```text
   ENTITY          ANCHOR_TIMESTAMP  PREDICTION  FALSE_PROB  TRUE_PROB
0     814 2023-07-30 13:00:00+00:00        True    0.014900   0.985100
1       0 2023-07-30 13:00:00+00:00        True    0.024704   0.975296
2     842 2023-07-30 13:00:00+00:00        True    0.004905   0.995095
```

`indices` selects which entities to score. The ids above are drivers who were
racing at the end of the dataset; RelBench renumbers ids from zero, so they are
not the ids used by the original Formula 1 data.

To use your own tables instead, pass DataFrames to
`relational.Graph.from_data({'users': users_df, 'orders': orders_df})`. Choose an
aggregation window your data can support: a query over `0, 90, days` needs at
least 90 days of history before the anchor time, or the request is rejected.

### Check Which Models the Client Can Serve

Both describe the client itself, not the endpoint it points at.
Neither contacts the NIM, so they answer before you have one running, and a NIM
serving only one of these models still reports both.

```python
from kumo_relational_client import RelationalClient

with RelationalClient(url='http://localhost:8000') as client:
    print(client.models())  # ['kumo-relational']
    print(client.capabilities('kumo-relational'))  # tasks and outputs the model supports
```

To check the endpoint itself, call `client.health_ready()`. It returns whether
the NIM answered `GET /v1/health/ready` with 200, and raises `RelationalError` with
code `TRANSPORT_ERROR` if the endpoint cannot be reached at all.

## Next Steps

- Learn how the pieces fit together in the [NVIDIA Kumo Relational Client Architecture](../about/architecture.md).
- Configure the client and drivers with the [NVIDIA Kumo Relational Client Environment Variables](../reference/environment-variables.md).

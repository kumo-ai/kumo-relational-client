---
title: "Quickstart for the NVIDIA SDFM SDK"
description: "Run your first Nemotron Tabular and Nemotron Relational predictions against a Universal TFM API NIM using the NVIDIA SDFM SDK."
template-library-version: "1.0.0"
---

# Quickstart for the NVIDIA SDFM SDK

This quickstart shows you how to connect to a NIM and run a prediction with each
model. A `PredictClient` owns one connection to a NIM, and each model has its own
typed handle.

## Prerequisites

Before you start, you must complete the following prerequisites:

1. Install the SDK. For the relational example, install the Nemotron Relational extra:
   `pip install "nemotron-predict-client[relational]"`.
2. Identify the URL of a running Universal TFM API NIM.

## Quickstart Steps

1. Create a `PredictClient` pointed at your NIM.
2. Build a model handle with `client.tabular(...)` or `client.relational(...)`.
3. Call `predict` and read the returned pandas DataFrame.

## Minimal Code Example

### Nemotron Tabular: Single Table

Provide labeled context rows and rows to predict:

```python
from nemotron_predict import PredictClient

# Connect to the NIM; the client is a context manager.
with PredictClient(url='http://localhost:8000') as client:
    # Build a Nemotron Tabular handle bound to the labeled context table.
    model = client.tabular(context_df, target='label', task='classification')
    # Score the unlabeled rows.
    predictions = model.predict(
        predict_df, outputs=['prediction', 'probabilities']
    )

print(predictions.head())
```

### Nemotron Relational: Relational Data

Build a graph from related tables, then express the target in PQL. This example
requires the `[relational]` extra:

```python
from nemotron_predict import PredictClient, relational

# Build a graph from related DataFrames; links are inferred.
graph = relational.Graph.from_data({'users': df1, 'items': df2, 'orders': df3})

with PredictClient(url='http://localhost:8000') as client:
    # Predict a per-item quantity over the next 30 days.
    predictions = client.relational(graph).predict(
        'PREDICT SUM(orders.price, 0, 30, days) FOR items.item_id=1',
        run_mode='fast',
    )

print(predictions.head())
```

### Check Which Models the Client Can Serve

Both calls describe the client's own adapters, not the endpoint it points at.
Neither contacts the NIM, so they answer before you have one running, and a NIM
serving only one of these models still reports both.

```python
from nemotron_predict import PredictClient

with PredictClient(url='http://localhost:8000') as client:
    print(client.models())  # ['nemotron-relational', 'nemotron-tabular']
    print(client.capabilities('nemotron-tabular'))  # tasks and outputs the model supports
```

To check the endpoint itself, call `client.health_ready()`. It returns whether
the NIM answered `GET /v1/health/ready` with 200, and raises `PredictError` with
code `TRANSPORT_ERROR` if the endpoint cannot be reached at all.

## Next Steps

- Learn how the pieces fit together in the [NVIDIA SDFM SDK Architecture](../about/architecture.md).
- Configure the client and drivers with the [NVIDIA SDFM SDK Environment Variables](../reference/environment-variables.md).

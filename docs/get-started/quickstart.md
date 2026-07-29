---
title: "Quickstart for the NVIDIA SDFM SDK"
description: "Run your first TabICL and KumoRFM predictions against a Universal TFM API NIM using the NVIDIA SDFM SDK."
template-library-version: "1.0.0"
---

# Quickstart for the NVIDIA SDFM SDK

This quickstart shows you how to connect to a NIM and run a prediction with each
model. A `SDFMClient` owns one connection to a NIM; requests are typed per model,
so the client validates them before sending.

## Prerequisites

Before you start, you must complete the following prerequisites:

1. Install the SDK. For the relational example, install the KumoRFM extra:
   `pip install "nvidia-sdfm[kumorfm]" --index-url https://pypi.org/simple`.
2. Identify the URL of a running Universal TFM API NIM.

## Quickstart Steps

1. Create a `SDFMClient` pointed at your NIM.
2. Build a model handle with `client.tabicl(...)` or `client.kumorfm(...)`.
3. Call `predict` and read the returned pandas DataFrame.

## Minimal Code Example

### TabICL: Single Table

Provide labeled context rows and rows to predict:

```python
from nvidia_sdfm import SDFMClient

# Connect to the NIM; the client is a context manager.
with SDFMClient(url="http://localhost:8000") as client:
    # Build a TabICL handle bound to the labeled context table.
    model = client.tabicl(context_df, target="label", task="classification")
    # Score the unlabeled rows.
    predictions = model.predict(predict_df, outputs=["prediction", "probabilities"])

print(predictions.head())
```

### KumoRFM: Relational Data

Build a graph from related tables, then express the target in PQL. This example
requires the `[kumorfm]` extra:

```python
from nvidia_sdfm import SDFMClient, kumorfm

# Build a graph from related DataFrames; links are inferred.
graph = kumorfm.Graph.from_data({"users": df1, "items": df2, "orders": df3})

with SDFMClient(url="http://localhost:8000") as client:
    # Predict a per-item quantity over the next 30 days.
    predictions = client.kumorfm(graph).predict(
        "PREDICT SUM(orders.price, 0, 30, days) FOR items.item_id=1",
        run_mode="fast",
    )

print(predictions.head())
```

### Discover What a NIM Serves

```python
from nvidia_sdfm import SDFMClient

with SDFMClient(url="http://localhost:8000") as client:
    print(client.models())              # e.g. ['kumo-rfm', 'tabicl']
    print(client.capabilities("tabicl"))  # tasks and outputs the model supports
```

## Next Steps

- Learn how the pieces fit together in the [NVIDIA SDFM SDK Architecture](../about/architecture.md).
- Configure the client and drivers with the [NVIDIA SDFM SDK Environment Variables](../reference/environment-variables.md).

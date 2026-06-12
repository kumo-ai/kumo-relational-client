# KumoRFM SDK

This repository contains the RFM-only subset of the Kumo Python SDK. It keeps
the documented import path:

```python
import kumoai.experimental.rfm as rfm
```

The package includes the RFM graph/table abstractions, local/sqlite/duckdb/
Snowflake backends, SageMaker adapters, the RFM HTTP client, and the native
local neighbor sampler. It intentionally excludes the Kumo Enterprise
fine-tuning SDK surfaces such as connectors, training jobs, the separate
`kumoai.pquery` management API, codegen, online serving, artifact export, and
session management.

## Install

```bash
pip install -e .
```

For local development without GitHub SSH access to `kumo-api`, point the
installer at a local checkout:

```bash
KUMO_API_PATH=/path/to/kumo-api pip install -e .
```

The native sampler is built through CMake/scikit-build. Set
`WITH_KUMOLIB=0` only for metadata-only workflows that do not import or run
local RFM backends.

## Quick Start

```python
import os
import pandas as pd
import kumoai.experimental.rfm as rfm

os.environ["KUMO_API_KEY"] = "ENTER_YOUR_API_KEY_HERE"
rfm.init()

graph = rfm.LocalGraph.from_data({
    "users": pd.DataFrame(...),
    "items": pd.DataFrame(...),
    "orders": pd.DataFrame(...),
})

model = rfm.KumoRFM(graph)
result = model.predict(
    "PREDICT SUM(orders.price, 0, 30, days) FOR items.item_id=1"
)
```

Public quick-start documentation:
https://kumo.ai/docs/quick-start/rfm/

## Tests

```bash
pytest test/rfm
```

Backend-specific tests require their optional dependencies, for example
`.[sqlite]`, `.[duckdb]`, `.[snowflake]`, or `.[sagemaker]`.

# KumoRFM SDK

This repository contains the RFM-only subset of the Kumo Python SDK. It keeps
the documented import path:

```python
import kumoai.rfm as rfm
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

For local development without GitHub SSH access to `kumo-api`, use the
published `kumo-api` wheel instead of the source tag:

```bash
KUMO_SDK_RELEASE=1 python -m pip install -e .
```

Use `KUMO_API_PATH=/path/to/kumo-api python -m pip install -e .` only when
intentionally testing a local `kumo-api` source checkout.

The native sampler is built through CMake/scikit-build. Set
`WITH_KUMOLIB=0` only for metadata-only workflows that do not import or run
local RFM backends.

### Native sampler build notes

The `kumoai.kumolib` module is built from this repository, not from
`kumo-api`. `CMakeLists.txt` compiles `kumoai/csrc/neighbor_sampler.cpp` into
the `kumolib` pybind11 extension, and `setup.py` enables that native build by
default unless `WITH_KUMOLIB=0` is set. Local backend imports such as
`kumoai.rfm.backend.local` require the compiled extension to be present.

`kumo-api` v0.92.0 supports Python 3.10 through 3.14 as an installed wheel.
If the SDK dependencies are already present, build only this package and its
native extension with:

```bash
python -m pip install -e . --no-deps
```

This avoids source-building `kumo-api` while still producing the SDK extension
artifact for the active interpreter, for example
`kumoai/kumolib.cpython-314-x86_64-linux-gnu.so`. The generated shared object
is ignored by git. Building `kumo-api` itself from source is still done on
Python 3.10 because of its protobuf generation toolchain, and source builds on
Python 3.14 are unsupported.

## Quick Start

```python
import os
import pandas as pd
import kumoai.rfm as rfm

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
pytest test
```

### RFM NIM Contract And Live Tests

The SDK-side contract tests are unit-only by default:

```bash
python -m pytest test/client/test_rfm_nim_contract.py
```

Live Kumo RFM NIM probes remain opt-in and accept an existing container URL.
The repeatable runner creates or repairs an ignored, repo-local virtualenv and
runs a fast SDK/NIM boundary smoke suite by default:

```bash
scripts/run_rfm_nim_live_tests.sh --url http://127.0.0.1:8002
```

Run the broader task, invalid-request recovery, and session suite explicitly:

```bash
scripts/run_rfm_nim_live_tests.sh --url http://127.0.0.1:8002 --full
```

To test a container running on a Colossus host port such as `8002`, forward the
remote port to the local machine first:

```bash
ssh <user>@<colossus-host> -N -L 8002:127.0.0.1:8002
```

Direct pytest usage is still supported. The URL is required for live tests;
without it, they are skipped as part of ordinary unit-test collection:

```bash
export RFM_NIM_BASE_URL=http://127.0.0.1:8002
python -m pytest test/client/test_rfm_nim_live.py \
  -m 'live_nim_smoke or live_nim_full'
```

Set `RFM_NIM_API_KEY` when the deployment requires `X-API-Key` authentication.
`RFM_NIM_TIMEOUT_SECONDS` changes the per-request timeout, and
`RFM_NIM_VERIFY_SSL=0` disables TLS verification for development endpoints.
The live suite validates the current `/v1/*` Universal TFM boundary using
semantic response and problem-details invariants; it intentionally does not
pin model scores, backend metadata, or implementation-specific error text.

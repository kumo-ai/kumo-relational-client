# nemotron_relational

The NemotronRelational driver for the [`nemotron-predict-client`](../nemotron-predict-client/README.md) SDK.

This distribution provides the heavy, client-side machinery a NemotronRelational prediction
needs before a request reaches a NIM: the relational `Graph`/`Table` abstractions,
the neighbor samplers (local native `relationallib`, plus DuckDB / SQLite / Snowflake /
Databricks backends), the PQL parser, and the HTTP client that talks to a Universal
TFM NIM.

It is imported as `nemotron_relational` and is normally installed transitively via the SDK's
`nemotron_relational` extra rather than on its own:

```bash
pip install nemotron-predict-client[nemotron_relational]      # pulls the nemotron_relational driver
```

It can also be installed on its own, for the graph, sampler and PQL machinery.
Predicting is not available that way: the engine's entry points refuse a direct
call, so a prediction has to go through `nemotron-predict-client`.

```bash
pip install nemotron_relational
# optional data backends:
pip install "nemotron-relational-v1[duckdb]"      # or [sqlite] / [snowflake] / [databricks]
# optional features:
pip install "nemotron-relational-v1[explain]"     # natural-language explanation summaries
pip install "nemotron-relational-v1[relbench]"    # Graph.from_relbench() dataset loading
pip install "nemotron-relational-v1[codegen]"     # regenerate the TFM API client (see scripts/)
```

Release wheels are built for CPython 3.10, 3.11, 3.12 and 3.13
(`manylinux_2_28` x86-64); no source distribution is published.

Application code should import the SDK's neutral surface, not this package
directly:

```python
from nemotron_predict import PredictClient, relational

graph = relational.Graph.from_data({'users': df1, 'items': df2, 'orders': df3})
with PredictClient(url='http://localhost:8000') as client:
    result = client.relational(graph).predict(
        'PREDICT SUM(orders.price, 0, 30, days) FOR items.item_id=1'
    )
```

## Local development

```bash
pip install -e ".[test,databricks]"
pytest tests -m "not live_nim"
```

The live-NIM suite (`scripts/run_rfm_nim_live_tests.sh --full`) includes durable,
ticket-free rejection regressions for fixed NIM defects; every rejection probe is
followed by a known-good prediction to detect service degradation.

The native neighbor sampler (`nemotron_relational.relationallib`) is built from
`src/nemotron_relational/csrc/neighbor_sampler.cpp` via CMake/scikit-build-core. Set
`WITH_KUMOLIB=0` to skip the native build for a pure-Python editable install.

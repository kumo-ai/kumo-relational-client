# kumorfm

The KumoRFM driver for the [`nvidia-sdfm`](../nvidia-sdfm/README.md) SDK.

This distribution provides the heavy, client-side machinery a KumoRFM prediction
needs before a request reaches a NIM: the relational `Graph`/`Table` abstractions,
the neighbor samplers (local native `kumolib`, plus DuckDB / SQLite / Snowflake /
Databricks backends), the PQL parser, and the HTTP client that talks to a Universal
TFM NIM.

It is imported as `kumorfm` and is normally installed transitively via the SDK's
`kumorfm` extra rather than on its own:

```bash
pip install nvidia-sdfm[kumorfm]      # pulls the kumorfm driver
```

It can also be installed on its own, for the graph, sampler and PQL machinery.
Predicting is not available that way: the engine's entry points refuse a direct
call, so a prediction has to go through `nvidia-sdfm`.

```bash
pip install kumorfm
# optional data backends:
pip install "kumorfm[duckdb]"      # or [sqlite] / [snowflake] / [databricks]
# optional features:
pip install "kumorfm[explain]"     # natural-language explanation summaries
pip install "kumorfm[relbench]"    # Graph.from_relbench() dataset loading
pip install "kumorfm[codegen]"     # regenerate the TFM API client (see scripts/)
```

Release wheels are built for CPython 3.10, 3.11 and 3.12 (`manylinux_2_28`
x86-64); no source distribution is published.

Application code should import the SDK's neutral surface, not this package
directly:

```python
from nvidia_sdfm import SDFMClient, kumorfm

graph = kumorfm.Graph.from_data({"users": df1, "items": df2, "orders": df3})
with SDFMClient(url="http://localhost:8000") as client:
    result = client.kumorfm(graph).predict(
        "PREDICT SUM(orders.price, 0, 30, days) FOR items.item_id=1")
```

## Local development

```bash
pip install -e ".[test,databricks]"
pytest tests -m "not live_nim"
```

The live-NIM suite (`scripts/run_rfm_nim_live_tests.sh --full`) includes durable,
ticket-free rejection regressions for fixed NIM defects; every rejection probe is
followed by a known-good prediction to detect service degradation.

The native neighbor sampler (`kumorfm.kumolib`) is built from
`src/kumorfm/csrc/neighbor_sampler.cpp` via CMake/scikit-build-core. Set
`WITH_KUMOLIB=0` to skip the native build for a pure-Python editable install.

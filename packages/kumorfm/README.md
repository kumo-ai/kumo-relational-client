# kumorfm

The KumoRFM driver for the [`nvidia-sdfm`](../../README.md) SDK.

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

Direct install (e.g. for benchmarking or reuse outside the SDK):

```bash
pip install kumorfm
# optional data backends:
pip install "kumorfm[duckdb]"      # or [sqlite] / [snowflake] / [databricks]
```

Application code should import the SDK's neutral surface, not this package
directly:

```python
from nvidia_sdfm import kumorfm

graph = kumorfm.LocalGraph.from_data({"users": df1, "items": df2, "orders": df3})
model = kumorfm.KumoRFM(graph)
result = model.predict("PREDICT SUM(orders.price, 0, 30, days) FOR items.item_id=1")
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

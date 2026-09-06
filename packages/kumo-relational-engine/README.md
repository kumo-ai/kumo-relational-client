# kumo_relational_engine

The Kumo Relational driver for the [`kumo-relational-client`](../kumo-relational-client/README.md) client.

This distribution provides the heavy, client-side machinery a Kumo Relational prediction
needs before a request reaches a NIM: the relational `Graph`/`Table` abstractions,
the neighbor samplers (local native `relationallib`, plus DuckDB / SQLite / Snowflake /
Databricks / PostgreSQL backends), the PQL parser, and the HTTP client that talks to a Universal
TFM NIM.

It is imported as `kumo_relational_engine` and is normally installed transitively via the client's
`relational` extra rather than on its own:

```bash
pip install kumo-relational-client[relational]      # pulls the kumo_relational_engine driver
```

It can also be installed on its own, for the graph, sampler and PQL machinery.
Predicting is not available that way: the engine's entry points refuse a direct
call, so a prediction has to go through `kumo-relational-client`.

```bash
pip install kumo-relational-engine
# optional data backends:
pip install "kumo-relational-engine[duckdb]"      # or [sqlite] / [snowflake] / [databricks] / [postgres]
# optional features:
pip install "kumo-relational-engine[explain]"     # natural-language explanation summaries
pip install "kumo-relational-engine[relbench]"    # Graph.from_relbench() dataset loading
pip install "kumo-relational-engine[codegen]"     # regenerate the TFM API client (see scripts/)
```

`Graph.from_postgres()` reads existing PostgreSQL tables:

```python
import psycopg
from kumo_relational_client import relational

connection = psycopg.connect(
    host=host,
    dbname=database,
    user=user,
    password=credential,
    sslmode='require',
)
graph = relational.Graph.from_postgres(connection, schema='public')
```

`schema` is the table namespace; omit it to use `current_schema()` (`public` is
the usual default). `connection` may instead be a PostgreSQL URI/libpq string
or a dictionary of psycopg arguments; if omitted, psycopg uses `PG*` environment
variables. See [`postgres.py`](examples/rfm/postgres.py). Lakebase uses this same
API through its PostgreSQL endpoint with TLS and either an OAuth token or an
enabled native Postgres password. It reads existing tables directly—without
Unity Catalog registration, copying data, or DDL.

Temporal sampling returns at most the requested latest N rows at or before each
anchor, ordered by time descending then primary key ascending. For large fact
tables, an index on `(foreign_key, time_column DESC, primary_key)` improves this
pushed-down query; the SDK never creates indexes or modifies source tables.

Release wheels are built for CPython 3.10, 3.11, 3.12 and 3.13
(`manylinux_2_28` x86-64), plus CPython 3.12 on Linux ARM64. No source
distribution is published.

Application code should import the client's neutral surface, not this package
directly:

```python
from kumo_relational_client import RelationalClient, relational

graph = relational.Graph.from_data({'users': df1, 'items': df2, 'orders': df3})
with RelationalClient(url='http://localhost:8000') as client:
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

The native neighbor sampler (`kumo_relational_engine.relationallib`) is built from
`src/kumo_relational_engine/csrc/neighbor_sampler.cpp` via CMake/scikit-build-core. Set
`WITH_RELATIONALLIB=0` to skip the native build for a pure-Python editable install.

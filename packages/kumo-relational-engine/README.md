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

Create a graph from PostgreSQL with `Graph.from_postgres()`:

```python
from kumo_relational_client import relational

graph = relational.Graph.from_postgres(schema='public')
```

A PostgreSQL schema is a namespace that groups tables within a database.
`public` is the usual default; replace it with your table namespace, or omit
`schema` to use the connection's current schema. Connection details can be an
open psycopg connection, a URI/libpq conninfo string, explicit keywords, or
standard `PG*` environment variables. See the generic
[`postgres.py`](examples/rfm/postgres.py) example.

Lakebase exposes a PostgreSQL endpoint, so tables that already exist there use
the same API. Obtain a database credential using the Databricks-supported
authentication flow, open a psycopg connection, and pass it to
`Graph.from_postgres()`:

```python
import psycopg
from kumo_relational_client import relational

connection = psycopg.connect(
    host=lakebase_host,
    dbname=database,
    user=database_user,
    password=oauth_token,
    sslmode='require',
)
graph = relational.Graph.from_postgres(
    connection=connection,
    schema='public',
)
```

This connects directly to existing tables. It does not register Lakebase in
Unity Catalog, copy data, or create schemas and tables.

Temporal neighbor sampling returns the exact latest rows at or before each
anchor time. On large fact tables, a PostgreSQL index matching
`(foreign_key, time_column DESC, primary_key)` can accelerate that query
without changing which rows are selected. The client never creates indexes or
otherwise modifies source tables.

Release wheels are built for CPython 3.10, 3.11, 3.12 and 3.13
(`manylinux_2_28` x86-64); no source distribution is published.

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

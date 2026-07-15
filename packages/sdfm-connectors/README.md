# sdfm-connectors

Shared data-source connectors for the [`nvidia-sdfm`](../README.md) SDK and its
model engines. One place that knows how to reach each warehouse, so the client
(flat table reads for TabICL) and the KumoRFM driver (warehouse connections for
its graph samplers) don't each carry their own copy.

It provides:

- `connect(backend, **kwargs)` — open a connection to `sqlite` / `duckdb` /
  `snowflake` / `databricks` (one driver per backend).
- `read(source, **kwargs) -> DataFrame` — read a table or query as a flat pandas
  DataFrame (also handles `local` DataFrames / CSV / Parquet, and `s3` object
  URIs: `read('s3', path='s3://bucket/table.parquet', storage_options=...)`).
- `read_table(connection, table=…, query=…) -> DataFrame` — driver-agnostic
  fetch over an open connection.
- `quote_ident(ident, char='"')` and `resolve_sql(table=…, query=…)` — SQL
  identifier quoting and a safe table/query guard.

Pure-python. Database drivers are optional extras:

```bash
pip install "sdfm-connectors[sqlite]"      # or [duckdb] / [snowflake] / [databricks] / [s3] / [all]
```

Consumers depend on it and surface these extras under their own name, e.g.
`nvidia-sdfm[snowflake]` and `kumorfm[databricks]` both resolve the corresponding
`sdfm-connectors` extra.

## Local development

```bash
pip install -e ".[sqlite,duckdb,test]"
pytest tests
```

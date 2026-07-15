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

Failures raised by `connect` / `read` / `read_table` surface as
`ConnectorError` with a stable `code`: `UNKNOWN_CONNECTOR`,
`INVALID_CONNECTOR_ARGS`, `CONNECT_FAILED`, `QUERY_FAILED`, `READ_FAILED`, or
`NOT_FOUND`, with the original driver exception chained as `__cause__`. Two
cases stay unwrapped by design: a missing optional driver raises
`MissingBackendError`, and a broken or incompatible driver installation raises
`ImportError` (consumers map it to their own broken-install category, e.g.
nvidia-sdfm's `DRIVER_LOAD_FAILED`).

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

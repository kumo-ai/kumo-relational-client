Vendored from `kumo-ai/kumo-pql` at commit
`6caec97943229a1e70581a4cfc465fbe37636d61`.

The package is kept under `kumorfm._vendor.kumopql` so the SDK does not depend
on an unreleased `kumo-pql` wheel. This copy is intentionally pruned to the SDK
runtime parse path: generated grammar Python modules, parser, and validators.
If `kumo-pql` becomes available as a normal runtime dependency, remove this
directory and switch `kumorfm.rfm.query_parser` back to importing from the
external package.

Local changes:

- `__init__.py` aliases the vendored package as `kumopql` in `sys.modules` so
  upstream absolute imports continue to resolve without rewriting the package.
- The `rewriting` package and non-runtime grammar source/metadata files were
  removed because `KumoRFM._parse_query()` does not use them.
- `validator/problem_type_validator.py` rejects `TOP K` unless it is used with
  `RANK`, and rejects non-positive `TOP K` values.

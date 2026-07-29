# kumorfm.pql

Internalized copy of the `kumo-pql` parser package, vendored so the SDK is
self-contained (no external `kumo-pql` dependency) for the open-source release.

- Scope: the SDK runtime parse path only — `grammar/`, `parser/`, `validator/`
  (the `rewriting/` module is not imported by the parse path and is omitted).
- Local patches applied on top of the baseline:
  1. `validator/problem_type_validator.py`: reject `TOP K` without `RANK` and
     reject non-positive `TOP K`.
  2. `validator/rfm_validator.py`: also shift the `whatif_ast` location interval
     by the query offset.
  3. `validator/type_validator.py`: map the `'Constant'` node kind to
     `ConstantNodeTypeValidator` (fixes a baseline bug that used
     `ColumnNodeTypeValidator`).
- All internal `kumopql.*` imports rewritten to `kumorfm.pql.*`, and `kumoapi.*`
  to `kumorfm.api.*`. The baseline `sys.modules['kumopql']` alias shim is dropped.

Re-sync procedure: see `scripts/sync_internal_packages.py`.

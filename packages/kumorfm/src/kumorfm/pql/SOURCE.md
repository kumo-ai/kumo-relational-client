# kumorfm.pql

Internalized copy of `kumo-pql`, vendored so the SDK is self-contained (no
external/unreleased `kumo-pql` dependency) for the open-source release.

- Source: `kumo-ai/kumo-pql`
- Commit: `4c8fc57` (latest `main`; kumo-pql is untagged upstream)
- Scope: the SDK runtime parse path only — `grammar/`, `parser/`, `validator/`
  (the `rewriting/` module is not imported by the parse path and is omitted).
- Local patches applied on top of upstream (all confirmed absent from `4c8fc57`):
  1. `validator/problem_type_validator.py`: reject `TOP K` without `RANK` and
     reject non-positive `TOP K`.
  2. `validator/rfm_validator.py`: also shift the `whatif_ast` location interval
     by the query offset.
  3. `validator/type_validator.py`: map the `'Constant'` node kind to
     `ConstantNodeTypeValidator` (upstream bug: it used `ColumnNodeTypeValidator`).
- All internal `kumopql.*` imports rewritten to `kumorfm.pql.*`, and `kumoapi.*`
  to `kumorfm.api.*`. The upstream `sys.modules['kumopql']` alias shim is dropped.

Re-sync procedure: see `scripts/sync_internal_packages.py`.

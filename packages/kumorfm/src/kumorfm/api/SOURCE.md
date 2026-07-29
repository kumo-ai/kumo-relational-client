# kumorfm.api

Internalized copy of the `kumo-api` wire-type package, vendored so the SDK is
self-contained (no external `kumo-api` dependency) for the open-source release.

- Version baseline: `0.92.0`
- Changes from the baseline:
  - All internal `kumoapi.*` imports rewritten to `kumorfm.api.*`.
  - `rfm/protos/` (protobuf `.proto` definitions) dropped: the SDK uses the
    JSON wire format only and never exercises the protobuf code paths, and the
    generated `_pb2` modules are build-time artifacts not shipped in source.

Re-sync procedure: see `scripts/sync_internal_packages.py`.

# kumorfm.api

Internalized copy of `kumo-api`, vendored so the SDK is self-contained (no
external `kumo-api` dependency) for the open-source release.

- Source: `kumo-ai/kumo-api`
- Version: `0.92.0` (tag `v0.92.0`, commit `9ee1c96`)
- Changes from upstream:
  - All internal `kumoapi.*` imports rewritten to `kumorfm.api.*`.
  - `rfm/protos/` (protobuf `.proto` definitions) dropped: the SDK uses the
    JSON wire format only and never exercises the protobuf code paths, and the
    generated `_pb2` modules are build-time artifacts not shipped in source.

Re-sync procedure: see `scripts/sync_internal_packages.py`.

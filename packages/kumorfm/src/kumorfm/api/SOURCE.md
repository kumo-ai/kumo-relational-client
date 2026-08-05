# kumorfm.api

Internalized copy of the `kumo-api` wire-type package, vendored so the SDK is
self-contained (no external `kumo-api` dependency) for the open-source release.

- Version baseline: `0.92.0`
- Changes from the baseline:
  - All internal `kumoapi.*` imports rewritten to `kumorfm.api.*`.
  - `rfm/protos/` (protobuf `.proto` definitions) dropped: the SDK uses the
    JSON wire format only and never exercises the protobuf code paths, and the
    generated `_pb2` modules are build-time artifacts not shipped in source.
  - The methods that imported `rfm/protos/` dropped with it, since every one of
    them raised `ModuleNotFoundError`: `Context.{fill_protobuf_,to_protobuf,
    serialize,from_protobuf,from_bytes,get_memory_stats}` and the parquet codec
    helpers they used (`rfm/context.py`); `InferenceConfig.{fill_protobuf,
    from_protobuf,_target_transform_from_proto}` (`rfm/inference.py`);
    `RFMPredictRequest.{to_protobuf,serialize,from_bytes}` and
    `RFMEvaluateRequest.{to_protobuf,serialize,from_bytes}` (`rfm/requests.py`).
  - Kumo Enterprise platform modules dropped as unreachable from the RFM path:
    `jobs.py`, `online_serving.py`, `distilled_model_plan.py`,
    `data_snapshot.py`, `rbac.py`. Nothing in `kumorfm`, `nvidia_sdfm` or
    `sdfm_connectors` imports them, and they describe batch jobs, serving
    endpoints, snapshots and RBAC — none of which this SDK offers.

Re-sync procedure: see `scripts/sync_internal_packages.py`. It re-applies the
module exclusions automatically and fails if a re-synced tree still references
`rfm.protos`, so the method removals above have to be re-applied by hand.


`model_plan.py`, `encoder.py` and `train.py` are excluded as well: they describe
model architectures, encoders and training jobs for a service that trains
models, and this SDK sends in-context examples to a NIM and trains nothing. So
are `rfm/pquery.py`, `rfm/explain.py` and four of the five `explain/` modules,
which describe server-side query and explanation types that nothing outside
`api/` names.

Two names from `model_plan.py` were reachable, `RunMode` and `MissingType`. They
live in `kumorfm/runmode.py`, outside this tree, because `sync_api()` replaces
`api/` wholesale and anything defined here is lost on the next sync.

The files in `API_OWNED` are the SDK's, and `sync_api()` preserves them rather
than taking upstream's copy: `rfm/__init__.py` and `explain/__init__.py` re-export
only the subset used here, and `rfm/requests.py` keeps only the prediction
request and response. An upstream change to the request envelope has to be
reviewed and applied by hand, which is the intent -- it is a wire contract.


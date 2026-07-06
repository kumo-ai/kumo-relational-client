# nvidia-sdfm

One client SDK for NVIDIA structured-data foundation model NIMs. Implements
`docs/v0_tfm_nim_release/nvidia-sdfm-sdk-proposal.md` from the `kumo-tfm-nims/docs` repo.

```python
import nvidia_sdfm as sdfm

sdfm.init(url="http://localhost:8000")

df = sdfm.predict(
    model="tabicl",
    context=context_df,
    predict=predict_df,
    task="classification",
    target="label",
    outputs=["prediction", "probabilities"],
)
```

## Architecture

```text
nvidia_sdfm/
  core/         shared: HTTP transport, response parsing, connectors, dtype mapping
  base.py       ModelAdapter interface + AdapterRegistry
  adapters/
    tabicl.py   single-table adapter
    rfm.py      relational adapter (wraps kumoai)
```

`predict(model=..., **kwargs)` dispatches to the adapter registered under `model`. Each adapter
owns its own accepted keyword arguments.

## Why the RFM adapter isn't symmetric with TabICL

The proposal's example shows both models called the same way:
`sdfm.predict(model=..., context=context_df, predict=predict_df, ...)`. That shape fits TabICL
exactly, but it does not fit how `kumoai.rfm.KumoRFM` actually works: `KumoRFM.predict()` takes a
PQL query and an entity-graph, and builds/samples/sends the request as one fused operation — there
is no standalone "build a payload from two flat DataFrames" step to call into.

Reimplementing that request-building logic outside `kumoai` would duplicate PQL parsing, subgraph
sampling, and point-in-time correctness logic that already exists and is tested there — squarely
in the "model-family internals... stay in their model adapter" territory the proposal itself rules
out of scope for the shared core.

So `adapters/rfm.py` takes the shape RFM actually needs:

```python
df = sdfm.predict(
    model="kumo-rfm",
    graph=graph,               # kumoai.rfm.Graph / LocalGraph
    query="PREDICT ...",       # PQL
    indices=[...],
    outputs=["prediction", "probabilities"],
)
```

It constructs a `kumoai.rfm.KumoRFM` for the given graph, calls `.predict(query, indices=...)`,
and normalizes the result into the same DataFrame shape `core.response` produces for TabICL, so
callers still get one consistent return type regardless of adapter. Requires the `rfm` extra
(`pip install nvidia-sdfm[rfm]`).

## Sessions

Not wired yet. `core/transport.py` implements `predict()` only; `create_session` /
`session_predict` / `delete_session` are deferred until the NIM session path is exercised, per the
proposal's open questions.

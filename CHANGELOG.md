# Changelog

## Unreleased

### Fixed

- **A redirect was an unbounded read.** `requests` releases each redirect hop's
  socket by reading its body in full, with no cap, before following it. This
  package never needs those bytes and they are attacker-controlled on a
  misconfigured endpoint, so a hop's socket is now closed rather than drained.
  The visible difference is that a connection is not reused across a redirect;
  redirects are still followed and the API key is still dropped when one crosses
  origin. One related gap is left open on purpose: urllib3 drains a retryable
  4xx/5xx body before retrying, so a hostile endpoint can still be read up to
  `max_retries` times. Closing that would mean not retrying a status at all,
  which is a change to what `max_retries` promises.
- The architecture guide said an endpoint must advertise `kumo-relational-engine`
  in `/v1/models`. That is the distribution name; the driver checks for the model
  id `kumo-relational`.

### Removed

- **The multi-model dispatch layer.** `ModelAdapter`, `AdapterRegistry` and the
  `registry=` argument to `RelationalClient` are gone. They existed to choose
  between a tabular and a relational model; with one model they were an
  interface with a single implementation. The client now holds one adapter.
  `models()` and `capabilities()` are unchanged from a caller's side and still
  need no live endpoint.
- **`Transport.predict`, `create_session`, `session_predict`, `delete_session`
  and their helpers.** No source path called them: predictions and sessions go
  through the driver's own connection, and the driver owns session lifecycle.
  The client's transport now does what it actually did -- carry configuration,
  report readiness, and close. `ServingTarget.predict` goes for the same reason.
- `format_invalid_params`, which only the removed error path used. The driver
  renders these and the adapter passes them through as error details.

### Added

- The generated TFM bindings now carry a `PROVENANCE.json` recording their own
  SHA-256 alongside the contract revision and spec hash they were built from,
  and a test asserts it. The drift tests need a checkout of the contract, which
  only a maintainer has, so they skip everywhere else; this catches the failure
  that does not need the contract to detect, which is the generated file being
  edited by hand rather than regenerated. The generator writes the record, so
  regenerating moves both together.

### Changed

- **All three distributions now carry the Kumo name.** `nemotron-relational`
  becomes `kumo-relational-engine` and `nemotron-structured-connectors` becomes
  `kumo-connectors`, alongside the client's earlier rename to
  `kumo-relational-client`. Import packages follow: `kumo_relational_engine`
  and `kumo_connectors`. `NemotronRelational` and `NemotronRelationalError`
  become `KumoRelational` and `KumoRelationalError`.
- The vendored contract examples record their source as `structured-data-api`
  rather than an internal URL that would be unreachable for anyone reading
  this repository.
- The engine's internal column sentinels move from `__nemotron_*` to
  `__kumo_*`. These name columns the request declares by name, so the NIM reads
  them as opaque identifiers; the rename was verified against a live NIM.
- The engine package is named for what it is rather than for the model: the
  model id stays `kumo-relational`, so a package called `kumo-relational` would
  have meant two different things.

### Removed

- **The tabular surface is gone.** `client.tabular()`, `TabularModel`, the
  `kumo-tabular` adapter and its request types are removed, and
  `client.models()` now reports `['kumo-relational']` alone. No endpoint serves
  `kumo-tabular`, and the model is not being released, so the client no longer
  offers a method that cannot reach a model. The contract still defines
  `kumo-tabular`, so re-adding an adapter later is additive rather than
  breaking.
- The end-to-end scripts that drove the tabular path are removed, replaced by
  `e2e/run_relational_e2e.py`. The connectors end-to-end script went with them:
  it scored warehouse tables through the tabular path, and the relational path
  needs a time column those tables do not have. The connectors package keeps its
  own unit tests.

### Changed

- The relational engine's HTTP connection to a NIM is now `NimClient`. It and
  the client's own `RelationalClient` were both named `RelationalClient`, so a
  module importing both silently bound whichever came last.

- **The models are now addressed as `kumo-relational` and `kumo-tabular`.** The
  previous ids `nemotron-relational` and `nemotron-tabular` are not accepted. A
  request naming an old id is rejected by the NIM, so this is a breaking change
  for any caller that passes a model id explicitly; `client.relational(...)` and
  `client.tabular(...)` pick the right id on their own and need no edit.
- The vendored contract examples and the generated TFM bindings were re-synced
  from the canonical contract, which carries the same rename.
- The bindings generator now emits output that already satisfies the repo's lint
  and formatting rules, so the generated file no longer has to be hand-formatted
  after generation and the byte-for-byte drift test against the contract is
  enforceable again.

## 1.0.0: all three packages

The first release under the `nemotron-*` names, and the first public one. All
three packages now share a single version: they are released together and only
ever tested against each other, so independent numbering carried information
nobody could act on. Earlier numbering (`kumo-relational-client` 0.x,
`nemotron_relational` 2.x, `nemotron-structured-connectors` 0.x) was published
under the previous distribution names and stops here.

The models are addressed as `nemotron-tabular` and `nemotron-relational`. The
older ids `tabicl`, `kumo-rfm` and `nemotron-relational-v1` are not accepted.

Prepares the client for release as open source. Alongside the licensing and
contribution scaffolding, this closes the defects found by testing the client
end to end against a live NIM, and by a file-by-file review of every
first-party module.

### Fixed: read this before upgrading

- **Python 3.10 was broken.** `nemotron_relational` imported `typing.assert_never`, which
  is 3.11+, so `import nemotron_relational` failed on the floor all three packages declare.
- **A credential in the endpoint URL reached `kumo-relational-client`'s error messages,
  logs and reprs.** `https://user:token@host` is a supported way to address a
  deployment; the userinfo is now stripped everywhere the URL is rendered, as
  `nemotron_relational` already did.
- `sample_rows` accepted any value: its bound returned the `ValueError` instead
  of raising it, so the exception was stored as the field.
- `Graph.from_relbench` could not name 22 of the 33 published RelBench
  datasets. A fixed-width prefix strip mangled every name without a `rel-`
  prefix, so no `dbinfer-`, `tgbl-`, `tgbn-` or `thgl-` dataset could be
  listed or fetched.
- Snowflake `SHOW`, `DESC` and `LIST` failed as `QUERY_FAILED: Unknown error`.
- A 15-table graph was refused by a guard whose message said the limit was
  "more than 15", which made RelBench `rel-trial` unusable.
- A negative `lag_timesteps` was ignored rather than rejected.

### Added

- **A root exception per package.** `nemotron_relational.NemotronRelationalError` and
  `nemotron_structured_connectors.ConnectorError` are now the single base each package raises
  from; `MissingBackendError` joins the latter. Every class keeps the built-in
  it derived from, so existing `except ValueError` / `except RuntimeError`
  keeps working.
- Connection-time failures are typed (`AuthenticationError`,
  `NimUnreachableError`, `NimTimeoutError`) and translated at the
  `kumo-relational-client` boundary, so a wrong API key or an unreachable NIM is caught by
  `except RelationalError` instead of escaping as a bare `ValueError`.
- Reading a warehouse raises `GraphConstructionError` rather than the driver's
  own exception, which shared no base with anything else the client raises.
- **Python 3.13 wheels.** `nemotron_relational` builds and tests cp310 through cp313.
- Composite primary keys, and quoted identifiers in predictive queries.

### Changed

- `import nemotron_relational` no longer reconfigures logging for the whole process. It had
  raised `matplotlib`, `urllib3` and `snowflake` to `ERROR`, and installed a
  handler even where the application had already configured one.
- Errors that reported a caller's mistake through an interpreter's internals
  now name the argument: `indices` says which element and what key type is
  expected, the file-backed connectors name `database`/`uri` rather than
  letting the driver complain about `path`, and an unsupported time unit is
  named instead of being blamed on the `PREDICT` clause.
- Multiclass classification is reachable from `predict()`; the reference said
  otherwise.

## kumo-relational-client 0.2.1 · nemotron_relational 2.24.1 · nemotron-structured-connectors 0.3.0

Supersedes 0.2.0 and 2.24.0, which were tagged before these fixes merged and
contain none of them. `kumo-relational-client 0.2.1` requires `nemotron_relational>=2.24.1` so it
cannot resolve the affected build.

The outcome of a full audit of the client's public surface, every connector, every
graph-construction path, every sampler, all five task types, the HTTP client, and
the error handling around each. 78 findings were reported and fixed.

### Removed: read this before upgrading

The typed-request surface `kumo-relational-client` 0.1.0 exported was replaced by the model
handles. Code written against 0.1.0 that used it will not import.

- `kumo_relational_client.ModelRequest`, `TabICLRequest`, `KumoRelationalRequest`, `ModelAdapter`
  and `AdapterRegistry` are no longer exported; the request types are an
  implementation detail of the handles. Use `client.tabicl(...)` /
  `client.relational(...)`.
- `RelationalClient.predict(request)` and `RelationalClient.register(adapter)` are internal
  (`_predict` / `_register`). Run inference through the handles, and pass a
  custom registry with `RelationalClient(url, registry=...)`.
- `kumo_relational_client.relational` no longer exports `NemotronRelational`, `LocalGraph`,
  `MaterializedPredictionRequest` or `TaskTable`. Direct engine use is refused,
  and nothing on the supported surface returns or accepts the other two, the
  shim now carries what a caller can actually reach. Build graphs with
  `kumo_relational_client.relational.Graph` and predict through `client.relational(graph)`.
  `Dtype`, `Stype` and `ViewConversionWarning` were added in their place.
- `Graph.visualize(backend=...)`: visualization is Mermaid-only; the parameter
  is replaced by `height=`.
- `nemotron_relational.rfm.init()` raises `RuntimeError` unless called by `RelationalClient`.
  Construct an `RelationalClient` instead.

### Compatibility

Everything else on the supported surface is unchanged, verified by an AST diff of
the public API against the 0.1.0 tag (`ed86392`): `RelationalClient`'s constructor and
its `tabicl` / `nemotron_relational` / `models` / `capabilities` / `health_ready` / `close`
methods, `RFMModel.predict` and `predict_task`, `TabICLModel.predict`, every
`Graph.from_*`, `NemotronRelational`'s methods, and `read` / `connect` / `quote_ident` /
`read_table` / `resolve_sql`. Every other signature change is an added parameter
with a default, so an existing call site is unaffected: `driver_options=` on the
Snowflake and Databricks `connect`, `database=` on the SQLite and DuckDB ones,
and `timeout=` / `max_retries=` on `nemotron_relational.init`.

### Behaviour changes to check before upgrading

These are visible to working code. Each is the fix for a case that previously
produced a wrong or silently-degraded result.

- **Multiclass `CLASS` column keeps its target's dtype** instead of always being
  `str`. `result['CLASS'] == '5'` becomes `result['CLASS'] == 5`. Previously
  `CLASS` could not be joined back to the table it names without a manual cast.
- **Client-side validation failures on the NemotronRelational path raise `RelationalError`**
  (code `INVALID_REQUEST`) rather than a bare `ValueError`, matching the contract
  TabICL already followed. Messages are unchanged. The `nemotron_relational` driver surface is
  unaffected: `NimFailureError` subclasses `RuntimeError` and `InvalidResponseError`
  subclasses `ValueError`.
- **Four cases that used to succeed silently now raise**: a `.json` or `.tsv` file
  read as CSV, an unrecognised connection keyword, a missing SQLite database path
  (which used to be created), and an API key sent over plaintext HTTP.
- **An unknown key in `inference_config` is rejected** instead of dropped. A typo
  such as `output_typ='mean'` used to return the *median* with no diagnostic. No
  call can depend on the dropped key having had an effect, by construction it had
  none, since only the declared fields reach the wire.
- **`edges=[]` now means "these edges and no others"**, as its docstring says. On a
  backend with declared foreign keys it used to add them anyway, so a caller who
  pinned the graph's shape got extra edges, and therefore different predictions.
  `edges=None` is unchanged and still applies them.
- **`RelationalClient(max_retries=...)` now governs the NemotronRelational transport too**, which
  previously used a fixed policy of its own. A caller who raised it will see
  transient failures retried where they were not before, and one who set `0` will
  see them surface immediately; failures on the NemotronRelational path therefore take longer
  or shorter to surface than in 0.1.0 according to what was asked for.
- **A caller mistake the engine reports as a `KeyError`**: a typo in
  `exclude_cols_dict`, a feature column present in `context` but not `predict`,
  is `RelationalError(INVALID_REQUEST)` naming the mistake, not `INTERNAL_ERROR` with an
  invitation to file a bug. Code branching on `.code` for those inputs sees the new
  value; `except RelationalError` is unaffected.
- **A malformed create-session response is `INVALID_RESPONSE`**, not
  `INVALID_REQUEST`, matching the prediction path.

### Fixed: wrong results

- TabICL no longer applies the context frame's dtype to the predict frame, which
  truncated fractional values (`[1.9, 2.5, 3.7]` was sent as `[1, 2, 3]`) and could
  change the predicted class.
- A nullable integer column no longer widens to `float64` and corrupts large ids
  (`9007199254740993` became `9007199254740992`), which affected nullable foreign
  keys, the join keys that become graph edges.
- Timezone-aware timestamps are converted to UTC rather than having their offset
  dropped, so tables in different zones no longer land on the same instant.
- Int64 values outside the JS-safe range are encoded as base-10 strings on the
  NemotronRelational path, as the contract requires; they previously failed with HTTP 422.
- `random_seed` now produces identical payloads across processes. Set iteration
  order leaked Python's per-process string hashing into the request.
- `random_seed` is honoured by the SQL samplers, or refused explicitly.

### Fixed: security

- Entity ids are parameter-bound rather than interpolated into SQL, and discovery
  queries quote their identifiers.
- Expressions imported from a metric or semantic view are refused if they carry a
  statement separator, comment, or statement keyword outside a quoted literal. A
  view definition is attacker-controllable input; a user-written `ColumnSpec(expr=)`
  is unchanged.
- An API key is no longer re-sent across a cross-origin redirect.
- Response bodies are read under a 64 MiB cap, so a hostile server cannot force
  unbounded decompression.
- `read('local', path=...)` refuses URI schemes instead of fetching remote URLs.
- `RelationalClient` refuses to send credentials over plaintext HTTP, matching `Transport`,
  and importing the package no longer opens a connection.
- Added `SECURITY.md`.

### Fixed: errors and diagnostics

- Raw `TypeError`, `KeyError` and `AssertionError` no longer escape the public API.
- The NIM's RFC-9457 `invalid_params` detail, which names the exact table, row and
  column rejected, is surfaced instead of discarded.
- `RelationalClient(timeout=..., max_retries=...)` reaches the NemotronRelational path; it was
  silently ignored there.
- `validate()` reports a graph inconsistency as `ValueError` naming the edge, rather
  than `KeyError` from a column lookup.
- Graph discovery constructors reject an empty result instead of returning a
  valid-looking empty graph.
- Primary-key inference reports candidates it declined, deferred until after link
  inference so junction tables stay quiet, and says whether their uniqueness was
  established over the whole table or only over the rows it sampled.
- View conversion diagnostics are readable from `Graph.conversion_messages`.
- FK/PK dtype compatibility is checked on every backend, not only the local one.
- A `context` / `predict` table given to a handle as a `dict`, a list of records or
  a `Series` is refused by name, rather than as `AttributeError: 'dict' object has
  no attribute 'columns'` from inside an adapter.
- `timeout=float('inf')` and `timeout=float('nan')` are refused at construction
  like every other unusable value, rather than at the first request as a raw
  `OverflowError`.
- The "too many entities at once" message suggests a `batch_size` under the cap it
  quotes; on temporal link prediction it suggested 500 against a cap of 200.
- An unsupported column dtype points at `astype(...)`, which works, rather than at
  a per-column override the local table does not offer.
- An invalid PQL query no longer prints ANTLR internals to stdout.

### Added

- NemotronRelational sessions upload a batched job's context once instead of per batch.
- TabICL reuses an uploaded context across predictions on the same handle,
  reducing a repeat call to about 1% of its former size with identical results.
  A handle shared across threads now opens one session rather than one per
  thread, and `POST /v1/sessions` is excluded from the transport retries, so a
  slow create no longer strands a pinned context on the NIM per attempt.
- `RFMModel.predict` / `predict_task` accept `verbose`, and pass unrecognised
  keywords through to the engine instead of raising `TypeError`.
- Unsigned integer columns are accepted at every width. Only `uint8` was, so
  `astype('uint32')`, or reading an unsigned Parquet column, refused the table.
- `kumo_relational_client.relational` exports `ViewConversionWarning`, so the diagnostics that
  view-based graph construction raises can be filtered without importing the
  driver package directly.

### Changed

- `nemotron-structured-connectors` and `nemotron_relational` minimum versions were raised in `kumo-relational-client`'s
  requirements so the client cannot resolve against pre-audit releases.
- Linting covers all three packages; it previously skipped the `nemotron_relational` package
  entirely.
- Removed unreferenced Kumo-Enterprise modules from the `nemotron_relational` wheel.
- Corrected README and `docs/` claims that the code did not support: requests are
  not validated against a NIM's advertised capabilities, there is no public
  `client.predict`, `client.models()` lists the local registry rather than
  discovering what a NIM serves, no macOS wheels are built, and
  `kumo-relational-client[all]` deliberately excludes `[explain]` and `[relbench]`.

# Security

## Reporting a vulnerability

Please do **not** open a public issue or merge request for a security problem.

Report suspected vulnerabilities in this project to the NVIDIA Product Security
Incident Response Team (PSIRT), following the process described at
<https://www.nvidia.com/en-us/security/psirt-policies/>. PSIRT will acknowledge
the report and coordinate the investigation, fix, and disclosure.

A report is most useful when it includes the affected package and version, the
minimal steps or script that reproduce the issue, and what an attacker gains.

## Supported versions

The SDK is distributed as three packages — `nvidia-sdfm`, `kumorfm`, and
`sdfm-connectors` — released together. Only the latest released version of each
receives security fixes; there are no long-term-support branches. Fixes ship in
a new release rather than as patches to older ones.

## Scope notes

The following are intentional design decisions rather than vulnerabilities.
Please read them before filing.

- **Authentication is deployment-owned.** The Universal TFM API contract defines
  no authentication. `api_key` exists only for deployments that front a NIM with
  their own authenticating gateway; it is sent as `X-API-Key` and is refused
  over plaintext `http://` to a non-localhost host.
- **The endpoint you configure is trusted with your data.** Prediction context
  — including the raw cell values of the tables in your graph — is sent to the
  URL you pass to `SDFMClient`. `predict(..., explain=True)` may additionally
  call an OpenAI-compatible endpoint; see
  [Environment Variables](docs/reference/environment-variables.md).
- **Data-source arguments are trusted input.** Connection settings, table names,
  `query=`, `path=` and `storage_options=` are executed or resolved as written,
  with the credentials you supply, as is a `ColumnSpec(expr=...)` you write
  yourself. Do not pass attacker-controlled values into them. Entity
  identifiers are the exception: the SQL samplers bind them as query
  parameters, so they may come from untrusted input.

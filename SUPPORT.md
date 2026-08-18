# Support

## Support level

**Maintained.** The client is developed and released by NVIDIA alongside the
Nemotron structured-data foundation model NIMs. Issues and pull requests are
reviewed, and fixes ship in the next release of all three packages.

It is not a supported NVIDIA product under a commercial agreement. There is no
response-time commitment, and nothing here changes the warranty terms of the
Apache-2.0 licence the code is released under.

## Getting help

| What you have | Where to take it |
| --- | --- |
| A bug, or behaviour that contradicts the documentation | [GitHub issue](https://github.com/NVIDIA/nemotron-structured-client/issues) |
| A question about usage | [GitHub issue](https://github.com/NVIDIA/nemotron-structured-client/issues) with the `question` label |
| A feature request | [GitHub issue](https://github.com/NVIDIA/nemotron-structured-client/issues) describing the problem, not only the proposed fix |
| A suspected security vulnerability | **Not GitHub.** Follow [SECURITY.md](SECURITY.md) |
| A problem with the NIM itself rather than the client | Your NVIDIA support channel for that NIM |

A bug report is most useful with the package versions (`pip show
nemotron-structured-client nemotron-structured-connectors nemotron-relational`), the
Python version and platform, the smallest script that reproduces the problem,
and the full traceback. `StructuredError` and its subclasses carry a stable `code`
field, include it.

## Which package

Three packages are released together and share a version:

- `nemotron-structured-client`: the client and its model handles
- `nemotron-structured-connectors`: reading source tables from warehouses
- `nemotron-relational`: the relational driver, graph building and PQL

If you are unsure which is at fault, file against the repository; triage is our
job, not yours.

## Versions

Only the latest release receives fixes. There are no long-term-support
branches. See [SECURITY.md](SECURITY.md) for the same statement as it applies to
security fixes.

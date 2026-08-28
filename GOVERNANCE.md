# Governance

## Who decides

The client is maintained by NVIDIA. Maintainers are listed in
[MAINTAINERS.md](MAINTAINERS.md) and hold final say on what is merged, on the
public API, and on release timing.

This is a single-vendor project rather than a foundation-governed one. Decisions
are made in the open on issues and pull requests, but there is no voting
process, no technical steering committee, and no path to commit access outside
NVIDIA today. Saying so plainly is better than implying a shared governance
model that does not exist.

## How a change gets in

1. Open an issue describing the problem before writing a large change, so the
   approach can be agreed. Small fixes can go straight to a pull request.
2. Open a pull request against `main`, signed off under the DCO (see
   [CONTRIBUTING.md](CONTRIBUTING.md)).
3. At least one maintainer reviews. CI must pass.
4. A maintainer merges. Contributors do not merge their own changes.

## What gets accepted

Changes are judged on whether they make the client more correct, clearer, or
better tested, and on whether they can be maintained afterwards.

The client deliberately stays thin: it builds a request, sends it to a NIM, and
turns the response into a DataFrame. Model behaviour lives in the NIM, not
here. Changes that move inference logic into the client, or that couple the
client to one deployment's conventions, are unlikely to be accepted regardless
of quality.

The wire contract is shared with the NIM. A change to the request or response
shape cannot be made in this repository alone and needs the server side to move
with it.

## Releases

The three packages are versioned and released together: a given version of one
is only tested against the same version of the others. Releases are cut by
maintainers; there is no fixed cadence.

Breaking changes are avoided within a major version. Where one is unavoidable,
it is called out in [CHANGELOG.md](CHANGELOG.md) under a heading that says so,
with the migration in the same entry.

### Cutting one

Two kinds of tag exist and they do different things. `v1.0.0` marks the release
of all three packages together and carries the GitHub release notes; it uploads
nothing. A tag of the form `<distribution>/v<version>`, such as
`kumo-connectors/v1.0.0`, is what publishes that one distribution to PyPI
through [the publish workflow](.github/workflows/publish.yml).

From a merged, green `main`:

1. Bump `__version__` in all three packages to the same number. If the number
   crosses a major boundary, move the `kumo-*` pins in
   `packages/kumo-relational-client/pyproject.toml` and
   `packages/kumo-relational-engine/setup.py` with it, or the client will
   refuse the engine it was released alongside.
2. Write the entry in [CHANGELOG.md](CHANGELOG.md).
3. Push the three publish tags **one at a time, in this order**:
   `kumo-connectors`, then `kumo-relational-engine`, then
   `kumo-relational-client`. Wait for each tag's publish run to go green before
   pushing the next; pushing them together starts three independent runs, and
   the client would try to install before its siblings are on the index. The
   client pins the other two, so its verify job is what proves the set released
   together installs together.
4. Tag and push `v<version>` and write the GitHub release notes.

A version can never be uploaded to PyPI twice, even after deleting it, so the
tag guard runs before anything is built and refuses a tag that disagrees with
the declared `__version__`. If you get that far and the number was wrong, the
only way out is a new version.

Publishing needs a PyPI trusted publisher for each distribution and a
repository environment named `pypi`; both are configured outside this
repository and are described at the top of the publish workflow.

## Changing this document

Governance changes are made by maintainers, as a pull request like any other, so
the history of how the project was run stays readable.

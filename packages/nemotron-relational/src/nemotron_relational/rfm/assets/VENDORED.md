# Vendored assets

## mermaid.min.js

- Project: https://github.com/mermaid-js/mermaid
- Version: 11.12.1 (browser IIFE bundle, `dist/mermaid.min.js`)
- Source: https://cdn.jsdelivr.net/npm/mermaid@11.12.1/dist/mermaid.min.js
- License: MIT (see `MERMAID_LICENSE`)

Bundled so that `Graph.visualize()` renders entirely offline: no system
executables (graphviz `dot`), no CDN fetch, and no calls to the mermaid.ink
web service are needed to display a graph in a notebook. This matters for
restricted notebook environments (Snowflake, Databricks) whose content
security policies or egress rules block external scripts and services.

To upgrade: replace `mermaid.min.js` and `MERMAID_LICENSE` from the same
jsdelivr path at the new version, update the version here, and re-run the
visualization tests.

# nemotron_relational.pql — provenance

This tree began as a copy of the internal `kumo-pql` parser package,
internalized so the SDK is self-contained (no external `kumo-pql` dependency)
for the open-source release.

**Upstream is gone.** `kumo-pql` has been deleted, so this is now the only copy
and the sole maintained source. There is nothing left to re-sync from; the
script that used to do it has been removed along with it. Edit these files
directly, as first-party code.

This file is kept as the attribution and provenance record for the
internalized code, not as an instruction for re-syncing.

## Baseline

| | |
|---|---|
| Upstream package | `kumo-pql` |
| Commit | `4c8fc57` (tip of `main`; `kumo-pql` was never tagged) |
| Scope taken | `grammar/`, `parser/`, `validator/` — the SDK runtime parse path only |

`rewriting/` was not imported by the parse path and was not taken.

## Changes made against that baseline

Import rewrites:

- `kumopql.*` → `nemotron_relational.pql.*`, and `kumoapi.*` → `nemotron_relational.api.*`.
- The baseline's `sys.modules['kumopql']` alias shim dropped.

Five local patches, all of which are now simply part of this code:

1. `validator/problem_type_validator.py` — reject `TOP K` without `RANK`, and
   reject non-positive `TOP K`.
2. `validator/rfm_validator.py` — also shift the `whatif_ast` location interval
   by the query offset.
3. `validator/type_validator.py` — map the `'Constant'` node kind to
   `ConstantNodeTypeValidator`. Fixes a baseline bug that used
   `ColumnNodeTypeValidator`.
4. `parser/error_translator.py` — a parse error is re-raised by the caller as a
   `ValueError` carrying the same message and the query, so this log record
   drops from WARNING to DEBUG and no longer logs the raw query. **A PQL query
   embeds literal filter values, and WARNING-and-above records are typically
   shipped to a central log store**, so this is a privacy fix, not a cosmetic
   one. Do not restore the query to this record.
5. `parser/visitor.py` — a backtick-quoted name is a spelling device only, so
   it is canonicalized through `nemotron_relational._names.canonical_fqn` as the query is
   parsed. Everything downstream — the sampler, a dataframe lookup, the graph —
   then reads the name the data itself uses.

Internal identifiers removed from shipped comments: three TODOs naming an
individual engineer.

## The grammar is a fork

`grammar/` diverges from the `4c8fc57` baseline: the backtick-quoted identifier
(patch 5's counterpart in the grammar) lets a query name a column a warehouse
spells with a space, such as `Customer ID`. A regenerated ANTLR parser is a
serialized ATN on a single line, so this could never be expressed as a textual
patch — the generated files were carried across a re-sync wholesale instead:

`PQLGrammar.g4`, `PQLGrammar.interp`, `PQLGrammar.tokens`,
`PQLGrammarLexer.interp`, `PQLGrammarLexer.py`, `PQLGrammarLexer.tokens`,
`PQLGrammarParser.py`, `PQLGrammarVisitor.py`.

That fork was never upstreamed, and with `kumo-pql` gone it no longer can be.
This grammar is now the only one. `PQLGrammarParser.py` and
`PQLGrammarLexer.py` are ANTLR output and stay excluded from linting; to change
the language, edit `PQLGrammar.g4` and regenerate.

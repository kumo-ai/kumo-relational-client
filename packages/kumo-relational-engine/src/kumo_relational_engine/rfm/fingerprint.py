# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A name for the shape of a graph, stable enough to key a cache on.

A caller that reuses a built graph needs to know when the one it holds is still
the graph the question would produce. ``hash()`` cannot answer that: Python salts
string hashing per process, so the same graph names itself differently in two
workers, and the built-in covers only table names and edges, so a column that
changed type, a key that moved and a time column that appeared all leave the
name untouched.

This walks the whole shape a prediction depends on, writes it in a fixed order,
and hashes the bytes. Two graphs that would behave identically get one name;
anything a prediction can see changes it.

The engine version is part of it. Rebuilding the same tables under a version
that reads them differently is a different graph, and a cache that could not
tell them apart would serve the older reading indefinitely.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from kumo_relational_engine._version import __version__

if TYPE_CHECKING:
    from kumo_relational_engine.rfm.base.graph import Graph


def _term(value: object) -> str:
    r"""One field, written so no value can be mistaken for a delimiter.

    Each is its length then its text, so a name containing whatever character
    was chosen as a separator cannot split into two fields, and two different
    shapes cannot write the same bytes.
    """
    text = '' if value is None else str(value)
    return f'{len(text)}:{text}'


def _column_terms(column: Any) -> list[str]:
    return [_term(column.name), _term(column.stype), _term(column.dtype)]


def _table_terms(table: Any) -> list[str]:
    r"""Everything about a table that a prediction can see.

    Columns are named in their own order rather than sorted, because that order
    is the one the sampler reads them in.
    """
    terms = [_term(table.name), _term(len(table.columns))]
    for column in table.columns:
        terms.extend(_column_terms(column))
    key = table.primary_key_columns or ()
    terms.append(_term(len(key)))
    terms.extend(_term(c) for c in key)
    for attribute in ('time_column', 'end_time_column'):
        column = getattr(table, attribute, None)
        terms.append(_term(column.name if column is not None else None))
    return terms


def graph_fingerprint(graph: Graph) -> str:
    r"""A hex name for *graph*, equal across processes for equal shapes.

    Tables are visited in sorted order so that two graphs built from the same
    source in a different order share a name. Edges are sorted for the same
    reason: they are a set, and the order they were added in says nothing about
    what the graph does.
    """
    names = sorted(graph.tables)
    terms: list[str] = [
        _term('kumo-relational-engine'),
        _term(__version__),
        _term(len(names)),
    ]

    for name in names:
        terms.extend(_table_terms(graph.tables[name]))

    edges = sorted(
        graph.edges, key=lambda e: (e.src_table, e.fkey, e.dst_table)
    )
    terms.append(_term(len(edges)))
    for edge in edges:
        terms.extend(
            [_term(edge.src_table), _term(edge.fkey), _term(edge.dst_table)]
        )

    inferred = tuple(sorted(graph.inferred_edges))
    terms.append(_term(len(inferred)))
    for src, fkey, dst in inferred:
        terms.extend([_term(src), _term(fkey), _term(dst)])

    return hashlib.sha256(''.join(terms).encode('utf-8')).hexdigest()

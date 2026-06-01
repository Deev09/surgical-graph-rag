"""RelationExtractor Protocol and shared types for per-family extractors.

Each relation family (directional, proximity, support, containment, ...)
lives in its own module. The GraphBuilder (P1.07) orchestrates them but
does not understand any single family's internals.

Phase 1 implements only directional and proximity. Support / containment /
attached / NEAR_SURFACE are Phase 3 work; the EdgeType schema already
reserves slots for them.

Modes (extractor-defined; each family knows its own compat and sparse
code paths):
  - 'compat' — legacy reproduction. Faithful port of relations/compute.py.
    Designed to recreate the existing Replica edge artifact
    (scenes/replica_room_0/computed_relations/scene_graph.json) edge-for-edge.
  - 'sparse' — desired graph going forward. Tighter sparsity, canonical-
    only edge storage. The executor derives inverse types at query time.

The pre-imported Replica scene_graph.json is the Phase 1 replay fixture;
importers/replica.py remains the raw-data ingestion path. See
adapters/oracle_replica.py module docstring for details.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Protocol

from extractors.base import EntityArtifacts
from graph.schema import Edge, EdgeRejection, EdgeType, GraphRef


# Canonical → inverse type mapping. Sparse mode stores only canonical
# types; the executor derives the inverse at query time. Canonical chosen
# alphabetically within each pair so the rule is mechanical and easy to
# audit.
CANONICAL_INVERSE_PAIRS: dict[EdgeType, EdgeType] = {
    "ABOVE": "BELOW",
    "BEHIND": "IN_FRONT_OF",
    "CONTAINS": "INSIDE",
    "LEFT_OF": "RIGHT_OF",
    "ON_TOP_OF": "SUPPORTS",
}

INVERSE_TO_CANONICAL: dict[EdgeType, EdgeType] = {
    v: k for k, v in CANONICAL_INVERSE_PAIRS.items()
}

SYMMETRIC_EDGE_TYPES: frozenset[EdgeType] = frozenset({"NEAR"})


class RelationExtractorConfig(Protocol):
    """Structural protocol for relation-extractor configs. Concrete
    family configs are dataclasses that happen to expose `.mode`."""
    mode: Literal["compat", "sparse"]


@dataclass(frozen=True)
class RelationExtractorDiagnostics:
    """Per-extractor diagnostics. Aggregated into BuildDiagnostics by the
    GraphBuilder.

    physical_edges_total counts every emitted Edge object.
    logical_edges_total normalizes: symmetric pairs counted once, inverse
    pairs counted once. In compat mode physical ≈ 2 × logical because
    compat duplicates NEAR and emits both directions of inverse pairs;
    in sparse mode the two should be equal."""
    extractor: str
    version: str
    mode: str
    physical_edges_per_type: dict[EdgeType, int]
    physical_edges_total: int
    logical_edges_total: int
    rejections_per_type: dict[EdgeType, int]
    rejection_samples: list[EdgeRejection]
    runtime_ms: int


class RelationExtractor(Protocol):
    name: str
    version: str
    edge_types: frozenset[EdgeType]

    def extract(
        self,
        entities: EntityArtifacts,
        config: RelationExtractorConfig,
    ) -> tuple[list[Edge], RelationExtractorDiagnostics]: ...


def count_logical_edges(edges: list[Edge]) -> int:
    """Normalize an edge list to count distinct logical facts.

    - Symmetric edge types (NEAR): an unordered pair {a, b} counts once
      regardless of how many physical edges encode it.
    - Inverse pair types (LEFT_OF/RIGHT_OF, ABOVE/BELOW, ...): a fact
      stored as either direction counts once. The non-canonical
      direction is mapped to its canonical via INVERSE_TO_CANONICAL.
    - All other edge types (ATTACHED_TO, NEAR_SURFACE, ...): each
      ordered (source, type, target) counts once.
    """
    canonical_keys: set[tuple[str, str, str, str, str]] = set()
    symmetric_keys: set[tuple[str, tuple]] = set()

    for e in edges:
        if e.type in SYMMETRIC_EDGE_TYPES:
            a = (e.source.kind, e.source.uid)
            b = (e.target.kind, e.target.uid)
            pair = tuple(sorted([a, b]))
            symmetric_keys.add((e.type, pair))
        elif e.type in CANONICAL_INVERSE_PAIRS:
            canonical_keys.add((
                e.type,
                e.source.kind, e.source.uid,
                e.target.kind, e.target.uid,
            ))
        elif e.type in INVERSE_TO_CANONICAL:
            canonical = INVERSE_TO_CANONICAL[e.type]
            canonical_keys.add((
                canonical,
                e.target.kind, e.target.uid,
                e.source.kind, e.source.uid,
            ))
        else:
            canonical_keys.add((
                e.type,
                e.source.kind, e.source.uid,
                e.target.kind, e.target.uid,
            ))

    return len(canonical_keys) + len(symmetric_keys)


def edge_key(e: Edge) -> tuple[str, str, str, str, str]:
    """Stable identity for set-comparison and gate diffs.
    (source.kind, source.uid, type, target.kind, target.uid)."""
    return (e.source.kind, e.source.uid, e.type, e.target.kind, e.target.uid)


def make_edge_id(
    extractor: str,
    version: str,
    source: GraphRef,
    type_: EdgeType,
    target: GraphRef,
) -> str:
    """Deterministic edge id over extractor + key."""
    payload = f"{extractor}|{version}|{source.kind}:{source.uid}|{type_}|{target.kind}:{target.uid}"
    return f"e_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def make_entity_ref(uid: str) -> GraphRef:
    return GraphRef(kind="entity", uid=uid)


def make_surface_ref(uid: str) -> GraphRef:
    return GraphRef(kind="surface", uid=uid)

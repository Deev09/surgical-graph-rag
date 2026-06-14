"""SUPPORTS derived views (P4.03, P6.03).

`SUPPORTS` is NOT a stored relation in Phase 4 (D3). It is a read-side
projection over `ON_SURFACE` edges: an `ON_SURFACE(entity -> surface)` edge
inverts to `SupportFact(supporter=surface, supported=entity)`. No
`Edge(type="SUPPORTS")` is ever produced, and `SupportFact` is a distinct
type so a caller cannot mistake a derived fact for a stored edge.

Phase 6 adds a separate entity-support projection over
`ON_ENTITY_SURFACE(entity -> entity)`: an object resting on a derived
furniture top is projected as `SupportFact(supporter=owner entity,
supported=resting entity)`. The derived top surface remains evidence on
the stored edge, not a graph ref.

Because `ON_SURFACE` (Design B) is already only support-capable rest-contacts,
the inverse needs NO role filter: exactly one `SupportFact` per `ON_SURFACE`
edge (`SUPPORTS_count == ON_SURFACE_count`).

Evidence is referenced, not copied: each `SupportFact` carries
`derived_from_edge_id` (the originating `ON_SURFACE` edge_id), so the
authoritative evidence stays on the stored edge and is never duplicated
into the view.

Strict: `support_facts` raises if the bundle contains a materialized
`SUPPORTS` edge. In Phase 4 that is a contract violation (`SUPPORTS` must be
derived, never stored); silently ignoring it would mask exactly the bug
this view exists to prevent. `SUPPORTS` remains a reserved legacy EdgeType
(graph/schema.py) — reserved, but not emitted by P4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from graph.schema import GraphRef, SceneGraphBundle


@dataclass(frozen=True)
class SupportFact:
    """A derived support relation. NOT an Edge: this is a read-side fact
    projected from an ON_SURFACE edge, distinct by type so it cannot be
    confused with a stored edge."""
    supporter: GraphRef          # the surface that supports
    supported: GraphRef          # the entity being supported
    derived_from_edge_id: str    # originating ON_SURFACE edge_id (evidence by reference)
    relation: Literal["SUPPORTS"] = "SUPPORTS"


def _raise_if_materialized_supports(bundle: SceneGraphBundle) -> None:
    materialized = [e for e in bundle.edges if e.type == "SUPPORTS"]
    if materialized:
        raise ValueError(
            "SUPPORTS must be a derived view, never stored; found "
            f"{len(materialized)} materialized SUPPORTS edge(s): "
            f"{[e.edge_id for e in materialized][:5]}"
        )


def support_facts(bundle: SceneGraphBundle) -> list[SupportFact]:
    """Project the SUPPORTS view from a bundle's ON_SURFACE edges.

    Returns one SupportFact per ON_SURFACE edge, inverting direction
    (surface supports entity). Raises ValueError if the bundle contains any
    materialized SUPPORTS edge (Phase 4 contract: SUPPORTS is derived-only).
    """
    _raise_if_materialized_supports(bundle)
    facts: list[SupportFact] = []
    for edge in bundle.edges:
        if edge.type != "ON_SURFACE":
            continue
        # Endpoint guard: ON_SURFACE is defined as entity -> surface. A
        # malformed edge would otherwise project a role-swapped SupportFact;
        # fail loud at the read boundary (same strictness as the
        # materialized-SUPPORTS check above).
        if edge.source.kind != "entity" or edge.target.kind != "surface":
            raise ValueError(
                "malformed ON_SURFACE edge "
                f"{edge.edge_id!r}: expected source.kind='entity' and "
                f"target.kind='surface', got source.kind="
                f"{edge.source.kind!r}, target.kind={edge.target.kind!r}"
            )
        facts.append(SupportFact(
            supporter=edge.target,   # surface
            supported=edge.source,   # entity
            derived_from_edge_id=edge.edge_id,
        ))
    return facts


def entity_support_facts(bundle: SceneGraphBundle) -> list[SupportFact]:
    """Project SUPPORTS from ON_ENTITY_SURFACE entity -> entity edges.

    The stored edge source is the supported entity and target is the
    support-capable owner entity. Evidence carries the derived top-surface
    provenance; the view names the supporter directly from edge.target.
    """
    _raise_if_materialized_supports(bundle)
    facts: list[SupportFact] = []
    for edge in bundle.edges:
        if edge.type != "ON_ENTITY_SURFACE":
            continue
        if edge.source.kind != "entity" or edge.target.kind != "entity":
            raise ValueError(
                "malformed ON_ENTITY_SURFACE edge "
                f"{edge.edge_id!r}: expected source.kind='entity' and "
                f"target.kind='entity', got source.kind="
                f"{edge.source.kind!r}, target.kind={edge.target.kind!r}"
            )
        owner_uid = edge.evidence.get("owner_entity_uid")
        if owner_uid is not None and owner_uid != edge.target.uid:
            raise ValueError(
                "malformed ON_ENTITY_SURFACE edge "
                f"{edge.edge_id!r}: target uid {edge.target.uid!r} does not "
                f"match evidence owner_entity_uid {owner_uid!r}"
            )
        facts.append(SupportFact(
            supporter=edge.target,
            supported=edge.source,
            derived_from_edge_id=edge.edge_id,
        ))
    return facts

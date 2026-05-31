"""SceneGraph schema — phase0_design.md §5.4 and §6.

Frozen meanings for EdgeType; changing one requires bumping schema_version
and rebuilding cached bundles. Edges use GraphRef on both sides so a target
can be either an entity or a structural surface (ATTACHED_TO, NEAR_SURFACE).

SceneGraphBundle is pure data (no methods). Indexed-access helpers are
module-level functions below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from common.types import JSON, OrientedBBox, SceneFrame, Vec3


EdgeType = Literal[
    "LEFT_OF", "RIGHT_OF", "IN_FRONT_OF", "BEHIND",
    "ABOVE", "BELOW",
    "ON_TOP_OF", "SUPPORTS", "ATTACHED_TO",
    "INSIDE", "CONTAINS",
    "NEAR_SURFACE",
    "NEAR",
    # FAR is intentionally not in this list — it is a query-time operator
    # over a centroid index, never stored. See reasoner/executor.
]


@dataclass(frozen=True)
class GraphRef:
    """Typed reference into a SceneGraphBundle. kind discriminates entities
    from structural surfaces; uid is stable within the EntityArtifacts
    bundle that produced this graph."""
    kind: Literal["entity", "surface"]
    uid: str


@dataclass(frozen=True)
class Node:
    """Graph-level entity node. Selected attributes copied from
    EntityArtifact at build time. Structural surfaces are NOT nodes; they
    are referenced via SceneGraphBundle.structural_surface_refs."""
    id: str
    label: str
    label_confidence: float
    centroid: Vec3
    bbox_aabb: tuple[Vec3, Vec3]
    bbox_obb: OrientedBBox | None
    embedding_ref: str | None
    attributes: dict[str, JSON] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    edge_id: str
    source: GraphRef
    type: EdgeType
    target: GraphRef
    frame: Literal["world", "viewpoint", "scene_canonical"]
    weight: float
    confidence: float
    extractor: str
    extractor_version: str
    evidence: dict[str, JSON] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeRejection:
    source: GraphRef
    type: EdgeType
    target: GraphRef
    extractor: str
    rejected_reason: str
    evidence: dict[str, JSON] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildDiagnostics:
    """Build-level diagnostics aggregated by the GraphBuilder.

    physical_edges_total counts every stored Edge. logical_edges_total
    normalizes per graph.relations.base.count_logical_edges: symmetric
    pairs once, inverse pairs once. mode is the single mode shared by
    all extractor runs (mixed-mode runs are rejected upstream).

    per_extractor preserves the full per-family RelationExtractorDiagnostics
    so callers can inspect physical and logical totals per family without
    re-running anything.
    """
    extractor_versions: dict[str, str]
    edges_emitted_per_type: dict[EdgeType, int]
    rejections_per_type: dict[EdgeType, int]
    rejection_samples: list[EdgeRejection]
    runtime_ms_per_extractor: dict[str, int]
    # P1.07 additions:
    per_extractor: list["RelationExtractorDiagnostics"] = field(default_factory=list)  # type: ignore[name-defined]  # noqa: F821
    physical_edges_total: int = 0
    logical_edges_total: int = 0
    mode: Literal["compat", "sparse"] = "sparse"


@dataclass(frozen=True)
class SceneGraphBundle:
    schema_version: int            # bump on breaking changes; loader checks
    bundle_hash: str
    scene_id: str
    frame: SceneFrame
    entity_bundle_hash: str
    nodes: list[Node]
    edges: list[Edge]
    structural_surface_refs: list[str]


def find_node(bundle: SceneGraphBundle, uid: str) -> Node | None:
    for n in bundle.nodes:
        if n.id == uid:
            return n
    return None


def edges_from(
    bundle: SceneGraphBundle,
    src: GraphRef,
    *,
    type: EdgeType | None = None,
) -> list[Edge]:
    return [
        e for e in bundle.edges
        if e.source == src and (type is None or e.type == type)
    ]


def edges_to(
    bundle: SceneGraphBundle,
    tgt: GraphRef,
    *,
    type: EdgeType | None = None,
) -> list[Edge]:
    return [
        e for e in bundle.edges
        if e.target == tgt and (type is None or e.type == type)
    ]

"""Thin wrapper that loads the frozen GRAFFITI_BATHROOM scene from
tiny_graph_demo.py into a SceneGraphBundle for regression smoke tests.

Per phase0_design.md §12: tiny_graph_demo.py is frozen and no longer
a CLI entry point. This wrapper exposes its hand-authored graph as a
SceneGraphBundle fixture so the new pipeline has a regression target
independent of the oracle Replica path.

Scope (deliberately narrow):
  - Preserves object IDs (obj_1 through obj_12).
  - Preserves authored edge weights (e.g. obj_1 BELOW obj_3 weight=0.4).
  - Bypasses adapter / extractor / builder: this graph is hand-authored,
    not extracted. The wrapper writes directly to SceneGraphBundle.
  - Does NOT carry over zones into the graph in a way that the Phase 1
    reasoner can query. Zones are recorded on each node's attributes
    for reference but are not used by the rules compiler. This is
    intentional per phase0_design.md §7.3 (zone deferral).

What this is NOT:
  - Not a candidate for the new oracle pipeline.
  - Not expected to answer every original bathroom question. Some of
    the legacy v1 questions depend on zone matching or v1-specific
    ranking semantics that the Phase 1 reasoner does not support.
"""
from __future__ import annotations

from common.types import SceneFrame
from graph.relations.base import make_edge_id, make_entity_ref
from graph.schema import Edge, Node, SceneGraphBundle
from graph.serde import CURRENT_SCHEMA_VERSION as GRAPH_SCHEMA_VERSION
from tiny_graph_demo import GRAFFITI_BATHROOM


_EXTRACTOR_NAME = "authored"
_EXTRACTOR_VERSION = "v1"
_BUNDLE_HASH = "graph_authored_bathroom_v1"
_ENTITY_BUNDLE_HASH = "ent_authored_bathroom_v1"


def _authored_frame() -> SceneFrame:
    return SceneFrame(
        gravity=(0.0, 0.0, -1.0),
        canonical_forward=None,
        canonical_right=None,
        units="meters",
        notes="authored graffiti_bathroom v1 (xyz centroids, no bboxes)",
    )


def _node_from_authored(obj: dict) -> Node:
    centroid = (
        float(obj["xyz"][0]), float(obj["xyz"][1]), float(obj["xyz"][2]),
    )
    # Hand-authored objects have no bbox. Use a tiny synthetic AABB around
    # the centroid so the schema field is well-formed; downstream relation
    # extractors are not run on this bundle, so the exact size does not
    # affect any computed edges.
    h = 0.05
    bbox_aabb = (
        (centroid[0] - h, centroid[1] - h, centroid[2] - h),
        (centroid[0] + h, centroid[1] + h, centroid[2] + h),
    )
    label = str(obj["label"])
    attrs: dict = {
        "display_label": label,
        "aliases": [],
        "zone": obj.get("zone"),
    }
    for k, v in (obj.get("attributes") or {}).items():
        # Don't clobber the new schema fields if a key collides.
        if k not in attrs:
            attrs[k] = v
    return Node(
        id=str(obj["id"]),
        label=label,
        label_confidence=1.0,
        centroid=centroid,
        bbox_aabb=bbox_aabb,
        bbox_obb=None,
        embedding_ref=None,
        attributes=attrs,
        provenance={
            "source": "graffiti_bathroom_authored",
            "frozen_by": "tiny_graph_demo.GRAFFITI_BATHROOM",
        },
    )


def _edge_from_authored(rel: dict) -> Edge:
    src = make_entity_ref(str(rel["source"]))
    tgt = make_entity_ref(str(rel["target"]))
    edge_type = str(rel["type"])
    weight = float(rel.get("weight", 1.0))
    return Edge(
        edge_id=make_edge_id(_EXTRACTOR_NAME, _EXTRACTOR_VERSION, src, edge_type, tgt),
        source=src,
        type=edge_type,            # type: ignore[arg-type]
        target=tgt,
        frame="world",
        weight=weight,
        confidence=1.0,
        extractor=_EXTRACTOR_NAME,
        extractor_version=_EXTRACTOR_VERSION,
        evidence={"authored_weight": weight} if "weight" in rel else {},
    )


def load_bathroom_bundle() -> SceneGraphBundle:
    """Build a SceneGraphBundle from the frozen GRAFFITI_BATHROOM record.

    Object IDs survive verbatim (obj_1 .. obj_12). Authored edge weights
    appear on Edge.weight and are also recorded under
    Edge.evidence['authored_weight'] for debugging.
    """
    nodes = [_node_from_authored(o) for o in GRAFFITI_BATHROOM["objects"]]
    edges = [_edge_from_authored(r) for r in GRAFFITI_BATHROOM["relations"]]
    return SceneGraphBundle(
        schema_version=GRAPH_SCHEMA_VERSION,
        bundle_hash=_BUNDLE_HASH,
        scene_id=str(GRAFFITI_BATHROOM["scene"]),
        frame=_authored_frame(),
        entity_bundle_hash=_ENTITY_BUNDLE_HASH,
        nodes=nodes,
        edges=edges,
        structural_surface_refs=[],
    )

"""Serde for SceneGraphBundle and BuildDiagnostics.

Both are pure JSON (no sidecar blobs). Edge source/target are
GraphRef-shaped objects, dict[EdgeType, int] flattens to JSON objects with
string keys (EdgeType is already a Literal of strings).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, get_args

from common.serde import (
    check_schema_version, obb_from_dict, obb_to_dict, plane_from_dict,
    plane_to_dict, scene_frame_from_dict, scene_frame_to_dict,
    vec3_from_list, vec3_to_list,
)
from graph.schema import (
    BuildDiagnostics, Edge, EdgeRejection, EdgeType, GraphRef, Node,
    SceneGraphBundle, SurfaceRecord,
)

CURRENT_SCHEMA_VERSION = 3  # P4.02: EdgeType gained ON_SURFACE (rest-contact). v2 bundles reject under strict v3 load (no migration path).

_VALID_SURFACE_TYPES = ("floor", "wall", "ceiling")
_VALID_SURFACE_SOURCES = ("habitat_label", "mesh_ransac", "synth_bbox_fallback")

_VALID_EDGE_TYPES = frozenset(get_args(EdgeType))
_VALID_FRAMES = ("world", "viewpoint", "scene_canonical")
_VALID_REF_KINDS = ("entity", "surface")


def _check_edge_type(t: str) -> str:
    if t not in _VALID_EDGE_TYPES:
        raise ValueError(f"unknown EdgeType {t!r}")
    return t


def graphref_to_dict(g: GraphRef) -> dict[str, str]:
    return {"kind": g.kind, "uid": g.uid}


def graphref_from_dict(d: dict[str, Any]) -> GraphRef:
    kind = d["kind"]
    if kind not in _VALID_REF_KINDS:
        raise ValueError(f"unknown GraphRef kind {kind!r}")
    return GraphRef(kind=kind, uid=str(d["uid"]))


def _surface_record_to_dict(s: SurfaceRecord) -> dict[str, Any]:
    return {
        "uid": s.uid,
        "surface_type": s.surface_type,
        "plane": plane_to_dict(s.plane),
        "polygon": (
            [vec3_to_list(v) for v in s.polygon] if s.polygon is not None else None
        ),
        "source": s.source,
        "confidence": s.confidence,
    }


def _surface_record_from_dict(d: dict[str, Any]) -> SurfaceRecord:
    stype = d["surface_type"]
    if stype not in _VALID_SURFACE_TYPES:
        raise ValueError(f"unknown SurfaceRecord surface_type {stype!r}")
    src = d.get("source")
    if src not in _VALID_SURFACE_SOURCES:
        raise ValueError(
            f"unknown SurfaceRecord source {src!r}; allowed: {_VALID_SURFACE_SOURCES}"
        )
    poly = d.get("polygon")
    return SurfaceRecord(
        uid=str(d["uid"]),
        surface_type=stype,
        plane=plane_from_dict(d["plane"]),
        polygon=(
            [vec3_from_list(v) for v in poly] if poly is not None else None
        ),
        source=src,
        confidence=float(d["confidence"]),
    )


def _node_to_dict(n: Node) -> dict[str, Any]:
    return {
        "id": n.id,
        "label": n.label,
        "label_confidence": n.label_confidence,
        "centroid": vec3_to_list(n.centroid),
        "bbox_aabb": [vec3_to_list(n.bbox_aabb[0]), vec3_to_list(n.bbox_aabb[1])],
        "bbox_obb": obb_to_dict(n.bbox_obb) if n.bbox_obb is not None else None,
        "embedding_ref": n.embedding_ref,
        "attributes": dict(n.attributes),
        "provenance": dict(n.provenance),
    }


def _node_from_dict(d: dict[str, Any]) -> Node:
    bb = d["bbox_aabb"]
    return Node(
        id=str(d["id"]),
        label=str(d["label"]),
        label_confidence=float(d["label_confidence"]),
        centroid=vec3_from_list(d["centroid"]),
        bbox_aabb=(vec3_from_list(bb[0]), vec3_from_list(bb[1])),
        bbox_obb=obb_from_dict(d["bbox_obb"]) if d.get("bbox_obb") is not None else None,
        embedding_ref=d.get("embedding_ref"),
        attributes=dict(d.get("attributes", {})),
        provenance={str(k): str(v) for k, v in d.get("provenance", {}).items()},
    )


def _validate_scene_graph_bundle(bundle: SceneGraphBundle) -> None:
    node_ids = [node.id for node in bundle.nodes]
    surface_uids = [surface.uid for surface in bundle.structural_surfaces]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("SceneGraphBundle contains duplicate Node.id values")
    if len(surface_uids) != len(set(surface_uids)):
        raise ValueError("SceneGraphBundle contains duplicate SurfaceRecord.uid values")
    if bundle.structural_surface_refs != surface_uids:
        raise ValueError(
            "SceneGraphBundle.structural_surface_refs must equal the ordered "
            "SurfaceRecord uid list"
        )
    node_id_set = set(node_ids)
    surface_uid_set = set(surface_uids)
    for edge in bundle.edges:
        for ref in (edge.source, edge.target):
            if ref.kind == "entity" and ref.uid not in node_id_set:
                raise ValueError(
                    f"edge {edge.edge_id!r} references unknown entity uid {ref.uid!r}"
                )
            if ref.kind == "surface" and ref.uid not in surface_uid_set:
                raise ValueError(
                    f"edge {edge.edge_id!r} references unknown surface uid {ref.uid!r}"
                )
            if ref.kind not in _VALID_REF_KINDS:
                raise ValueError(f"edge {edge.edge_id!r} has unknown GraphRef kind {ref.kind!r}")


def _edge_to_dict(e: Edge) -> dict[str, Any]:
    return {
        "edge_id": e.edge_id,
        "source": graphref_to_dict(e.source),
        "type": e.type,
        "target": graphref_to_dict(e.target),
        "frame": e.frame,
        "weight": e.weight,
        "confidence": e.confidence,
        "extractor": e.extractor,
        "extractor_version": e.extractor_version,
        "evidence": dict(e.evidence),
    }


def _edge_from_dict(d: dict[str, Any]) -> Edge:
    frame = d["frame"]
    if frame not in _VALID_FRAMES:
        raise ValueError(f"unknown frame {frame!r}")
    return Edge(
        edge_id=str(d["edge_id"]),
        source=graphref_from_dict(d["source"]),
        type=_check_edge_type(d["type"]),
        target=graphref_from_dict(d["target"]),
        frame=frame,
        weight=float(d["weight"]),
        confidence=float(d["confidence"]),
        extractor=str(d["extractor"]),
        extractor_version=str(d["extractor_version"]),
        evidence=dict(d.get("evidence", {})),
    )


def _rejection_to_dict(r: EdgeRejection) -> dict[str, Any]:
    return {
        "source": graphref_to_dict(r.source),
        "type": r.type,
        "target": graphref_to_dict(r.target),
        "extractor": r.extractor,
        "rejected_reason": r.rejected_reason,
        "evidence": dict(r.evidence),
    }


def _rejection_from_dict(d: dict[str, Any]) -> EdgeRejection:
    return EdgeRejection(
        source=graphref_from_dict(d["source"]),
        type=_check_edge_type(d["type"]),
        target=graphref_from_dict(d["target"]),
        extractor=str(d["extractor"]),
        rejected_reason=str(d["rejected_reason"]),
        evidence=dict(d.get("evidence", {})),
    )


def _relation_diagnostics_to_dict(d: Any) -> dict[str, Any]:
    """Serialize a RelationExtractorDiagnostics. Defined here (rather
    than in graph/relations/base.py) so graph-level serde stays
    self-contained; import is local to avoid a graph.schema → graph.relations
    cycle at module load."""
    return {
        "extractor": d.extractor,
        "version": d.version,
        "mode": d.mode,
        "physical_edges_per_type": dict(d.physical_edges_per_type),
        "physical_edges_total": d.physical_edges_total,
        "logical_edges_total": d.logical_edges_total,
        "rejections_per_type": dict(d.rejections_per_type),
        "rejection_samples": [_rejection_to_dict(r) for r in d.rejection_samples],
        "runtime_ms": d.runtime_ms,
    }


def _relation_diagnostics_from_dict(d: dict[str, Any]) -> Any:
    from graph.relations.base import RelationExtractorDiagnostics
    return RelationExtractorDiagnostics(
        extractor=str(d["extractor"]),
        version=str(d["version"]),
        mode=str(d["mode"]),
        physical_edges_per_type={
            _check_edge_type(str(k)): int(v)
            for k, v in d.get("physical_edges_per_type", {}).items()
        },
        physical_edges_total=int(d["physical_edges_total"]),
        logical_edges_total=int(d["logical_edges_total"]),
        rejections_per_type={
            _check_edge_type(str(k)): int(v)
            for k, v in d.get("rejections_per_type", {}).items()
        },
        rejection_samples=[_rejection_from_dict(r) for r in d.get("rejection_samples", [])],
        runtime_ms=int(d["runtime_ms"]),
    )


def _build_diagnostics_to_dict(b: BuildDiagnostics) -> dict[str, Any]:
    return {
        "extractor_versions": dict(b.extractor_versions),
        "edges_emitted_per_type": dict(b.edges_emitted_per_type),
        "rejections_per_type": dict(b.rejections_per_type),
        "rejection_samples": [_rejection_to_dict(r) for r in b.rejection_samples],
        "runtime_ms_per_extractor": dict(b.runtime_ms_per_extractor),
        "per_extractor": [_relation_diagnostics_to_dict(d) for d in b.per_extractor],
        "physical_edges_total": b.physical_edges_total,
        "logical_edges_total": b.logical_edges_total,
        "mode": b.mode,
        "density_policy": b.density_policy,
        "density_ratio": b.density_ratio,
        "sparse_density_limit": b.sparse_density_limit,
    }


def _build_diagnostics_from_dict(d: dict[str, Any]) -> BuildDiagnostics:
    mode = d.get("mode", "sparse")
    if mode not in ("compat", "sparse"):
        raise ValueError(f"unknown BuildDiagnostics mode {mode!r}")
    policy = d.get("density_policy", "phase1_block")
    if policy not in ("phase1_block", "phase2_telemetry_only"):
        raise ValueError(f"unknown density_policy {policy!r}")
    raw_ratio = d.get("density_ratio")
    return BuildDiagnostics(
        extractor_versions={str(k): str(v) for k, v in d.get("extractor_versions", {}).items()},
        edges_emitted_per_type={
            _check_edge_type(str(k)): int(v)
            for k, v in d.get("edges_emitted_per_type", {}).items()
        },
        rejections_per_type={
            _check_edge_type(str(k)): int(v)
            for k, v in d.get("rejections_per_type", {}).items()
        },
        rejection_samples=[_rejection_from_dict(r) for r in d.get("rejection_samples", [])],
        runtime_ms_per_extractor={str(k): int(v) for k, v in d.get("runtime_ms_per_extractor", {}).items()},
        per_extractor=[_relation_diagnostics_from_dict(rd) for rd in d.get("per_extractor", [])],
        physical_edges_total=int(d.get("physical_edges_total", 0)),
        logical_edges_total=int(d.get("logical_edges_total", 0)),
        mode=mode,
        density_policy=policy,
        density_ratio=float(raw_ratio) if raw_ratio is not None else None,
        sparse_density_limit=float(d.get("sparse_density_limit", 14.0)),
    )


def dump_scene_graph_bundle(bundle: SceneGraphBundle, out_dir: Path) -> Path:
    _validate_scene_graph_bundle(bundle)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": bundle.schema_version,
        "bundle_hash": bundle.bundle_hash,
        "scene_id": bundle.scene_id,
        "frame": scene_frame_to_dict(bundle.frame),
        "entity_bundle_hash": bundle.entity_bundle_hash,
        "nodes": [_node_to_dict(n) for n in bundle.nodes],
        "edges": [_edge_to_dict(e) for e in bundle.edges],
        "structural_surface_refs": list(bundle.structural_surface_refs),
        "structural_surfaces": [_surface_record_to_dict(s) for s in bundle.structural_surfaces],
    }
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_scene_graph_bundle(in_dir: Path) -> SceneGraphBundle:
    manifest = in_dir / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    check_schema_version(
        int(payload["schema_version"]), CURRENT_SCHEMA_VERSION, "SceneGraphBundle",
    )
    bundle = SceneGraphBundle(
        schema_version=int(payload["schema_version"]),
        bundle_hash=str(payload["bundle_hash"]),
        scene_id=str(payload["scene_id"]),
        frame=scene_frame_from_dict(payload["frame"]),
        entity_bundle_hash=str(payload["entity_bundle_hash"]),
        nodes=[_node_from_dict(n) for n in payload["nodes"]],
        edges=[_edge_from_dict(e) for e in payload["edges"]],
        structural_surface_refs=[str(x) for x in payload.get("structural_surface_refs", [])],
        structural_surfaces=[
            _surface_record_from_dict(s) for s in payload.get("structural_surfaces", [])
        ],
    )
    _validate_scene_graph_bundle(bundle)
    return bundle


def dump_build_diagnostics(b: BuildDiagnostics, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_build_diagnostics_to_dict(b), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out_path


def load_build_diagnostics(in_path: Path) -> BuildDiagnostics:
    return _build_diagnostics_from_dict(json.loads(in_path.read_text(encoding="utf-8")))

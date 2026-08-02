"""Serde for EntityArtifacts and BundleCorrespondence.

EntityArtifact embedding ndarrays are stored as sidecar .npy files at
<bundle_dir>/embeddings/<object_uid>.npy; the JSON references them via
relative path. dtype and shape round-trip exactly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common.serde import (
    check_schema_version, obb_from_dict, obb_to_dict, plane_from_dict,
    plane_to_dict, read_npy_sidecar, scene_frame_from_dict,
    scene_frame_to_dict, vec3_from_list, vec3_to_list, write_npy_sidecar,
)
from extractors.base import (
    BundleCorrespondence, EntityArtifact, EntityArtifacts, EntityIdentity,
    ExtractionDiagnostics, SemanticHypothesis, StructuralSurface,
)

CURRENT_SCHEMA_VERSION = 2  # P2.06: StructuralSurface gained a `source` field

_VALID_SURFACE_TYPES = ("floor", "wall", "ceiling")
_VALID_SURFACE_SOURCES = (
    "habitat_label", "mesh_ransac", "mesh_region_fit", "synth_bbox_fallback",
)


def _identity_to_dict(i: EntityIdentity) -> dict[str, Any]:
    return {
        "object_uid": i.object_uid,
        "display_label": i.display_label,
        "aliases": list(i.aliases),
        "source_instance_ref": i.source_instance_ref,
    }


def _identity_from_dict(d: dict[str, Any]) -> EntityIdentity:
    return EntityIdentity(
        object_uid=str(d["object_uid"]),
        display_label=str(d["display_label"]),
        aliases=[str(a) for a in d.get("aliases", [])],
        source_instance_ref=str(d.get("source_instance_ref", "")),
    )


def _hyp_to_dict(h: SemanticHypothesis) -> dict[str, Any]:
    return {"label": h.label, "confidence": h.confidence, "source": h.source}


def _hyp_from_dict(d: dict[str, Any]) -> SemanticHypothesis:
    return SemanticHypothesis(
        label=str(d["label"]),
        confidence=float(d["confidence"]),
        source=str(d["source"]),
    )


def _surface_to_dict(s: StructuralSurface) -> dict[str, Any]:
    return {
        "surface_uid": s.surface_uid,
        "surface_type": s.surface_type,
        "plane": plane_to_dict(s.plane),
        "polygon": [vec3_to_list(v) for v in s.polygon] if s.polygon is not None else None,
        "confidence": s.confidence,
        "source": s.source,
    }


def _surface_from_dict(d: dict[str, Any]) -> StructuralSurface:
    st = d["surface_type"]
    if st not in _VALID_SURFACE_TYPES:
        raise ValueError(f"unknown surface_type {st!r}")
    if "source" not in d:
        raise ValueError(
            f"StructuralSurface manifest missing 'source' field "
            f"(present in serde schema v{CURRENT_SCHEMA_VERSION}); "
            f"re-dump or migrate the bundle."
        )
    src = d["source"]
    if src not in _VALID_SURFACE_SOURCES:
        raise ValueError(
            f"unknown StructuralSurface source {src!r}; "
            f"allowed: {_VALID_SURFACE_SOURCES}"
        )
    poly = d.get("polygon")
    return StructuralSurface(
        surface_uid=str(d["surface_uid"]),
        surface_type=st,
        plane=plane_from_dict(d["plane"]),
        polygon=[vec3_from_list(v) for v in poly] if poly is not None else None,
        confidence=float(d["confidence"]),
        source=src,
    )


def _entity_to_dict(e: EntityArtifact, bundle_dir: Path) -> dict[str, Any]:
    embedding_ref: str | None = None
    if e.embedding is not None:
        rel = f"embeddings/{e.identity.object_uid}.npy"
        write_npy_sidecar(e.embedding, bundle_dir / rel)
        embedding_ref = rel
    return {
        "identity": _identity_to_dict(e.identity),
        "bbox_aabb": [vec3_to_list(e.bbox_aabb[0]), vec3_to_list(e.bbox_aabb[1])],
        "bbox_obb": obb_to_dict(e.bbox_obb) if e.bbox_obb is not None else None,
        "centroid": vec3_to_list(e.centroid),
        "geometry_handle": e.geometry_handle,
        "semantic_hypotheses": [_hyp_to_dict(h) for h in e.semantic_hypotheses],
        "embedding_ref": embedding_ref,
        "extraction_diagnostics": dict(e.extraction_diagnostics),
    }


def _entity_from_dict(d: dict[str, Any], bundle_dir: Path) -> EntityArtifact:
    bb = d["bbox_aabb"]
    embedding = None
    if d.get("embedding_ref"):
        embedding = read_npy_sidecar(bundle_dir / d["embedding_ref"])
    return EntityArtifact(
        identity=_identity_from_dict(d["identity"]),
        bbox_aabb=(vec3_from_list(bb[0]), vec3_from_list(bb[1])),
        bbox_obb=obb_from_dict(d["bbox_obb"]) if d.get("bbox_obb") is not None else None,
        centroid=vec3_from_list(d["centroid"]),
        geometry_handle=d.get("geometry_handle"),
        semantic_hypotheses=[_hyp_from_dict(h) for h in d.get("semantic_hypotheses", [])],
        embedding=embedding,
        extraction_diagnostics=dict(d.get("extraction_diagnostics", {})),
    )


def _diagnostics_to_dict(d: ExtractionDiagnostics) -> dict[str, Any]:
    return {
        "n_entities": d.n_entities,
        "n_structural_surfaces": d.n_structural_surfaces,
        "runtime_seconds": d.runtime_seconds,
        "coverage_score": d.coverage_score,
        "notes": d.notes,
    }


def _diagnostics_from_dict(d: dict[str, Any]) -> ExtractionDiagnostics:
    return ExtractionDiagnostics(
        n_entities=int(d["n_entities"]),
        n_structural_surfaces=int(d["n_structural_surfaces"]),
        runtime_seconds=float(d["runtime_seconds"]),
        coverage_score=d.get("coverage_score"),
        notes=str(d.get("notes", "")),
    )


def dump_entity_artifacts(bundle: EntityArtifacts, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": bundle.schema_version,
        "bundle_hash": bundle.bundle_hash,
        "scene_id": bundle.scene_id,
        "frame": scene_frame_to_dict(bundle.frame),
        "representation_hash": bundle.representation_hash,
        "extractor_name": bundle.extractor_name,
        "extractor_version": bundle.extractor_version,
        "entities": [_entity_to_dict(e, out_dir) for e in bundle.entities],
        "structural_surfaces": [_surface_to_dict(s) for s in bundle.structural_surfaces],
        "geometry_store_path": str(bundle.geometry_store_path) if bundle.geometry_store_path is not None else None,
        "diagnostics": _diagnostics_to_dict(bundle.diagnostics),
        "notes": dict(bundle.notes),
    }
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_entity_artifacts(in_dir: Path) -> EntityArtifacts:
    manifest = in_dir / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    check_schema_version(
        int(payload["schema_version"]), CURRENT_SCHEMA_VERSION, "EntityArtifacts",
    )
    gsp = payload.get("geometry_store_path")
    return EntityArtifacts(
        schema_version=int(payload["schema_version"]),
        bundle_hash=str(payload["bundle_hash"]),
        scene_id=str(payload["scene_id"]),
        frame=scene_frame_from_dict(payload["frame"]),
        representation_hash=str(payload["representation_hash"]),
        extractor_name=str(payload["extractor_name"]),
        extractor_version=str(payload["extractor_version"]),
        entities=[_entity_from_dict(e, in_dir) for e in payload["entities"]],
        structural_surfaces=[_surface_from_dict(s) for s in payload.get("structural_surfaces", [])],
        geometry_store_path=Path(gsp) if gsp is not None else None,
        diagnostics=_diagnostics_from_dict(payload["diagnostics"]),
        notes=dict(payload.get("notes", {})),
    )


def dump_bundle_correspondence(c: BundleCorrespondence, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_bundle_hash": c.source_bundle_hash,
        "target_bundle_hash": c.target_bundle_hash,
        "entity_pairs": [list(p) for p in c.entity_pairs],
        "surface_pairs": [list(p) for p in c.surface_pairs],
        "method": c.method,
        "score": dict(c.score),
        "unmatched_source_entities": list(c.unmatched_source_entities),
        "unmatched_target_entities": list(c.unmatched_target_entities),
        "unmatched_source_surfaces": list(c.unmatched_source_surfaces),
        "unmatched_target_surfaces": list(c.unmatched_target_surfaces),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def load_bundle_correspondence(in_path: Path) -> BundleCorrespondence:
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    return BundleCorrespondence(
        source_bundle_hash=str(payload["source_bundle_hash"]),
        target_bundle_hash=str(payload["target_bundle_hash"]),
        entity_pairs=[(str(p[0]), str(p[1])) for p in payload["entity_pairs"]],
        surface_pairs=[(str(p[0]), str(p[1])) for p in payload["surface_pairs"]],
        method=str(payload["method"]),
        score={str(k): float(v) for k, v in payload["score"].items()},
        unmatched_source_entities=[str(x) for x in payload["unmatched_source_entities"]],
        unmatched_target_entities=[str(x) for x in payload["unmatched_target_entities"]],
        unmatched_source_surfaces=[str(x) for x in payload["unmatched_source_surfaces"]],
        unmatched_target_surfaces=[str(x) for x in payload["unmatched_target_surfaces"]],
    )

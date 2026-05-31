"""Serde for SceneRepresentationBundle.

Layout on disk:
  <bundle_dir>/
    manifest.json   # the bundle data, no large blobs

Geometry blobs are referenced by GeometryHandle.uri (relative or absolute
path); the loader's job is to deref via a representation-specific loader,
not to embed geometry in the manifest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common.serde import (
    check_schema_version, camera_pose_from_dict, camera_pose_to_dict,
    scene_frame_from_dict, scene_frame_to_dict,
)
from representations.base import (
    GeometryHandle, ReconstructionDiagnostics, RepresentationCapabilities,
    SceneRepresentationBundle,
)

CURRENT_SCHEMA_VERSION = 1

_VALID_CHANNELS = ("rgb", "depth", "normals", "semantic_features", "instance_features")
_VALID_GEOMETRY_KINDS = (
    "mesh_file", "pointcloud_file", "splat_file", "nerf_checkpoint",
    "in_memory", "oracle_passthrough",
)


def _geometry_handle_to_dict(h: GeometryHandle) -> dict[str, Any]:
    return {"kind": h.kind, "uri": h.uri, "notes": dict(h.notes)}


def _geometry_handle_from_dict(d: dict[str, Any]) -> GeometryHandle:
    kind = d["kind"]
    if kind not in _VALID_GEOMETRY_KINDS:
        raise ValueError(f"unknown geometry kind {kind!r}")
    return GeometryHandle(kind=kind, uri=str(d["uri"]), notes=dict(d.get("notes", {})))


def _capabilities_to_dict(c: RepresentationCapabilities) -> dict[str, Any]:
    return {
        "renderable_channels": sorted(c.renderable_channels),
        "supports_arbitrary_pose": c.supports_arbitrary_pose,
        "deterministic": c.deterministic,
        "typical_render_ms": c.typical_render_ms,
    }


def _capabilities_from_dict(d: dict[str, Any]) -> RepresentationCapabilities:
    chans = list(d.get("renderable_channels", []))
    for c in chans:
        if c not in _VALID_CHANNELS:
            raise ValueError(f"unknown channel {c!r}")
    return RepresentationCapabilities(
        renderable_channels=frozenset(chans),
        supports_arbitrary_pose=bool(d["supports_arbitrary_pose"]),
        deterministic=bool(d["deterministic"]),
        typical_render_ms=int(d["typical_render_ms"]),
    )


def _diagnostics_to_dict(d: ReconstructionDiagnostics) -> dict[str, Any]:
    return {
        "loss": d.loss,
        "coverage": d.coverage,
        "pose_rmse": d.pose_rmse,
        "runtime_seconds": d.runtime_seconds,
        "notes": d.notes,
    }


def _diagnostics_from_dict(d: dict[str, Any]) -> ReconstructionDiagnostics:
    return ReconstructionDiagnostics(
        loss=d.get("loss"),
        coverage=d.get("coverage"),
        pose_rmse=d.get("pose_rmse"),
        runtime_seconds=float(d["runtime_seconds"]),
        notes=str(d.get("notes", "")),
    )


def dump_scene_repr_bundle(bundle: SceneRepresentationBundle, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": bundle.schema_version,
        "representation_hash": bundle.representation_hash,
        "scene_id": bundle.scene_id,
        "frame": scene_frame_to_dict(bundle.frame),
        "capabilities": _capabilities_to_dict(bundle.capabilities),
        "geometry_handle": _geometry_handle_to_dict(bundle.geometry_handle),
        "poses": [camera_pose_to_dict(p) for p in bundle.poses],
        "diagnostics": _diagnostics_to_dict(bundle.diagnostics),
        "notes": dict(bundle.notes),
    }
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_scene_repr_bundle(in_dir: Path) -> SceneRepresentationBundle:
    manifest = in_dir / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    check_schema_version(
        int(payload["schema_version"]), CURRENT_SCHEMA_VERSION,
        "SceneRepresentationBundle",
    )
    return SceneRepresentationBundle(
        schema_version=int(payload["schema_version"]),
        representation_hash=str(payload["representation_hash"]),
        scene_id=str(payload["scene_id"]),
        frame=scene_frame_from_dict(payload["frame"]),
        capabilities=_capabilities_from_dict(payload["capabilities"]),
        geometry_handle=_geometry_handle_from_dict(payload["geometry_handle"]),
        poses=[camera_pose_from_dict(p) for p in payload.get("poses", [])],
        diagnostics=_diagnostics_from_dict(payload["diagnostics"]),
        notes=dict(payload.get("notes", {})),
    )

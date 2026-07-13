"""C1.02 — anonymous candidate EntityArtifacts from a SegmentationOutput.

The deployable half of C1: raw mesh.ply + a segmenter's dense vertex
assignment -> EntityArtifacts with NO oracle content. Labels are anonymous
("segment_<id>"), semantic_hypotheses empty, structural_surfaces empty, and
notes prove semantic_source="none" / surface_source="none" (contract G5).

The canonical frame (rotation + z_translation) is CALLER-SUPPLIED — this
module never opens info_semantic.json. For A/B/C comparability the caller
passes the frame from variant A (see tools/c1_exact_eval.py); a fully raw C3
pipeline would pass a mesh-derived frame instead.

Per retained segment:
  - object_uid  "obj_<instance_id>", source_instance_ref "segmenter:<id>"
  - bbox_aabb + centroid from its assigned vertices (canonical frame)
  - bbox_obb: gravity-aligned OBB — 2D PCA yaw in XY + vertical extent —
    fitted from the same vertices
  - geometry_handle "<bundle_dir>/vertex_instance_ids.npy#<id>" referencing
    the dense assignment sidecar (exact per-vertex membership, which boxes
    alone cannot carry)

Fails loudly on: input mesh hash mismatch, assignment length mismatch,
out-of-range instance ids, non-finite vertices, and segments left empty
after min_vertices filtering (contract C1.02 fail conditions).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from common.types import OrientedBBox
from extractors.base import (
    EntityArtifact,
    EntityArtifacts,
    EntityIdentity,
    ExtractionDiagnostics,
    SceneFrame,
)
from segmenter.base import SegmentationOutput, sha256_file
from segmenter.ply import parse_vertices

CANDIDATE_SCHEMA_VERSION = 2
CANDIDATE_EXTRACTOR_VERSION = "0.1"


def _yaw_obb(pts: np.ndarray) -> OrientedBBox:
    """Gravity-aligned OBB: principal yaw from 2D PCA on XY, extents from the
    yaw-rotated bounds, z extent direct. Degenerate XY spreads fall back to
    yaw=0 (axis-aligned)."""
    xy = pts[:, :2]
    centered = xy - xy.mean(axis=0)
    cov = centered.T @ centered / max(1, len(xy) - 1)
    if not np.isfinite(cov).all() or np.allclose(cov, 0.0):
        theta = 0.0
    else:
        evals, evecs = np.linalg.eigh(cov)
        major = evecs[:, int(np.argmax(evals))]
        theta = math.atan2(float(major[1]), float(major[0]))
    c, s = math.cos(-theta), math.sin(-theta)
    rx = xy[:, 0] * c - xy[:, 1] * s
    ry = xy[:, 0] * s + xy[:, 1] * c
    z = pts[:, 2]
    lo = (rx.min(), ry.min(), z.min())
    hi = (rx.max(), ry.max(), z.max())
    cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
    # rotate the rotated-frame center back to world
    cw, sw = math.cos(theta), math.sin(theta)
    center = (cx * cw - cy * sw, cx * sw + cy * cw, (lo[2] + hi[2]) / 2.0)
    extents = ((hi[0] - lo[0]) / 2.0, (hi[1] - lo[1]) / 2.0, (hi[2] - lo[2]) / 2.0)
    half = theta / 2.0
    quat = (0.0, 0.0, math.sin(half), math.cos(half))  # x, y, z, w about +z
    return OrientedBBox(center=center, extents=extents, rotation_quat=quat)


def build_candidate_artifacts(
    mesh_path: Path,
    seg: SegmentationOutput,
    scene_id: str,
    *,
    rotation: tuple[tuple[float, float, float], ...],
    z_translation: float,
    bundle_dir: Path | None = None,
    min_vertices: int = 20,
) -> EntityArtifacts:
    """Anonymous candidate bundle (no oracle labels, no oracle surfaces)."""
    mesh_sha = sha256_file(mesh_path)
    if mesh_sha != seg.input_mesh_sha256:
        raise ValueError(
            f"input mesh hash mismatch: bundle was produced from "
            f"{seg.input_mesh_sha256[:16]}..., got {mesh_sha[:16]}... ({mesh_path})")

    xyz = parse_vertices(mesh_path)
    if len(xyz) != seg.n_vertices or len(xyz) != len(seg.vertex_instance_ids):
        raise ValueError(
            f"assignment length mismatch: mesh has {len(xyz)} vertices, "
            f"assignment has {len(seg.vertex_instance_ids)} (meta {seg.n_vertices})")
    if not np.isfinite(xyz).all():
        raise ValueError("non-finite vertex coordinates in raw mesh")

    R = np.array(rotation, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {R.shape}")
    xyz = np.einsum("ij,nj->ni", R, xyz)
    if not np.isfinite(xyz).all():
        raise ValueError("non-finite vertex coordinates after frame transform")
    xyz[:, 2] += z_translation

    ids = seg.vertex_instance_ids
    known = set(seg.instance_ids())
    present = {int(i) for i in np.unique(ids) if i >= 0}
    if present - known:
        raise ValueError(f"assignment references unknown instance ids: "
                         f"{sorted(present - known)[:5]}")

    entities: list[EntityArtifact] = []
    n_dropped_small = 0
    for inst in sorted(present):
        pts = xyz[ids == inst]
        if len(pts) < min_vertices:
            n_dropped_small += 1
            continue
        lo = tuple(float(v) for v in pts.min(axis=0))
        hi = tuple(float(v) for v in pts.max(axis=0))
        handle = (f"{bundle_dir}/vertex_instance_ids.npy#{inst}"
                  if bundle_dir is not None else f"vertex_instance_ids#{inst}")
        entities.append(EntityArtifact(
            identity=EntityIdentity(
                object_uid=f"obj_{inst}",
                display_label=f"segment_{inst}",
                aliases=[],
                source_instance_ref=f"segmenter:{inst}",
            ),
            bbox_aabb=(lo, hi),
            bbox_obb=_yaw_obb(pts),
            centroid=tuple((lo[i] + hi[i]) / 2.0 for i in range(3)),
            geometry_handle=handle,
            semantic_hypotheses=[],
            extraction_diagnostics={"n_vertices": int(len(pts))},
        ))
    if not entities:
        raise ValueError(
            f"no retained segments (all {len(present)} under min_vertices="
            f"{min_vertices}) — refusing to emit an empty candidate")

    return EntityArtifacts(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        bundle_hash=f"c1cand_{seg.output_sha256[:16]}",
        scene_id=scene_id,
        frame=SceneFrame(gravity=(0.0, 0.0, -1.0), canonical_forward=None,
                         canonical_right=None, units="meters",
                         notes="anonymous C1 candidate; caller-supplied frame"),
        representation_hash=f"mesh_{mesh_sha[:16]}",
        extractor_name="c1_candidate",
        extractor_version=CANDIDATE_EXTRACTOR_VERSION,
        entities=entities,
        structural_surfaces=[],
        geometry_store_path=str(bundle_dir) if bundle_dir is not None else None,
        diagnostics=ExtractionDiagnostics(
            n_entities=len(entities),
            n_structural_surfaces=0,
            runtime_seconds=0.0,
            coverage_score=None,
            notes=(f"anonymous segments from {seg.segmenter_name} "
                   f"v{seg.segmenter_version}; retained={len(entities)} "
                   f"dropped_small={n_dropped_small}"),
        ),
        notes={
            "source": "mesh.ply (geometry) + segmenter assignment",
            "semantic_source": "none",
            "surface_source": "none",
            "frame_source": "caller_supplied",
            "z_translation": z_translation,
            "segmenter": {
                "name": seg.segmenter_name,
                "version": seg.segmenter_version,
                "config_params_json": seg.config_params_json,
                "output_sha256": seg.output_sha256,
            },
            "min_vertices": min_vertices,
            "n_dropped_small_segments": n_dropped_small,
        },
    )

"""Phase 3 P3.05 — polygon-clip telemetry + monotonicity report.

Answers the six questions the Phase 3 closeout needs:

  1. How many NEAR_SURFACE edges does plane mode (Phase 2 default) emit?
  2. How many does polygon mode (use_polygon_clip=True) emit?
  3. Which edges flip from near to not_near when polygon clipping is on?
  4. Are polygon-mode edges a subset of plane-mode edges for surfaces
     whose `polygon` is not None?  (A6 subset claim.)
  5. Does default plane mode remain Phase 2 byte-equivalent at the
     GraphBuilder bundle_hash level when use_polygon_clip is added to
     the config dataclass?  (A4.)
  6. Is the artifact deterministic and timestamp-free?

Output:
  scenes/replica_room_0/eval/phase3_polygon_clip_telemetry.json

Determinism: stable. No timestamp. Edge UIDs are sorted before emission;
JSON is dumped with `sort_keys=True` and a trailing newline so byte-level
diffs against the committed copy are signal, not noise. Re-running the
tool must produce an identical file (tested in
tests/tools/test_phase3_polygon_clip_telemetry.py).

Skip-on-missing: enriched-v2 importer output must exist.

Run: python tools/phase3_polygon_clip_telemetry.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.base import ReconstructionConfig
from adapters.oracle_replica import OracleReplicaAdapter, build_replica_capture_bundle
from extractors.base import InstanceExtractorConfig
from extractors.oracle_replica import OracleReplicaExtractor
from geometry.surface_distance import POLYGON_ON_PLANE_TOL_M
from graph.builder import ExtractorRun, build_graph
from graph.relations.surface import (
    PLANE_MODE_VERSION,
    POLYGON_CLIPPED_VERSION,
    SurfaceProximityConfig,
    SurfaceProximityExtractor,
)
from representations.mesh import MeshRepresentation


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
REPLICA_V2_DIR = REPLICA_SCENE_DIR / "enriched" / "v2"
ARTIFACT_PATH = (
    REPLICA_SCENE_DIR / "eval" / "phase3_polygon_clip_telemetry.json"
)


def _build_replica_artifacts():
    capture = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    representation = MeshRepresentation(
        bundle=OracleReplicaAdapter().reconstruct(
            capture,
            ReconstructionConfig(name="oracle_replica", version="0.1", params={}),
        ),
    )
    return OracleReplicaExtractor(enriched_v2_path=REPLICA_V2_DIR).extract(
        representation,
        InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )


def _edges_by_pair(edges) -> dict[tuple[str, str], dict]:
    """Index edges by (entity_uid, surface_uid) for cross-mode comparison.
    Both modes evaluate the same (entity, surface) pairs, so the natural
    join key is the pair, not the edge_id (which differs across modes
    because version is embedded in edge_id)."""
    return {
        (e.source.uid, e.target.uid): {
            "edge_id": e.edge_id,
            "extractor_version": e.extractor_version,
            "evidence": dict(e.evidence),
        }
        for e in edges
    }


def _count_by_surface_type(edges, surface_type_by_uid: dict[str, str]) -> dict:
    counts: Counter = Counter()
    for e in edges:
        counts[surface_type_by_uid.get(e.target.uid, "?")] += 1
    return dict(sorted(counts.items()))


def _surface_lookup(artifacts) -> tuple[dict[str, str], set[str], set[str]]:
    surface_type_by_uid: dict[str, str] = {}
    polygon_present_uids: set[str] = set()
    polygon_none_uids: set[str] = set()
    for surface in artifacts.structural_surfaces:
        surface_type_by_uid[surface.surface_uid] = surface.surface_type
        if surface.polygon is None:
            polygon_none_uids.add(surface.surface_uid)
        else:
            polygon_present_uids.add(surface.surface_uid)
    return surface_type_by_uid, polygon_present_uids, polygon_none_uids


def _max_polygon_off_plane_drift_m(artifacts) -> float:
    """Worst-case |signed plane distance| across every polygon vertex of
    every polygoned surface in the bundle. Records the actual round-off
    pressure the importer is putting on POLYGON_ON_PLANE_TOL_M, so
    creep over time is visible per-scene in the artifact."""
    worst = 0.0
    for surface in artifacts.structural_surfaces:
        if surface.polygon is None:
            continue
        p = surface.plane
        for vertex in surface.polygon:
            sd = abs(p.a * vertex[0] + p.b * vertex[1] + p.c * vertex[2] + p.d)
            if sd > worst:
                worst = sd
    return worst


def _build_minimal_bundle_hash(artifacts, config: SurfaceProximityConfig) -> str:
    """GraphBuilder bundle_hash for a NEAR_SURFACE-only run on the real
    Replica room_0 entity bundle. Validates A4 at full scene scale, not
    just the synthetic builder check from P3.03."""
    runs = [ExtractorRun(SurfaceProximityExtractor(), config)]
    bundle, _diag = build_graph(
        artifacts, runs, density_policy="phase2_telemetry_only",
    )
    return bundle.bundle_hash


def main() -> int:
    if not (REPLICA_V2_DIR / "scene_graph.json").exists():
        print("Refusing: enriched-v2 importer output is missing.")
        return 1

    artifacts = _build_replica_artifacts()
    entity_count = len(artifacts.entities)
    surface_type_by_uid, polygon_present_uids, polygon_none_uids = (
        _surface_lookup(artifacts)
    )
    max_off_plane_drift = _max_polygon_off_plane_drift_m(artifacts)

    plane_cfg = SurfaceProximityConfig()
    plane_cfg_explicit = SurfaceProximityConfig(use_polygon_clip=False)
    polygon_cfg = SurfaceProximityConfig(use_polygon_clip=True)

    # Q1, Q2: edge counts in each mode.
    extractor = SurfaceProximityExtractor()
    plane_edges, plane_diag = extractor.extract(artifacts, plane_cfg)
    polygon_edges, polygon_diag = extractor.extract(artifacts, polygon_cfg)

    plane_by_pair = _edges_by_pair(plane_edges)
    polygon_by_pair = _edges_by_pair(polygon_edges)

    plane_count_by_type = _count_by_surface_type(plane_edges, surface_type_by_uid)
    polygon_count_by_type = _count_by_surface_type(
        polygon_edges, surface_type_by_uid,
    )

    all_surface_types = sorted(
        set(plane_count_by_type) | set(polygon_count_by_type)
    )
    by_surface_type_combined = {
        stype: {
            "plane": plane_count_by_type.get(stype, 0),
            "polygon": polygon_count_by_type.get(stype, 0),
            "delta": (
                polygon_count_by_type.get(stype, 0)
                - plane_count_by_type.get(stype, 0)
            ),
        }
        for stype in all_surface_types
    }

    # Q3: which (entity, surface) pairs flipped between modes.
    flipped_near_to_not_near = []
    for pair in sorted(plane_by_pair.keys() - polygon_by_pair.keys()):
        entity_uid, surface_uid = pair
        plane_evidence = plane_by_pair[pair]["evidence"]
        # Recompute polygon-mode distance for the rejected pair by reading
        # the extractor's rejection sample is brittle (capped at 64); cheaper
        # is to ask the dispatcher directly, but the diagnostics aren't
        # carrying it for rejected edges. We record what we KNOW from the
        # plane-mode edge plus the surface type / polygon presence.
        flipped_near_to_not_near.append({
            "entity_uid": entity_uid,
            "surface_uid": surface_uid,
            "surface_type": surface_type_by_uid.get(surface_uid, "?"),
            "polygon_present": surface_uid in polygon_present_uids,
            "plane_mode_distance_m": plane_evidence.get("distance_m"),
            "plane_mode_threshold_m": plane_evidence.get("threshold_m"),
        })

    flipped_not_near_to_near = []
    for pair in sorted(polygon_by_pair.keys() - plane_by_pair.keys()):
        entity_uid, surface_uid = pair
        polygon_evidence = polygon_by_pair[pair]["evidence"]
        flipped_not_near_to_near.append({
            "entity_uid": entity_uid,
            "surface_uid": surface_uid,
            "surface_type": surface_type_by_uid.get(surface_uid, "?"),
            "polygon_present": surface_uid in polygon_present_uids,
            "polygon_mode_distance_m": polygon_evidence.get("distance_m"),
            "polygon_mode_normal_gap_m": polygon_evidence.get("normal_gap_m"),
            "polygon_mode_in_plane_gap_m": polygon_evidence.get("in_plane_gap_m"),
            "polygon_mode_threshold_m": polygon_evidence.get("threshold_m"),
        })

    # Q4: subset claim, restricted to surfaces with polygons present.
    plane_pairs_polygoned = {
        pair for pair in plane_by_pair.keys() if pair[1] in polygon_present_uids
    }
    polygon_pairs_polygoned = {
        pair for pair in polygon_by_pair.keys() if pair[1] in polygon_present_uids
    }
    subset_violations = sorted(polygon_pairs_polygoned - plane_pairs_polygoned)

    # Q5: Phase 2 byte-equivalence at GraphBuilder bundle_hash level.
    bundle_hash_default = _build_minimal_bundle_hash(artifacts, plane_cfg)
    bundle_hash_explicit_false = _build_minimal_bundle_hash(
        artifacts, plane_cfg_explicit,
    )
    bundle_hash_polygon = _build_minimal_bundle_hash(artifacts, polygon_cfg)

    # Q6: plane-mode determinism — run again, compare edge_ids.
    plane_edges_run2, _ = extractor.extract(artifacts, plane_cfg)
    plane_edge_ids_run1 = sorted(e.edge_id for e in plane_edges)
    plane_edge_ids_run2 = sorted(e.edge_id for e in plane_edges_run2)
    two_runs_byte_equal = plane_edge_ids_run1 == plane_edge_ids_run2

    polygon_edges_run2, _ = extractor.extract(artifacts, polygon_cfg)
    polygon_edge_ids_run1 = sorted(e.edge_id for e in polygon_edges)
    polygon_edge_ids_run2 = sorted(e.edge_id for e in polygon_edges_run2)
    polygon_two_runs_byte_equal = polygon_edge_ids_run1 == polygon_edge_ids_run2

    payload = {
        "gate": "phase3_polygon_clip_telemetry",
        "scene_id": artifacts.scene_id,
        "entity_count": entity_count,
        "surface_counts": {
            "total": len(artifacts.structural_surfaces),
            "polygon_present": len(polygon_present_uids),
            "polygon_none": len(polygon_none_uids),
            "by_surface_type": dict(sorted(Counter(
                s.surface_type for s in artifacts.structural_surfaces
            ).items())),
        },
        "polygon_on_plane_drift": {
            "max_polygon_off_plane_drift_m": max_off_plane_drift,
            "polygon_on_plane_tolerance_m": POLYGON_ON_PLANE_TOL_M,
            "within_tolerance": max_off_plane_drift <= POLYGON_ON_PLANE_TOL_M,
        },
        "config": {
            "plane_mode": {
                "use_polygon_clip": False,
                "extractor_version": PLANE_MODE_VERSION,
                "floor_threshold_m": plane_cfg.floor_threshold_m,
                "wall_threshold_m": plane_cfg.wall_threshold_m,
                "ceiling_threshold_m": plane_cfg.ceiling_threshold_m,
                "include_synth_fallback": plane_cfg.include_synth_fallback,
            },
            "polygon_mode": {
                "use_polygon_clip": True,
                "extractor_version": POLYGON_CLIPPED_VERSION,
                "floor_threshold_m": polygon_cfg.floor_threshold_m,
                "wall_threshold_m": polygon_cfg.wall_threshold_m,
                "ceiling_threshold_m": polygon_cfg.ceiling_threshold_m,
                "include_synth_fallback": polygon_cfg.include_synth_fallback,
            },
        },
        "edge_counts": {
            "plane_mode_total": plane_diag.logical_edges_total,
            "polygon_mode_total": polygon_diag.logical_edges_total,
            "delta_polygon_minus_plane": (
                polygon_diag.logical_edges_total - plane_diag.logical_edges_total
            ),
            "by_surface_type": by_surface_type_combined,
        },
        "flipped_edges": {
            "near_to_not_near_count": len(flipped_near_to_not_near),
            "near_to_not_near": flipped_near_to_not_near,
            "not_near_to_near_count": len(flipped_not_near_to_near),
            "not_near_to_near": flipped_not_near_to_near,
        },
        "subset_claim_for_surfaces_with_polygons": {
            "applicable_surface_count": len(polygon_present_uids),
            "plane_mode_pair_count_on_polygoned_surfaces": (
                len(plane_pairs_polygoned)
            ),
            "polygon_mode_pair_count_on_polygoned_surfaces": (
                len(polygon_pairs_polygoned)
            ),
            "violation_count": len(subset_violations),
            "subset_holds": len(subset_violations) == 0,
            "violations": [
                {"entity_uid": e, "surface_uid": s}
                for e, s in subset_violations
            ],
        },
        "phase2_byte_equivalence": {
            "method": (
                "GraphBuilder.bundle_hash on a NEAR_SURFACE-only run "
                "against real Replica room_0 artifacts"
            ),
            "default_bundle_hash": bundle_hash_default,
            "explicit_false_bundle_hash": bundle_hash_explicit_false,
            "polygon_mode_bundle_hash": bundle_hash_polygon,
            "default_equals_explicit_false": (
                bundle_hash_default == bundle_hash_explicit_false
            ),
            "polygon_mode_differs": (
                bundle_hash_default != bundle_hash_polygon
            ),
        },
        "determinism": {
            "method": (
                "extractor invoked twice with the same config; sorted "
                "edge_id lists compared"
            ),
            "plane_mode_two_runs_byte_equal": two_runs_byte_equal,
            "polygon_mode_two_runs_byte_equal": polygon_two_runs_byte_equal,
            "plane_mode_edge_id_count": len(plane_edge_ids_run1),
            "polygon_mode_edge_id_count": len(polygon_edge_ids_run1),
        },
        "notes": [
            "Closeout reference for P3.04: A5 (distinct extractor version + "
            "richer evidence for opt-in mode) materially landed in P3.03, "
            "verified by tests/relations/test_near_surface_polygon.py. The "
            "polygon-mode bundle_hash here differs from plane-mode by "
            "design — use_polygon_clip=True is hashed, plane vs polygon are "
            "honestly distinct modes.",
            "Per A6, polygon-mode edges are expected to be a subset of "
            "plane-mode edges for surfaces with polygon present. A non-zero "
            "violation_count would indicate a geometry bug, not a tuning "
            "issue, and must block Phase 3 closeout.",
            "Polygon=None surfaces are excluded from the subset claim "
            "because the dispatcher falls back to bbox_to_plane for those, "
            "making the edge sets coincide on them by construction.",
            "Phase 2 NEAR_SURFACE remains the default. This artifact is "
            "decision input for P3.06 (promote opt-in to default or keep "
            "opt-in); no promotion happens in P3.05.",
            "No timestamp is recorded; the file is byte-stable when the "
            "underlying scene, thresholds, and geometry helpers are "
            "unchanged. Drift against the committed copy is signal.",
        ],
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rel = ARTIFACT_PATH.relative_to(REPO_ROOT)
    print(f"Wrote {rel}")
    print(f"  entities:                       {entity_count}")
    print(f"  surfaces (polygon / none):      "
          f"{len(polygon_present_uids)} / {len(polygon_none_uids)}")
    print(f"  plane_mode edges:               {plane_diag.logical_edges_total}")
    print(f"  polygon_mode edges:             {polygon_diag.logical_edges_total}")
    print(f"  flipped near -> not_near:       {len(flipped_near_to_not_near)}")
    print(f"  flipped not_near -> near:       {len(flipped_not_near_to_near)}")
    print(f"  subset violations (polygoned):  {len(subset_violations)}")
    print(f"  default == explicit-False hash: "
          f"{bundle_hash_default == bundle_hash_explicit_false}")
    print(f"  polygon-mode hash differs:      "
          f"{bundle_hash_default != bundle_hash_polygon}")
    print(f"  plane-mode determinism:         {two_runs_byte_equal}")
    print(f"  max polygon-off-plane drift:    {max_off_plane_drift!r}")
    print(f"  polygon-on-plane tolerance:     {POLYGON_ON_PLANE_TOL_M!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

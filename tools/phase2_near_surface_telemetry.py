"""Phase 2 P2.09 — NEAR_SURFACE telemetry on Replica room_0.

Records how many NEAR_SURFACE edges the canonical extractor emits and
how that interacts with the existing combined-graph density. The output
artifact is the input to the P2.10 version-aware guardrail decision —
the extractor is NOT wired into any default builder run yet.

Output:
  scenes/replica_room_0/eval/phase2_near_surface_telemetry.json

Determinism: stable. No timestamp — the artifact only changes when the
underlying data or thresholds change. Diff churn against committed
copies is therefore signal, not noise.

Skip-on-missing: enriched-v2 importer output must exist.

Run: python tools/phase2_near_surface_telemetry.py
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
from graph.builder import SPARSE_DENSITY_LIMIT
from graph.relations.directional import DirectionalConfig, DirectionalExtractor
from graph.relations.proximity import ProximityConfig, ProximityExtractor
from graph.relations.surface import (
    SurfaceProximityConfig,
    SurfaceProximityExtractor,
)
from representations.mesh import MeshRepresentation


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
REPLICA_V2_DIR = REPLICA_SCENE_DIR / "enriched" / "v2"
ARTIFACT_PATH = REPLICA_SCENE_DIR / "eval" / "phase2_near_surface_telemetry.json"


def main() -> int:
    if not (REPLICA_V2_DIR / "scene_graph.json").exists():
        print("Refusing: enriched-v2 importer output is missing.")
        return 1

    capture = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    representation = MeshRepresentation(
        bundle=OracleReplicaAdapter().reconstruct(
            capture, ReconstructionConfig(name="oracle_replica", version="0.1", params={}),
        ),
    )
    artifacts = OracleReplicaExtractor(enriched_v2_path=REPLICA_V2_DIR).extract(
        representation,
        InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )
    entity_count = len(artifacts.entities)

    # Existing relation families (no NEAR_SURFACE).
    _de, dd = DirectionalExtractor().extract(artifacts, DirectionalConfig(mode="sparse"))
    _v1, v1d = ProximityExtractor().extract(
        artifacts, ProximityConfig(mode="sparse", sparse_near_threshold=1.0),
    )
    _v2, v2d = ProximityExtractor().extract(
        artifacts, ProximityConfig(
            mode="sparse", sparse_version=2, sparse_near_threshold=1.0,
        ),
    )

    cfg = SurfaceProximityConfig()
    ns_edges, ns_diag = SurfaceProximityExtractor().extract(artifacts, cfg)
    skipped_synth_fallback_surfaces = sum(
        1
        for surface in artifacts.structural_surfaces
        if surface.source == "synth_bbox_fallback" and not cfg.include_synth_fallback
    )
    by_surface_type: Counter = Counter()
    by_surface_uid: Counter = Counter()
    for e in ns_edges:
        by_surface_uid[e.target.uid] += 1
        by_surface_type[e.evidence.get("surface_type", "?")] += 1

    base_v1_logical = dd.logical_edges_total + v1d.logical_edges_total
    base_v2_logical = dd.logical_edges_total + v2d.logical_edges_total
    combined_v1_plus_ns = base_v1_logical + ns_diag.logical_edges_total
    combined_v2_plus_ns = base_v2_logical + ns_diag.logical_edges_total

    payload = {
        "gate": "phase2_near_surface_telemetry",
        "scene_id": artifacts.scene_id,
        "entity_count": entity_count,
        "config": {
            "floor_threshold_m": cfg.floor_threshold_m,
            "wall_threshold_m": cfg.wall_threshold_m,
            "ceiling_threshold_m": cfg.ceiling_threshold_m,
            "include_synth_fallback": cfg.include_synth_fallback,
            "distance_metric": "bbox_to_plane",
            "threshold_comparison": "less_than_or_equal",
        },
        "near_surface": {
            "physical_edges": ns_diag.physical_edges_total,
            "logical_edges": ns_diag.logical_edges_total,
            "edges_per_surface_type": dict(sorted(by_surface_type.items())),
            "edges_per_surface_uid": dict(sorted(by_surface_uid.items())),
            "skipped_synth_fallback_surfaces": skipped_synth_fallback_surfaces,
        },
        "combined_density_versus_phase1_guardrail": {
            "phase1_sparse_density_limit": SPARSE_DENSITY_LIMIT,
            "directional_sparse_logical": dd.logical_edges_total,
            "proximity_v1_logical": v1d.logical_edges_total,
            "proximity_v2_logical": v2d.logical_edges_total,
            "near_surface_logical": ns_diag.logical_edges_total,
            "combined_v1_plus_near_surface": {
                "logical_edges": combined_v1_plus_ns,
                "density_ratio_per_entity": combined_v1_plus_ns / max(1, entity_count),
                "exceeds_phase1_guardrail": (
                    combined_v1_plus_ns / max(1, entity_count) > SPARSE_DENSITY_LIMIT
                ),
            },
            "combined_v2_plus_near_surface": {
                "logical_edges": combined_v2_plus_ns,
                "density_ratio_per_entity": combined_v2_plus_ns / max(1, entity_count),
                "exceeds_phase1_guardrail": (
                    combined_v2_plus_ns / max(1, entity_count) > SPARSE_DENSITY_LIMIT
                ),
            },
        },
        "notes": [
            "Recorded for the P2.10 version-aware guardrail decision (per the "
            "P2.09 sign-off). NEAR_SURFACE is NOT yet wired into any default "
            "builder run.",
            "Per A8, the frozen smoke list at "
            "eval/questions/phase2_near_surface_smoke.json must continue to "
            "pass; any change to thresholds/policy that breaks it is a "
            "behavior change, not a tuning step.",
            "Thresholds (0.05/0.30/0.10 m) are provisional Replica-calibrated "
            "config values per Q4 — not generalization evidence.",
        ],
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    rel = ARTIFACT_PATH.relative_to(REPO_ROOT)
    print(f"Wrote {rel}")
    print(f"  entities:                       {entity_count}")
    print(f"  NEAR_SURFACE logical edges:     {ns_diag.logical_edges_total}")
    print(f"  per surface_type:               {dict(by_surface_type)}")
    print(f"  combined v1+NS density/entity:  "
          f"{combined_v1_plus_ns / max(1, entity_count):.3f}")
    print(f"  combined v2+NS density/entity:  "
          f"{combined_v2_plus_ns / max(1, entity_count):.3f}")
    print(f"  Phase 1 guardrail:              {SPARSE_DENSITY_LIMIT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

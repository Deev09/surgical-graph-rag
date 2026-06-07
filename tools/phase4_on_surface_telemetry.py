"""Phase 4 P4.05 — ON_SURFACE coverage telemetry on Replica room_0.

Honest, floor-only coverage report for the ON_SURFACE rest-contact relation
and its derived SUPPORTS view. This is telemetry, NOT a benchmark claim:

  - ON_SURFACE only emits on support-capable (up-facing) surfaces — in
    Replica room_0 that is the floor. wall/ceiling on_surface_edges = 0 is
    BY DESIGN (not support-capable under Design B), not a measured absence.
  - support_facts_total == on_surface_edges_total (clean inverse, P4.03).
  - materialized_supports_edges_total stays 0 (P4.03 invariant, kept visible).
  - "deferred" QA items (table/chair/wall) are NOT empty — they are
    unanswerable because the geometry (EntitySurface / wall-attachment
    relation) does not exist yet (D1/D2).

Output:
  scenes/replica_room_0/eval/phase4_on_surface_telemetry.json

Determinism: stable. No timestamp; sorted keys + trailing newline. Re-running
produces a byte-identical file (tested). Diff churn is signal, not noise.

Skip-on-missing: enriched-v2 importer output must exist.

Run: python tools/phase4_on_surface_telemetry.py
"""
from __future__ import annotations

import json
import math
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
from graph.builder import ExtractorRun, build_graph
from graph.relations.on_surface import (
    ON_SURFACE_VERSION, OnSurfaceConfig, OnSurfaceExtractor,
)
from graph.views.support import support_facts
from representations.mesh import MeshRepresentation


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
REPLICA_V2_DIR = REPLICA_SCENE_DIR / "enriched" / "v2"
ARTIFACT_PATH = REPLICA_SCENE_DIR / "eval" / "phase4_on_surface_telemetry.json"

CANONICAL_SURFACE_TYPES = ("floor", "wall", "ceiling")


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


def _up_from_gravity(gravity):
    gx, gy, gz = gravity
    mag = math.sqrt(gx * gx + gy * gy + gz * gz)
    return (-gx / mag, -gy / mag, -gz / mag)


def _type_is_support_capable(artifacts, surface_type, up, cos_max) -> bool:
    """Measured (not assumed): does any surface of this type have an
    up-facing normal within max_tilt?"""
    for s in artifacts.structural_surfaces:
        if s.surface_type != surface_type:
            continue
        a, b, c = s.plane.a, s.plane.b, s.plane.c
        nmag = math.sqrt(a * a + b * b + c * c)
        if nmag == 0.0:
            continue
        if (a * up[0] + b * up[1] + c * up[2]) / nmag >= cos_max:
            return True
    return False


def main() -> int:
    if not (REPLICA_V2_DIR / "scene_graph.json").exists():
        print("Refusing: enriched-v2 importer output is missing.")
        return 1

    artifacts = _build_replica_artifacts()
    cfg = OnSurfaceConfig()
    bundle, _diag = build_graph(
        artifacts,
        [ExtractorRun(OnSurfaceExtractor(), cfg)],
        density_policy="phase2_telemetry_only",
    )

    surface_type_by_uid = {
        s.surface_uid: s.surface_type for s in artifacts.structural_surfaces
    }
    on_edges = [e for e in bundle.edges if e.type == "ON_SURFACE"]
    materialized_supports = sum(1 for e in bundle.edges if e.type == "SUPPORTS")
    facts = support_facts(bundle)  # raises if the invariant is violated

    up = _up_from_gravity(artifacts.frame.gravity)
    cos_max = math.cos(math.radians(cfg.max_tilt_deg))

    by_surface_type: dict[str, dict] = {}
    for stype in CANONICAL_SURFACE_TYPES:
        edges_of_type = [
            e for e in on_edges
            if surface_type_by_uid.get(e.target.uid) == stype
        ]
        unique_entities = len({e.source.uid for e in edges_of_type})
        by_surface_type[stype] = {
            "on_surface_edges": len(edges_of_type),
            "unique_entities": unique_entities,
            "support_capable": _type_is_support_capable(
                artifacts, stype, up, cos_max,
            ),
        }

    payload = {
        "scene_id": artifacts.scene_id,
        "phase": "P4.05",
        "artifact_kind": "on_surface_coverage_telemetry",
        "schema_version": 1,
        "inputs": {
            "entity_bundle_hash": artifacts.bundle_hash,
            "graph_bundle_hash": bundle.bundle_hash,
            "extractor": "on_surface",
            "extractor_version": ON_SURFACE_VERSION,
            "config": {
                "contact_threshold_m": cfg.contact_threshold_m,
                "penetration_tolerance_m": cfg.penetration_tolerance_m,
                "max_tilt_deg": cfg.max_tilt_deg,
                "footprint_tolerance_m": cfg.footprint_tolerance_m,
                "near_surface_threshold_m": cfg.near_surface_threshold_m,
                "include_synth_fallback": cfg.include_synth_fallback,
            },
        },
        "coverage_summary": {
            "on_surface_edges_total": len(on_edges),
            "support_facts_total": len(facts),
            "materialized_supports_edges_total": materialized_supports,
            "by_surface_type": by_surface_type,
        },
        "qa_readiness": {
            "what_is_on_the_floor": "answerable",
            "what_is_on_the_table": "deferred_needs_entity_surface",
            "what_is_on_the_chair": "deferred_needs_entity_surface",
            "what_is_against_the_wall": "deferred_needs_wall_attachment_relation",
        },
        "deferred_semantics": {
            "deferred_not_zero": True,
            "note": (
                "A 'deferred' QA item is NOT an empty result. Table/chair/wall "
                "support is unanswerable in P4 because the geometry "
                "(EntitySurface / wall-attachment relation) does not exist "
                "yet, not because nothing is there."
            ),
        },
        "interpretation_limits": [
            "floor support only (Replica room_0 has no furniture-top geometry)",
            "deferred does not mean empty",
            "wall/ceiling on_surface_edges = 0 is by-design not-support-capable "
            "under Design B, not a measured absence",
            "support_facts_total equals on_surface_edges_total by construction "
            "(clean inverse); this is not independent corroboration",
            "not a v1 benchmark improvement claim",
        ],
        "determinism": {
            "timestamp_free": True,
        },
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    rel = ARTIFACT_PATH.relative_to(REPO_ROOT)
    print(f"Wrote {rel}")
    print(f"  ON_SURFACE edges total:          {len(on_edges)}")
    print(f"  support facts total:             {len(facts)}")
    print(f"  materialized SUPPORTS edges:     {materialized_supports}")
    for stype in CANONICAL_SURFACE_TYPES:
        b = by_surface_type[stype]
        print(
            f"  {stype:8s} edges={b['on_surface_edges']:3d} "
            f"entities={b['unique_entities']:3d} "
            f"support_capable={b['support_capable']}"
        )
    print("  qa: floor=answerable; table/chair/wall=deferred (NOT empty)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 2 exit gate (P2.10).

Runs G1–G7 against the Replica enriched-v2 bundle and writes a
deterministic report:

  scenes/replica_room_0/eval/phase2_exit_gate_report.json

Blocking gates (failure → exit 1):
  G1  Structural surfaces emitted: ≥1 floor, ≥2 walls, ≥1 ceiling, all
      with source != "synth_bbox_fallback".
  G2  World-frame OBBs: every v2-path entity has a non-None bbox_obb.
  G3  Phase 1 compat reproduction byte-equal (re-runs tools/phase1_gates.py
      compat reproduction; diff must remain empty 5414/5414).
  G4  Phase 2 candidate build is deterministic + round-trips: two builds
      yield identical bundle_hash AND dump→load equals via array_aware_equal.
  G5  Frozen NEAR_SURFACE smoke list passes (eval/questions/
      phase2_near_surface_smoke.json) — ≥4 near AND ≥4 not_near each match.
  G7  Builder C1: structural_surfaces field equals the entity bundle's
      surface set; edges referencing unknown entity/surface UIDs are
      rejected with GraphBuildError.

Telemetry only (recorded, never blocks):
  G6  Combined sparse-v2 + NEAR_SURFACE density vs Phase 1 cap of 14.
      The Phase 2 candidate uses density_policy="phase2_telemetry_only";
      G6 records actual_ratio and exceeds_phase1_cap.

Deterministic artifact: no timestamp; any diff churn is a behavior
change. Phase 1 eval artifacts are NOT touched.

Skip-on-missing: enriched-v2 importer output must exist; otherwise the
gate exits with a clear message and no artifact.

Run: python tools/phase2_exit_gate.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.base import ReconstructionConfig
from adapters.oracle_replica import OracleReplicaAdapter, build_replica_capture_bundle
from common.equality import array_aware_equal
from extractors.base import InstanceExtractorConfig
from extractors.oracle_replica import OracleReplicaExtractor
from graph.builder import (
    ExtractorRun, GraphBuildError, SPARSE_DENSITY_LIMIT, build_graph,
)
from graph.relations.directional import DirectionalConfig, DirectionalExtractor
from graph.relations.proximity import ProximityConfig, ProximityExtractor
from graph.relations.surface import SurfaceProximityConfig, SurfaceProximityExtractor
from graph.schema import GraphRef, SurfaceRecord
from graph.serde import dump_scene_graph_bundle, load_scene_graph_bundle
from representations.mesh import MeshRepresentation


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
REPLICA_V2_DIR = REPLICA_SCENE_DIR / "enriched" / "v2"
LEGACY_RELATIONS_PATH = REPLICA_SCENE_DIR / "computed_relations" / "scene_graph.json"
SMOKE_LIST_PATH = REPO_ROOT / "eval" / "questions" / "phase2_near_surface_smoke.json"
ARTIFACT_PATH = REPLICA_SCENE_DIR / "eval" / "phase2_exit_gate_report.json"


def _real_replica_artifacts():
    capture = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    representation = MeshRepresentation(
        bundle=OracleReplicaAdapter().reconstruct(
            capture, ReconstructionConfig(name="oracle_replica", version="0.1", params={}),
        ),
    )
    return OracleReplicaExtractor(enriched_v2_path=REPLICA_V2_DIR).extract(
        representation,
        InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )


def _phase2_runs():
    return [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse")),
        ExtractorRun(
            ProximityExtractor(),
            ProximityConfig(mode="sparse", sparse_version=2),
        ),
        ExtractorRun(SurfaceProximityExtractor(), SurfaceProximityConfig()),
    ]


def _gate_g1(artifacts) -> tuple[bool, dict]:
    counts = Counter(s.surface_type for s in artifacts.structural_surfaces)
    sources = sorted({s.source for s in artifacts.structural_surfaces})
    any_synth = any(s.source == "synth_bbox_fallback" for s in artifacts.structural_surfaces)
    passed = (
        counts.get("floor", 0) >= 1
        and counts.get("wall", 0) >= 2
        and counts.get("ceiling", 0) >= 1
        and not any_synth
    )
    return passed, {
        "counts_per_type": dict(sorted(counts.items())),
        "sources_observed": sources,
        "synth_fallback_present": any_synth,
    }


def _gate_g2(artifacts) -> tuple[bool, dict]:
    missing = [
        e.identity.object_uid for e in artifacts.entities if e.bbox_obb is None
    ]
    return not missing, {
        "entity_count": len(artifacts.entities),
        "with_bbox_obb": len(artifacts.entities) - len(missing),
        "missing_bbox_obb": missing[:10],
    }


def _gate_g3() -> tuple[bool, dict]:
    """Replicates the P1.08 compat reproduction check inline to avoid
    overwriting the Phase 1 eval artifact (which would re-stamp its
    timestamp)."""
    from graph.relations.base import edge_key
    from graph.relations.directional import DirectionalExtractor, DirectionalConfig
    from graph.relations.proximity import ProximityExtractor, ProximityConfig
    capture = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    representation = MeshRepresentation(
        bundle=OracleReplicaAdapter().reconstruct(
            capture, ReconstructionConfig(name="oracle_replica", version="0.1", params={}),
        ),
    )
    phase1_artifacts = OracleReplicaExtractor().extract(
        representation,
        InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )
    bundle, _diag = build_graph(phase1_artifacts, [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="compat")),
        ExtractorRun(ProximityExtractor(), ProximityConfig(mode="compat")),
    ])
    produced_keys = {edge_key(e) for e in bundle.edges}
    legacy = json.loads(LEGACY_RELATIONS_PATH.read_text(encoding="utf-8"))
    legacy_keys = {
        ("entity", r["source"], r["type"], "entity", r["target"])
        for r in legacy["relations"]
    }
    missing = sorted(legacy_keys - produced_keys)
    extra = sorted(produced_keys - legacy_keys)
    passed = not missing and not extra and len(produced_keys) == 5414
    return passed, {
        "produced_total": len(produced_keys),
        "expected_total": len(legacy_keys),
        "missing_count": len(missing),
        "extra_count": len(extra),
    }


def _gate_g4(artifacts) -> tuple[bool, dict]:
    bundle_a, _ = build_graph(
        artifacts, _phase2_runs(), density_policy="phase2_telemetry_only",
    )
    bundle_b, _ = build_graph(
        artifacts, _phase2_runs(), density_policy="phase2_telemetry_only",
    )
    bundle_hashes_match = bundle_a.bundle_hash == bundle_b.bundle_hash
    round_trip_ok = False
    with tempfile.TemporaryDirectory() as td:
        dump_dir = Path(td) / "graph"
        dump_scene_graph_bundle(bundle_a, dump_dir)
        loaded = load_scene_graph_bundle(dump_dir)
        round_trip_ok = array_aware_equal(bundle_a, loaded)
    return (bundle_hashes_match and round_trip_ok), {
        "two_run_hash_match": bundle_hashes_match,
        "round_trip_equal": round_trip_ok,
        "bundle_hash": bundle_a.bundle_hash,
    }


def _gate_g5(artifacts) -> tuple[bool, dict]:
    smoke = json.loads(SMOKE_LIST_PATH.read_text(encoding="utf-8"))
    cfg = SurfaceProximityConfig(
        floor_threshold_m=smoke["thresholds_m"]["floor"],
        wall_threshold_m=smoke["thresholds_m"]["wall"],
        ceiling_threshold_m=smoke["thresholds_m"]["ceiling"],
        include_synth_fallback=smoke["policy"]["include_synth_fallback"],
    )
    edges, _ = SurfaceProximityExtractor().extract(artifacts, cfg)
    emitted = {(e.source.uid, e.target.uid) for e in edges}
    failures = []
    for case in smoke["cases"]:
        pair = (case["entity_uid"], case["surface_uid"])
        present = pair in emitted
        if case["expectation"] == "near" and not present:
            failures.append(f"missing NEAR: {pair}")
        if case["expectation"] == "not_near" and present:
            failures.append(f"unexpected NEAR: {pair}")
    near_count = sum(1 for c in smoke["cases"] if c["expectation"] == "near")
    not_near_count = sum(1 for c in smoke["cases"] if c["expectation"] == "not_near")
    enough_cases = near_count >= 4 and not_near_count >= 4
    return (not failures and enough_cases), {
        "near_cases": near_count,
        "not_near_cases": not_near_count,
        "failures": failures,
    }


def _gate_g6(artifacts) -> dict:
    """Telemetry only. Records combined v2 + NEAR_SURFACE density vs
    Phase 1 cap; never blocks."""
    bundle, diag = build_graph(
        artifacts, _phase2_runs(), density_policy="phase2_telemetry_only",
    )
    ratio = diag.density_ratio if diag.density_ratio is not None else 0.0
    return {
        "logical_edges_total": diag.logical_edges_total,
        "entity_count": len(artifacts.entities),
        "density_ratio_per_entity": ratio,
        "phase1_sparse_density_limit": SPARSE_DENSITY_LIMIT,
        "exceeds_phase1_cap": ratio > SPARSE_DENSITY_LIMIT,
        "density_policy_used": diag.density_policy,
        "note": (
            "Phase 1 cap stays blocking for sparse-v1 only. "
            "Phase 2 candidate explicitly opts into telemetry."
        ),
    }


def _gate_g7(artifacts) -> tuple[bool, dict]:
    """Two assertions: (1) bundle.structural_surfaces equals the entity
    bundle's surface set; (2) an unknown surface_uid in an edge is
    rejected with GraphBuildError."""
    bundle, _diag = build_graph(
        artifacts, _phase2_runs(), density_policy="phase2_telemetry_only",
    )
    expected_records = [
        SurfaceRecord(
            uid=s.surface_uid,
            surface_type=s.surface_type,
            plane=s.plane,
            polygon=list(s.polygon) if s.polygon is not None else None,
            source=s.source,
            confidence=s.confidence,
        )
        for s in artifacts.structural_surfaces
    ]
    retention_ok = expected_records == bundle.structural_surfaces
    refs_derived_ok = (
        bundle.structural_surface_refs == [s.uid for s in bundle.structural_surfaces]
    )

    # Inject phantom edges via a tiny stub extractor to confirm rejection.
    from dataclasses import dataclass
    from graph.relations.base import RelationExtractorDiagnostics
    from graph.schema import Edge

    @dataclass(frozen=True)
    class _PhantomConfig:
        mode: str = "sparse"

    class _PhantomExtractor:
        name = "phantom"
        version = "0.1"
        edge_types = frozenset({"NEAR_SURFACE"})

        def __init__(self, edge):
            self.edge = edge

        def extract(self, entities, config):
            return [self.edge], RelationExtractorDiagnostics(
                extractor=self.name, version=self.version, mode=config.mode,
                physical_edges_per_type={"NEAR_SURFACE": 1},
                physical_edges_total=1, logical_edges_total=1,
                rejections_per_type={}, rejection_samples=[], runtime_ms=0,
            )

    known_entity_uid = artifacts.entities[0].identity.object_uid
    unknown_surface_edge = Edge(
        edge_id="e_phantom_surface",
        source=GraphRef(kind="entity", uid=known_entity_uid),
        type="NEAR_SURFACE",
        target=GraphRef(kind="surface", uid="ghost_surface"),
        frame="world", weight=1.0, confidence=1.0,
        extractor="phantom", extractor_version="0.1", evidence={},
    )
    unknown_entity_edge = Edge(
        edge_id="e_phantom_entity",
        source=GraphRef(kind="entity", uid="ghost_entity"),
        type="NEAR",
        target=GraphRef(kind="entity", uid=known_entity_uid),
        frame="world", weight=1.0, confidence=1.0,
        extractor="phantom", extractor_version="0.1", evidence={},
    )

    def rejected(edge, expected_text: str) -> bool:
        try:
            build_graph(
                artifacts,
                [ExtractorRun(_PhantomExtractor(edge), _PhantomConfig())],
                density_policy="phase2_telemetry_only",
            )
        except GraphBuildError as e:
            return expected_text in str(e)
        return False

    unknown_surface_rejected = rejected(unknown_surface_edge, "unknown surface")
    unknown_entity_rejected = rejected(unknown_entity_edge, "unknown entity")
    return (
        retention_ok
        and refs_derived_ok
        and unknown_surface_rejected
        and unknown_entity_rejected
    ), {
        "full_surface_records_retained": retention_ok,
        "refs_derived_from_full_set": refs_derived_ok,
        "unknown_surface_ref_rejected": unknown_surface_rejected,
        "unknown_entity_ref_rejected": unknown_entity_rejected,
        "expected_surface_count": len(expected_records),
        "actual_surface_count": len(bundle.structural_surfaces),
    }


def main() -> int:
    if not (REPLICA_V2_DIR / "scene_graph.json").exists():
        print("Refusing: enriched-v2 importer output is missing.")
        return 1
    if not LEGACY_RELATIONS_PATH.exists():
        print(f"Refusing: legacy relations missing at {LEGACY_RELATIONS_PATH}.")
        return 1

    artifacts = _real_replica_artifacts()

    g1_pass, g1_detail = _gate_g1(artifacts)
    g2_pass, g2_detail = _gate_g2(artifacts)
    g3_pass, g3_detail = _gate_g3()
    g4_pass, g4_detail = _gate_g4(artifacts)
    g5_pass, g5_detail = _gate_g5(artifacts)
    g6_telemetry = _gate_g6(artifacts)
    g7_pass, g7_detail = _gate_g7(artifacts)

    blocking = {
        "G1_structural_surfaces": (g1_pass, g1_detail),
        "G2_world_frame_obbs": (g2_pass, g2_detail),
        "G3_phase1_compat_reproduction": (g3_pass, g3_detail),
        "G4_deterministic_and_replayable": (g4_pass, g4_detail),
        "G5_near_surface_smoke": (g5_pass, g5_detail),
        "G7_builder_structural_completeness": (g7_pass, g7_detail),
    }
    all_blocking_pass = all(v[0] for v in blocking.values())

    payload = {
        "gate": "phase2_exit_gate",
        "scene_id": artifacts.scene_id,
        "entity_count": len(artifacts.entities),
        "blocking_gates": {
            name: {"pass": p, **detail} for name, (p, detail) in blocking.items()
        },
        "telemetry_gates": {
            "G6_sparse_v2_plus_near_surface_density": g6_telemetry,
        },
        "limitations_recorded_for_phase3": [
            "NEAR_SURFACE measures distance to the INFINITE plane, not "
            "clipped to polygon extents. An entity above the same plane "
            "but well outside the polygon footprint still registers as "
            "NEAR. Acceptable for Phase 2; Phase 3 may add polygon-clipped "
            "variants.",
            "OBB-to-OBB distance is deferred to Phase 3 (P2.07 A5). Phase 2 "
            "uses exact AABB surface distance.",
        ],
        "policy_decisions_recorded": [
            "Phase 1 density cap (<=14) stays BLOCKING for sparse-v1 only.",
            "Phase 2 candidate (sparse-v2 + NEAR_SURFACE) uses "
            "density_policy='phase2_telemetry_only'. The cap is recorded, "
            "not silently raised.",
            "Canonical NEAR_SURFACE extractor excludes synth_bbox_fallback "
            "surfaces; opt-in via include_synth_fallback=True only.",
        ],
        "overall_blocking_pass": all_blocking_pass,
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\nPhase 2 exit gate report → {ARTIFACT_PATH.relative_to(REPO_ROOT)}")
    for name, (p, _detail) in blocking.items():
        marker = "PASS" if p else "FAIL"
        print(f"  [{marker}] {name}")
    print(
        f"  [TELEMETRY] G6_sparse_v2_plus_near_surface_density "
        f"= {g6_telemetry['density_ratio_per_entity']:.3f}/entity "
        f"(cap {g6_telemetry['phase1_sparse_density_limit']}; "
        f"exceeds={g6_telemetry['exceeds_phase1_cap']})"
    )
    print(f"\nOverall blocking: {'PASS' if all_blocking_pass else 'FAIL'}")
    return 0 if all_blocking_pass else 1


if __name__ == "__main__":
    sys.exit(main())

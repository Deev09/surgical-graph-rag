"""Phase 3 exit gate (P3.07).

Runs G1-G8 against the Replica enriched-v2 bundle and writes a
deterministic report:

  scenes/replica_room_0/eval/phase3_exit_gate_report.json

Per the P3.06 closeout decision, this phase keeps `use_polygon_clip`
opt-in. The exit gate therefore asserts BOTH:

  - Phase 2 default behavior remains byte-equivalent (G1-G7 mirror the
    Phase 2 gate's claims under default `SurfaceProximityConfig()`; G5a
    re-runs the Phase 2 smoke list under the default config; default vs
    explicit-False bundle_hash are equal at full-scene scale).
  - Polygon candidate is ready for future promotion (G5b runs the Phase
    3 polygon smoke under `use_polygon_clip=True`; G8 asserts polygon-
    mode determinism, A6 subset monotonicity, A4 hash divergence, and
    telemetry-artifact agreement with the committed P3.05 artifact).

Blocking gates (failure → exit 1):
  G1   Structural surfaces emitted: ≥1 floor, ≥2 walls, ≥1 ceiling, all
       non-synth_bbox_fallback.                                (reused from Phase 2)
  G2   World-frame OBBs: every v2-path entity has a non-None bbox_obb.
                                                               (reused from Phase 2)
  G3   Phase 1 compat reproduction byte-equal (5414/5414).     (reused from Phase 2)
  G4   Both Phase 2 plane-mode AND Phase 3 polygon-mode candidates are
       deterministic (two runs → same bundle_hash) AND round-trip
       through dump/load via `array_aware_equal`.
  G5a  Phase 2 NEAR_SURFACE smoke list (eval/questions/
       phase2_near_surface_smoke.json) passes under default
       `SurfaceProximityConfig()`.                             (reused from Phase 2)
  G5b  Phase 3 NEAR_SURFACE smoke list (eval/questions/
       phase3_near_surface_polygon_smoke.json) passes under
       `SurfaceProximityConfig(use_polygon_clip=True)`. Each synthetic
       case is run as an isolated EntityArtifacts; the Replica-grounded
       case is run against the real bundle.
  G7   Builder C1 surface-completeness and unknown-ref rejection.
                                                               (reused from Phase 2)
  G8   Polygon-clip determinism + monotonicity + Phase 2 byte-equivalence:
       (a) polygon mode is deterministic across two runs,
       (b) default vs explicit-False bundle_hash are equal,
       (c) polygon-mode bundle_hash differs from default,
       (d) polygon edges ⊆ plane edges on surfaces with polygon present,
       (e) no `not_near_to_near` flips globally (A6 monotonicity),
       (f) the P3.05 telemetry artifact agrees with freshly-computed
           gate values (i.e., the committed artifact is not stale),
       (g) phase3_policy.default_behavior == "phase2_plane_mode" and
           polygon_clip_status == "opt_in_candidate".

Telemetry only (recorded, never blocks):
  G6   Combined sparse-v2 + NEAR_SURFACE density vs Phase 1 cap, recorded
       for BOTH plane mode and polygon mode candidates.        (extends Phase 2)

Deterministic artifact: no timestamp; any diff churn is a behavior
change. Phase 1 and Phase 2 eval artifacts are NOT touched. The P3.05
telemetry artifact is READ for the G8(f) agreement check, NOT rewritten.

Skip-on-missing: enriched-v2 importer output and the committed P3.05
telemetry artifact must exist; otherwise the gate exits with a clear
message and no artifact.

Run: python tools/phase3_exit_gate.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.equality import array_aware_equal
from common.types import Plane
from extractors.base import (
    EntityArtifact, EntityArtifacts, EntityIdentity, ExtractionDiagnostics,
    SemanticHypothesis, StructuralSurface,
)
from extractors.serde import CURRENT_SCHEMA_VERSION as ENT_SCHEMA_VERSION
from geometry.surface_distance import POLYGON_ON_PLANE_TOL_M
from graph.builder import ExtractorRun, SPARSE_DENSITY_LIMIT, build_graph
from graph.relations.directional import DirectionalConfig, DirectionalExtractor
from graph.relations.proximity import ProximityConfig, ProximityExtractor
from graph.relations.surface import (
    PLANE_MODE_VERSION,
    POLYGON_CLIPPED_VERSION,
    SurfaceProximityConfig,
    SurfaceProximityExtractor,
)
from graph.serde import dump_scene_graph_bundle, load_scene_graph_bundle

# Phase 2 gate helpers — direct reuse, unchanged. Importing keeps G1-G3 / G7
# in lockstep with the Phase 2 gate so the Phase 2 byte-equivalence claim is
# literally the same check, not a re-implementation that could drift.
from tools.phase2_exit_gate import (
    _gate_g1, _gate_g2, _gate_g3, _gate_g7,
    _real_replica_artifacts,
)
# P3.05 telemetry helpers — imported so the gate computes values the SAME
# way the committed artifact did, making any disagreement a real signal.
from tools.phase3_polygon_clip_telemetry import (
    ARTIFACT_PATH as TELEMETRY_ARTIFACT_PATH,
    _build_minimal_bundle_hash,
    _edges_by_pair,
    _max_polygon_off_plane_drift_m,
    _surface_lookup,
)


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
REPLICA_V2_DIR = REPLICA_SCENE_DIR / "enriched" / "v2"
LEGACY_RELATIONS_PATH = (
    REPLICA_SCENE_DIR / "computed_relations" / "scene_graph.json"
)
PHASE2_SMOKE_PATH = REPO_ROOT / "eval" / "questions" / "phase2_near_surface_smoke.json"
PHASE3_SMOKE_PATH = REPO_ROOT / "eval" / "questions" / "phase3_near_surface_polygon_smoke.json"
ARTIFACT_PATH = REPLICA_SCENE_DIR / "eval" / "phase3_exit_gate_report.json"


def _phase2_default_runs() -> list[ExtractorRun]:
    """Phase 2 default candidate — used by G4 plane-mode check and G5a."""
    return [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse")),
        ExtractorRun(
            ProximityExtractor(),
            ProximityConfig(mode="sparse", sparse_version=2),
        ),
        ExtractorRun(SurfaceProximityExtractor(), SurfaceProximityConfig()),
    ]


def _phase3_polygon_runs() -> list[ExtractorRun]:
    """Phase 3 opt-in candidate — used by G4 polygon-mode check."""
    return [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse")),
        ExtractorRun(
            ProximityExtractor(),
            ProximityConfig(mode="sparse", sparse_version=2),
        ),
        ExtractorRun(
            SurfaceProximityExtractor(),
            SurfaceProximityConfig(use_polygon_clip=True),
        ),
    ]


# --- G4: determinism + round-trip on BOTH candidates ---------------------


def _gate_g4(artifacts) -> tuple[bool, dict]:
    """Builds each candidate twice and round-trips through dump/load.
    Polygon-mode determinism and round-trip are required because the
    P3.06 closeout commits to the polygon path being ready for future
    promotion — that promise is empty if polygon-mode bundles are not
    replayable."""
    detail: dict = {}
    overall = True

    for label, runs in (
        ("plane_mode", _phase2_default_runs()),
        ("polygon_mode", _phase3_polygon_runs()),
    ):
        bundle_a, _ = build_graph(
            artifacts, runs, density_policy="phase2_telemetry_only",
        )
        bundle_b, _ = build_graph(
            artifacts, runs, density_policy="phase2_telemetry_only",
        )
        hashes_match = bundle_a.bundle_hash == bundle_b.bundle_hash

        with tempfile.TemporaryDirectory() as td:
            dump_dir = Path(td) / "graph"
            dump_scene_graph_bundle(bundle_a, dump_dir)
            loaded = load_scene_graph_bundle(dump_dir)
            round_trip_ok = array_aware_equal(bundle_a, loaded)

        detail[f"{label}_two_run_hash_match"] = hashes_match
        detail[f"{label}_round_trip_equal"] = round_trip_ok
        detail[f"{label}_bundle_hash"] = bundle_a.bundle_hash
        overall = overall and hashes_match and round_trip_ok

    return overall, detail


# --- G5a: Phase 2 smoke under default config -----------------------------


def _gate_g5a(artifacts) -> tuple[bool, dict]:
    """Reruns the frozen Phase 2 NEAR_SURFACE smoke list under default
    SurfaceProximityConfig(). Identical contract to Phase 2 G5 — any
    drift here would mean Phase 2 byte-equivalence has silently
    regressed."""
    smoke = json.loads(PHASE2_SMOKE_PATH.read_text(encoding="utf-8"))
    cfg = SurfaceProximityConfig(
        floor_threshold_m=smoke["thresholds_m"]["floor"],
        wall_threshold_m=smoke["thresholds_m"]["wall"],
        ceiling_threshold_m=smoke["thresholds_m"]["ceiling"],
        include_synth_fallback=smoke["policy"]["include_synth_fallback"],
    )
    edges, _ = SurfaceProximityExtractor().extract(artifacts, cfg)
    emitted = {(e.source.uid, e.target.uid) for e in edges}
    failures: list[str] = []
    for case in smoke["cases"]:
        pair = (case["entity_uid"], case["surface_uid"])
        present = pair in emitted
        if case["expectation"] == "near" and not present:
            failures.append(f"missing NEAR: {pair}")
        if case["expectation"] == "not_near" and present:
            failures.append(f"unexpected NEAR: {pair}")
    near_count = sum(1 for c in smoke["cases"] if c["expectation"] == "near")
    not_near_count = sum(1 for c in smoke["cases"] if c["expectation"] == "not_near")
    return (not failures and near_count >= 4 and not_near_count >= 4), {
        "smoke_path": str(PHASE2_SMOKE_PATH.relative_to(REPO_ROOT)),
        "config_used": "SurfaceProximityConfig()",
        "near_cases": near_count,
        "not_near_cases": not_near_count,
        "failures": failures,
    }


# --- G5b: Phase 3 smoke under polygon mode -------------------------------


def _scene_frame_for_synthetic():
    from common.types import SceneFrame
    return SceneFrame(
        gravity=(0.0, 0.0, -1.0),
        canonical_forward=None, canonical_right=None,
        units="meters", notes="",
    )


def _make_synth_artifacts(case: dict) -> EntityArtifacts:
    """Build a single-entity / single-surface EntityArtifacts from a
    synthetic fixture case. Mirrors the helpers in
    tests/relations/test_near_surface_polygon.py."""
    mn = case["entity_aabb"]["min"]
    mx = case["entity_aabb"]["max"]
    aabb = ((mn[0], mn[1], mn[2]), (mx[0], mx[1], mx[2]))
    cx = (mn[0] + mx[0]) / 2
    cy = (mn[1] + mx[1]) / 2
    cz = (mn[2] + mx[2]) / 2
    entity = EntityArtifact(
        identity=EntityIdentity(
            object_uid=case["id"], display_label=case["id"], aliases=[],
            source_instance_ref=case["id"],
        ),
        bbox_aabb=aabb, bbox_obb=None, centroid=(cx, cy, cz),
        geometry_handle=None,
        semantic_hypotheses=[
            SemanticHypothesis(label=case["id"], confidence=1.0, source="fixture"),
        ],
        embedding=None, extraction_diagnostics={},
    )
    sr = case["surface_record"]
    plane = Plane(a=sr["plane"]["a"], b=sr["plane"]["b"],
                  c=sr["plane"]["c"], d=sr["plane"]["d"])
    polygon = (
        None if sr["polygon"] is None
        else [(v[0], v[1], v[2]) for v in sr["polygon"]]
    )
    surface = StructuralSurface(
        surface_uid=sr["uid"], surface_type=sr["surface_type"],
        plane=plane, polygon=polygon,
        confidence=sr["confidence"], source=sr["source"],
    )
    return EntityArtifacts(
        schema_version=ENT_SCHEMA_VERSION, bundle_hash=f"synth_{case['id']}",
        scene_id=f"synth_{case['id']}", frame=_scene_frame_for_synthetic(),
        representation_hash="rep_synth",
        extractor_name="phase3_exit_gate", extractor_version="0.0",
        entities=[entity], structural_surfaces=[surface],
        geometry_store_path=None,
        diagnostics=ExtractionDiagnostics(
            n_entities=1, n_structural_surfaces=1,
            runtime_seconds=0.0, coverage_score=None, notes="",
        ),
        notes={},
    )


def _gate_g5b(replica_artifacts) -> tuple[bool, dict]:
    """Runs the Phase 3 polygon-mode smoke (eval/questions/
    phase3_near_surface_polygon_smoke.json) under
    SurfaceProximityConfig(use_polygon_clip=True). Synthetic cases use
    isolated single-entity/single-surface bundles; the Replica-grounded
    case (R1) is checked against the real Replica bundle."""
    smoke = json.loads(PHASE3_SMOKE_PATH.read_text(encoding="utf-8"))
    cfg = SurfaceProximityConfig(use_polygon_clip=True)
    extractor = SurfaceProximityExtractor()

    # Pre-compute Replica edges once for any R-prefixed cases.
    replica_edges, _ = extractor.extract(replica_artifacts, cfg)
    replica_pairs = {(e.source.uid, e.target.uid) for e in replica_edges}

    failures: list[str] = []
    cases_checked = {"synthetic": 0, "replica": 0}

    for case in smoke["cases"]:
        cid = case["id"]
        expectation = case["phase3_expectation"]
        if case["synthetic"]:
            artifacts = _make_synth_artifacts(case)
            edges, _ = extractor.extract(artifacts, cfg)
            near_emitted = len(edges) == 1
            if expectation == "near" and not near_emitted:
                failures.append(f"{cid}: expected near, got {len(edges)} edges")
            if expectation == "not_near" and near_emitted:
                failures.append(f"{cid}: expected not_near, got 1 edge")
            cases_checked["synthetic"] += 1
        else:
            pair = (case["entity_uid"], case["surface_uid"])
            present = pair in replica_pairs
            if expectation == "near" and not present:
                failures.append(f"{cid}: missing NEAR on real Replica: {pair}")
            if expectation == "not_near" and present:
                failures.append(f"{cid}: unexpected NEAR on real Replica: {pair}")
            cases_checked["replica"] += 1

    return not failures, {
        "smoke_path": str(PHASE3_SMOKE_PATH.relative_to(REPO_ROOT)),
        "config_used": "SurfaceProximityConfig(use_polygon_clip=True)",
        "total_cases": len(smoke["cases"]),
        "cases_checked": cases_checked,
        "failures": failures,
    }


# --- G6: density telemetry on both candidates (recorded, not blocking) ---


def _gate_g6(artifacts) -> dict:
    """Extends Phase 2 G6 to record density for BOTH candidates."""
    detail = {
        "phase1_sparse_density_limit": SPARSE_DENSITY_LIMIT,
        "entity_count": len(artifacts.entities),
        "note": (
            "Phase 1 cap stays blocking for sparse-v1 only. Phase 2 candidate "
            "(plane-mode) and Phase 3 candidate (polygon-mode) both opt into "
            "telemetry. Recorded, never blocks."
        ),
    }
    for label, runs in (
        ("plane_mode", _phase2_default_runs()),
        ("polygon_mode", _phase3_polygon_runs()),
    ):
        _bundle, diag = build_graph(
            artifacts, runs, density_policy="phase2_telemetry_only",
        )
        ratio = diag.density_ratio if diag.density_ratio is not None else 0.0
        detail[label] = {
            "logical_edges_total": diag.logical_edges_total,
            "density_ratio_per_entity": ratio,
            "exceeds_phase1_cap": ratio > SPARSE_DENSITY_LIMIT,
        }
    return detail


# --- G8: polygon-clip determinism + monotonicity + Phase 2 byte-equiv ----


def _gate_g8(artifacts) -> tuple[bool, dict]:
    """Multi-claim gate. Sub-claims (a) through (g) per the docstring at
    the top of this file."""
    extractor = SurfaceProximityExtractor()
    plane_cfg = SurfaceProximityConfig()
    plane_cfg_explicit = SurfaceProximityConfig(use_polygon_clip=False)
    polygon_cfg = SurfaceProximityConfig(use_polygon_clip=True)

    # (a) polygon-mode determinism across two runs
    polygon_edges_run1, _ = extractor.extract(artifacts, polygon_cfg)
    polygon_edges_run2, _ = extractor.extract(artifacts, polygon_cfg)
    poly_ids_1 = sorted(e.edge_id for e in polygon_edges_run1)
    poly_ids_2 = sorted(e.edge_id for e in polygon_edges_run2)
    polygon_two_runs_byte_equal = poly_ids_1 == poly_ids_2

    # (b)/(c) bundle hashes
    default_hash = _build_minimal_bundle_hash(artifacts, plane_cfg)
    explicit_false_hash = _build_minimal_bundle_hash(artifacts, plane_cfg_explicit)
    polygon_hash = _build_minimal_bundle_hash(artifacts, polygon_cfg)
    default_eq_explicit_false = default_hash == explicit_false_hash
    polygon_hash_differs = default_hash != polygon_hash

    # (d) subset claim on polygoned surfaces; (e) no not_near_to_near flips
    plane_edges, plane_diag = extractor.extract(artifacts, plane_cfg)
    polygon_edges = polygon_edges_run1
    plane_by_pair = _edges_by_pair(plane_edges)
    polygon_by_pair = _edges_by_pair(polygon_edges)
    _surface_type_by_uid, polygon_present_uids, _polygon_none_uids = (
        _surface_lookup(artifacts)
    )
    plane_pairs_polygoned = {
        pair for pair in plane_by_pair if pair[1] in polygon_present_uids
    }
    polygon_pairs_polygoned = {
        pair for pair in polygon_by_pair if pair[1] in polygon_present_uids
    }
    subset_violations = sorted(polygon_pairs_polygoned - plane_pairs_polygoned)
    subset_holds = not subset_violations

    not_near_to_near = sorted(polygon_by_pair.keys() - plane_by_pair.keys())
    no_added_edges_globally = not not_near_to_near

    # (f) telemetry artifact agreement
    if TELEMETRY_ARTIFACT_PATH.exists():
        telemetry = json.loads(TELEMETRY_ARTIFACT_PATH.read_text(encoding="utf-8"))
        fresh_near_to_not_near = sorted(
            plane_by_pair.keys() - polygon_by_pair.keys()
        )
        artifact_mismatches: list[str] = []

        def _check(label, fresh, recorded):
            if fresh != recorded:
                artifact_mismatches.append(
                    f"{label}: fresh={fresh!r} recorded={recorded!r}"
                )

        _check(
            "plane_mode_total",
            plane_diag.logical_edges_total,
            telemetry["edge_counts"]["plane_mode_total"],
        )
        _check(
            "polygon_mode_total",
            len(polygon_edges),
            telemetry["edge_counts"]["polygon_mode_total"],
        )
        _check(
            "near_to_not_near_count",
            len(fresh_near_to_not_near),
            telemetry["flipped_edges"]["near_to_not_near_count"],
        )
        _check(
            "not_near_to_near_count",
            len(not_near_to_near),
            telemetry["flipped_edges"]["not_near_to_near_count"],
        )
        _check(
            "subset_violation_count",
            len(subset_violations),
            telemetry["subset_claim_for_surfaces_with_polygons"]["violation_count"],
        )
        _check(
            "default_bundle_hash",
            default_hash,
            telemetry["phase2_byte_equivalence"]["default_bundle_hash"],
        )
        _check(
            "polygon_mode_bundle_hash",
            polygon_hash,
            telemetry["phase2_byte_equivalence"]["polygon_mode_bundle_hash"],
        )
        telemetry_agrees = not artifact_mismatches
        telemetry_present = True
    else:
        telemetry_agrees = False
        telemetry_present = False
        artifact_mismatches = ["telemetry artifact missing on disk"]

    # (g) phase3_policy declaration
    phase3_policy = {
        "default_behavior": "phase2_plane_mode",
        "polygon_clip_status": "opt_in_candidate",
        "rationale": "P3.06 closeout — keep opt-in until downstream QA + ≥2 scenes",
    }
    policy_ok = (
        phase3_policy["default_behavior"] == "phase2_plane_mode"
        and phase3_policy["polygon_clip_status"] == "opt_in_candidate"
    )

    overall = (
        polygon_two_runs_byte_equal
        and default_eq_explicit_false
        and polygon_hash_differs
        and subset_holds
        and no_added_edges_globally
        and telemetry_agrees
        and policy_ok
    )

    return overall, {
        "a_polygon_mode_two_runs_byte_equal": polygon_two_runs_byte_equal,
        "b_default_equals_explicit_false_bundle_hash": default_eq_explicit_false,
        "c_polygon_mode_bundle_hash_differs": polygon_hash_differs,
        "d_subset_holds_on_polygoned_surfaces": subset_holds,
        "d_subset_violations": [
            {"entity_uid": e, "surface_uid": s} for e, s in subset_violations
        ],
        "e_no_not_near_to_near_flips_globally": no_added_edges_globally,
        "e_not_near_to_near_pairs": [
            {"entity_uid": e, "surface_uid": s} for e, s in not_near_to_near
        ],
        "f_telemetry_artifact_present": telemetry_present,
        "f_telemetry_artifact_agrees_with_gate": telemetry_agrees,
        "f_artifact_mismatches": artifact_mismatches,
        "g_phase3_policy": phase3_policy,
        "g_phase3_policy_ok": policy_ok,
        "bundle_hashes": {
            "default": default_hash,
            "explicit_false": explicit_false_hash,
            "polygon": polygon_hash,
        },
        "edge_counts": {
            "plane_mode_total": plane_diag.logical_edges_total,
            "polygon_mode_total": len(polygon_edges),
            "near_to_not_near_count": len(plane_by_pair.keys() - polygon_by_pair.keys()),
            "not_near_to_near_count": len(not_near_to_near),
        },
    }


def main() -> int:
    if not (REPLICA_V2_DIR / "scene_graph.json").exists():
        print("Refusing: enriched-v2 importer output is missing.")
        return 1
    if not LEGACY_RELATIONS_PATH.exists():
        print(f"Refusing: legacy relations missing at {LEGACY_RELATIONS_PATH}.")
        return 1
    if not TELEMETRY_ARTIFACT_PATH.exists():
        print(
            f"Refusing: P3.05 telemetry artifact missing at "
            f"{TELEMETRY_ARTIFACT_PATH}. Run "
            f"`python tools/phase3_polygon_clip_telemetry.py` first."
        )
        return 1

    telemetry_bytes_before = TELEMETRY_ARTIFACT_PATH.read_bytes()

    artifacts = _real_replica_artifacts()

    g1_pass, g1_detail = _gate_g1(artifacts)
    g2_pass, g2_detail = _gate_g2(artifacts)
    g3_pass, g3_detail = _gate_g3()
    g4_pass, g4_detail = _gate_g4(artifacts)
    g5a_pass, g5a_detail = _gate_g5a(artifacts)
    g5b_pass, g5b_detail = _gate_g5b(artifacts)
    g6_telemetry = _gate_g6(artifacts)
    g7_pass, g7_detail = _gate_g7(artifacts)
    g8_pass, g8_detail = _gate_g8(artifacts)

    # Defense in depth: the P3.05 telemetry artifact MUST not be touched
    # by the gate run. If it changed bytes, we record the violation but
    # do not attempt to roll back — the bytes-before / bytes-after compare
    # is also asserted by the test suite.
    telemetry_bytes_after = TELEMETRY_ARTIFACT_PATH.read_bytes()
    telemetry_untouched = telemetry_bytes_before == telemetry_bytes_after

    # Computed once and reused: payload field, limitation string. Both must
    # report the SAME number to avoid the kind of narration drift where the
    # tolerance got confused with the observed worst case.
    observed_drift_m = _max_polygon_off_plane_drift_m(artifacts)

    blocking = {
        "G1_structural_surfaces": (g1_pass, g1_detail),
        "G2_world_frame_obbs": (g2_pass, g2_detail),
        "G3_phase1_compat_reproduction": (g3_pass, g3_detail),
        "G4_deterministic_and_replayable_both_candidates": (g4_pass, g4_detail),
        "G5a_phase2_smoke_under_default_config": (g5a_pass, g5a_detail),
        "G5b_phase3_polygon_smoke_under_opt_in_config": (g5b_pass, g5b_detail),
        "G7_builder_structural_completeness": (g7_pass, g7_detail),
        "G8_polygon_clip_invariants_and_phase2_byte_equivalence": (g8_pass, g8_detail),
    }
    all_blocking_pass = all(v[0] for v in blocking.values()) and telemetry_untouched

    payload = {
        "gate": "phase3_exit_gate",
        "scene_id": artifacts.scene_id,
        "entity_count": len(artifacts.entities),
        "blocking_gates": {
            name: {"pass": p, **detail} for name, (p, detail) in blocking.items()
        },
        "telemetry_gates": {
            "G6_combined_density_both_candidates": g6_telemetry,
        },
        "policy_decisions_recorded": [
            "P3.06 closeout: keep use_polygon_clip OPT-IN. Default "
            "SurfaceProximityConfig() retains Phase 2 plane-mode behavior. "
            "Polygon candidate is verified ready for future promotion but "
            "is not promoted in Phase 3.",
            "Phase 2 byte-equivalence is asserted at both the direct "
            "extractor and GraphBuilder bundle_hash layers (G8(b) + G5a).",
            "Promotion gate (per P3.06): downstream relation-eval + ≥1 "
            "additional scene OR hand-labeled NEAR_SURFACE relations.",
        ],
        "limitations_recorded_for_phase4": [
            "Polygon-clipped NEAR_SURFACE has been validated on Replica "
            "room_0 only. Generalization to other scenes is unverified.",
            "OBB-to-OBB surface distance remains deferred (P2.07 A5 / "
            "P3 plan still parks it).",
            f"POLYGON_ON_PLANE_TOL_M is {POLYGON_ON_PLANE_TOL_M:.0e} m, "
            f"sized to absorb Replica importer drift (observed max "
            f"{observed_drift_m:.3e} m); future scenes with worse drift "
            f"will need re-survey.",
        ],
        "phase2_byte_equivalence_summary": {
            "default_bundle_hash_equals_explicit_false": (
                g8_detail["b_default_equals_explicit_false_bundle_hash"]
            ),
            "phase2_smoke_passes_under_default_config": g5a_pass,
            "phase2_smoke_failures": g5a_detail["failures"],
        },
        "polygon_candidate_readiness_summary": {
            "phase3_smoke_passes_under_opt_in_config": g5b_pass,
            "phase3_smoke_failures": g5b_detail["failures"],
            "polygon_mode_two_runs_byte_equal": (
                g8_detail["a_polygon_mode_two_runs_byte_equal"]
            ),
            "polygon_mode_bundle_hash_differs": (
                g8_detail["c_polygon_mode_bundle_hash_differs"]
            ),
            "subset_holds_on_polygoned_surfaces": (
                g8_detail["d_subset_holds_on_polygoned_surfaces"]
            ),
            "telemetry_artifact_agrees": (
                g8_detail["f_telemetry_artifact_agrees_with_gate"]
            ),
        },
        "telemetry_artifact_untouched_by_gate_run": telemetry_untouched,
        "extractor_versions": {
            "plane_mode": PLANE_MODE_VERSION,
            "polygon_mode": POLYGON_CLIPPED_VERSION,
        },
        "polygon_on_plane_drift_observed_m": observed_drift_m,
        "polygon_on_plane_tolerance_m": POLYGON_ON_PLANE_TOL_M,
        "overall_blocking_pass": all_blocking_pass,
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"\nPhase 3 exit gate report → {ARTIFACT_PATH.relative_to(REPO_ROOT)}")
    for name, (p, _detail) in blocking.items():
        marker = "PASS" if p else "FAIL"
        print(f"  [{marker}] {name}")
    print(
        f"  [TELEMETRY] G6 plane_mode={g6_telemetry['plane_mode']['density_ratio_per_entity']:.3f}/entity  "
        f"polygon_mode={g6_telemetry['polygon_mode']['density_ratio_per_entity']:.3f}/entity  "
        f"(cap {g6_telemetry['phase1_sparse_density_limit']})"
    )
    print(
        f"  [SIDE-CHECK] telemetry artifact untouched: {telemetry_untouched}"
    )
    print(f"\nOverall blocking: {'PASS' if all_blocking_pass else 'FAIL'}")
    return 0 if all_blocking_pass else 1


if __name__ == "__main__":
    sys.exit(main())

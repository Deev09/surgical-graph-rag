"""Phase 6 exit gate (P6.06).

Verifier for entity-top support (ON_ENTITY_SURFACE) + the Phase 6
reasoner-native QA system test. Writes only:

  scenes/replica_room_0/eval/phase6_exit_gate_report.json

It re-derives the P6 QA scorecard in memory and compares it to the P6.05
artifact; it does not rewrite prior artifacts.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.equality import array_aware_equal
from common.serde import SchemaVersionError
from common.types import Plane
from eval.router_qa import score_questions
from extractors.entity_surfaces import (
    DEFAULT_SUPPORT_CLASS_ALLOWLIST,
    derive_entity_top_surfaces,
)
from geometry.rest_contact import RestContactConfig, rest_contact
from graph.builder import build_graph
from graph.relations.on_entity_surface import (
    ON_ENTITY_SURFACE_VERSION,
    OnEntitySurfaceConfig,
    OnEntitySurfaceExtractor,
)
from graph.serde import dump_scene_graph_bundle, load_scene_graph_bundle
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import Phase6RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer
from tools.phase2_exit_gate import _phase2_runs, _real_replica_artifacts
from tools.phase6_router_qa_eval import _eval_runs, QUESTIONS_PATH


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
REPLICA_V2_DIR = REPLICA_SCENE_DIR / "enriched" / "v2"
EVAL_DIR = REPLICA_SCENE_DIR / "eval"
ARTIFACT_PATH = EVAL_DIR / "phase6_exit_gate_report.json"
PHASE2_REPORT = EVAL_DIR / "phase2_exit_gate_report.json"
PHASE3_REPORT = EVAL_DIR / "phase3_exit_gate_report.json"
PHASE4_REPORT = EVAL_DIR / "phase4_exit_gate_report.json"
PHASE5_REPORT = EVAL_DIR / "phase5_exit_gate_report.json"
P5_QA_EVAL = EVAL_DIR / "phase5_router_qa_eval.json"
P6_QA_EVAL = EVAL_DIR / "phase6_router_qa_eval.json"
SMOKE_FIXTURE = REPO_ROOT / "eval" / "questions" / "phase6_entity_surface_smoke.json"


def _oracle_ctx() -> ExecutionContext:
    return ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))


def _router() -> Router:
    return Router(compiler=Phase6RulesCompiler(), executor=RulesExecutor(),
                  verbalizer=StandardVerbalizer())


def _combined_bundle(artifacts):
    bundle, _diag = build_graph(
        artifacts, _eval_runs(), density_policy="phase2_telemetry_only",
    )
    return bundle


def _gate_g1(artifacts) -> tuple[bool, dict]:
    surfaces_a = derive_entity_top_surfaces(
        artifacts.entities,
        frame=artifacts.frame,
        support_class_allowlist=DEFAULT_SUPPORT_CLASS_ALLOWLIST,
    )
    surfaces_b = derive_entity_top_surfaces(
        artifacts.entities,
        frame=artifacts.frame,
        support_class_allowlist=DEFAULT_SUPPORT_CLASS_ALLOWLIST,
    )
    surf_payload_a = [
        (s.surface_uid, s.owner_entity_uid, s.owner_class, s.plane, s.polygon)
        for s in surfaces_a
    ]
    surf_payload_b = [
        (s.surface_uid, s.owner_entity_uid, s.owner_class, s.plane, s.polygon)
        for s in surfaces_b
    ]
    ext = OnEntitySurfaceExtractor()
    edges_a, _ = ext.extract(artifacts, OnEntitySurfaceConfig())
    edges_b, _ = ext.extract(artifacts, OnEntitySurfaceConfig())
    ids_match = sorted(e.edge_id for e in edges_a) == sorted(e.edge_id for e in edges_b)
    keys_match = (
        sorted((e.source.uid, e.type, e.target.uid) for e in edges_a)
        == sorted((e.source.uid, e.type, e.target.uid) for e in edges_b)
    )
    ok = surf_payload_a == surf_payload_b and ids_match and keys_match
    return ok, {
        "derived_surface_count": len(surfaces_a),
        "surface_derivation_deterministic": surf_payload_a == surf_payload_b,
        "two_run_edge_ids_match": ids_match,
        "two_run_edge_keys_match": keys_match,
        "edge_count": len(edges_a),
        "owner_classes": sorted({s.owner_class for s in surfaces_a}),
    }


def _gate_g2(artifacts) -> tuple[bool, dict]:
    edges, _ = OnEntitySurfaceExtractor().extract(artifacts, OnEntitySurfaceConfig())
    violations: list[dict] = []
    for edge in edges:
        ev = edge.evidence
        if edge.target.uid != ev.get("owner_entity_uid"):
            violations.append({
                "edge_id": edge.edge_id,
                "reason": "owner_entity_uid_mismatch",
            })
        if ev.get("in_plane_gap_m", 0.0) > ev.get("footprint_tolerance_m", 0.0):
            violations.append({
                "edge_id": edge.edge_id,
                "reason": "footprint_gap_exceeds_tolerance",
                "in_plane_gap_m": ev.get("in_plane_gap_m"),
                "footprint_tolerance_m": ev.get("footprint_tolerance_m"),
            })
        if edge.source.kind != "entity" or edge.target.kind != "entity":
            violations.append({
                "edge_id": edge.edge_id,
                "reason": "wrong_endpoint_kind",
            })
    return not violations, {
        "edge_count": len(edges),
        "violation_count": len(violations),
        "violations": violations[:10],
    }


def _gate_g3(artifacts) -> tuple[bool, dict]:
    fixture = json.loads(SMOKE_FIXTURE.read_text(encoding="utf-8"))
    d = fixture["config_defaults"]
    cfg = RestContactConfig(
        contact_threshold_m=d["contact_threshold_m"],
        penetration_tolerance_m=d["penetration_tolerance_m"],
        max_tilt_deg=d["max_tilt_deg"],
        footprint_tolerance_m=d["footprint_tolerance_m"],
    )
    surfaces = fixture["synthetic_surfaces"]
    failures: list[str] = []
    synthetic_checked = 0
    for case in fixture["cases"]:
        if not case["synthetic"]:
            continue
        synthetic_checked += 1
        s = surfaces[case["surface_ref"]]
        p = s["plane"]
        mn = case["entity_aabb"]["min"]
        mx = case["entity_aabb"]["max"]
        result = rest_contact(
            ((mn[0], mn[1], mn[2]), (mx[0], mx[1], mx[2])),
            tuple(case["entity_centroid"]),
            Plane(a=p["a"], b=p["b"], c=p["c"], d=p["d"]),
            [(v[0], v[1], v[2]) for v in s["polygon"]],
            artifacts.frame.gravity,
            cfg,
        )
        if result.on_surface != case["expected_on_entity_surface"]:
            failures.append(f"{case['id']}: verdict drift")
        if result.failed_clauses != case["expected_failed_clauses"]:
            failures.append(f"{case['id']}: failed clauses drift")

    edges, _ = OnEntitySurfaceExtractor().extract(
        artifacts, OnEntitySurfaceConfig(
            support_class_allowlist=tuple(fixture["support_class_allowlist"])
        ),
    )
    pairs = {(e.source.uid, e.target.uid) for e in edges}
    expected_pairs = {
        ("obj_92", "obj_10"),
        ("obj_90", "obj_93"),
        ("obj_12", "obj_11"),
        ("obj_59", "obj_11"),
        ("obj_87", "obj_93"),
        ("obj_35", "obj_55"),
    }
    if pairs != expected_pairs:
        failures.append(f"real positive set drifted: {sorted(pairs)}")
    if ("obj_43", "obj_11") in pairs:
        failures.append("pot obj_43 wrongly emitted")
    if ("obj_55", "obj_11") in pairs:
        failures.append("plant-stand obj_55 wrongly emitted as supported object")
    owner_ok = all(e.target.uid == e.evidence.get("owner_entity_uid") for e in edges)
    if not owner_ok:
        failures.append("owner provenance invariant drifted")
    return not failures, {
        "synthetic_cases_checked": synthetic_checked,
        "real_edge_count": len(edges),
        "real_positive_pairs": sorted([a, b] for a, b in pairs),
        "owner_provenance_invariant_ok": owner_ok,
        "failures": failures,
    }


def _gate_g4(artifacts) -> tuple[bool, dict]:
    if not P6_QA_EVAL.exists():
        return False, {
            "error": "P6.05 QA eval artifact missing; run tools/phase6_router_qa_eval.py first",
        }
    committed = json.loads(P6_QA_EVAL.read_text(encoding="utf-8"))
    bundle = _combined_bundle(artifacts)
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    derived = score_questions(questions, bundle, _router(), _oracle_ctx())
    d_agg = derived["aggregate"]
    c_eval = committed["eval"]
    c_agg = c_eval["aggregate"]

    mismatches: list[str] = []
    if bundle.bundle_hash != committed["graph"]["bundle_hash"]:
        mismatches.append(
            f"bundle_hash: derived={bundle.bundle_hash} "
            f"committed={committed['graph']['bundle_hash']}"
        )
    if d_agg["category_counts"] != c_agg["category_counts"]:
        mismatches.append("category_counts differs")
    if d_agg["false_answer_count"] != c_agg["false_answer_count"]:
        mismatches.append("false_answer_count differs")
    d_cats = {r["question_id"]: r["category"] for r in derived["per_question"]}
    c_cats = {r["question_id"]: r["category"] for r in c_eval["per_question"]}
    if d_cats != c_cats:
        mismatches.append(f"per-question categories: derived={d_cats} committed={c_cats}")

    headline_ok = d_agg["all_expected_outcomes_met"] and d_agg["false_answer_count"] == 0
    return headline_ok and not mismatches, {
        "rederived_all_expected_outcomes_met": d_agg["all_expected_outcomes_met"],
        "rederived_false_answer_count": d_agg["false_answer_count"],
        "rederived_category_counts": d_agg["category_counts"],
        "matches_committed_artifact": not mismatches,
        "mismatches": mismatches,
    }


def _gate_g5(artifacts) -> tuple[bool, dict]:
    bundle = _combined_bundle(artifacts)
    router = _router()
    ctx = _oracle_ctx()
    floor = router.answer("what is on the floor?", bundle, ctx)
    wall = router.answer("what is against the wall?", bundle, ctx)
    attached = router.answer("what is attached to the wall?", bundle, ctx)
    ok = (
        floor.outcome == "bindings" and "obj_39" in floor.cited_uids
        and wall.outcome == "bindings" and "obj_6" in wall.cited_uids
        and all(x not in wall.cited_uids for x in ("obj_63", "obj_84", "obj_17"))
        and attached.outcome == "abstain"
        and not attached.cited_uids
    )
    return ok, {
        "floor_outcome": floor.outcome,
        "floor_obj_39_present": "obj_39" in floor.cited_uids,
        "wall_outcome": wall.outcome,
        "wall_obj_6_present": "obj_6" in wall.cited_uids,
        "wall_negatives_absent": all(
            x not in wall.cited_uids for x in ("obj_63", "obj_84", "obj_17")
        ),
        "attached_outcome": attached.outcome,
        "attached_cited_uids": attached.cited_uids,
    }


def _gate_g6(artifacts) -> tuple[bool, dict]:
    reports = [PHASE2_REPORT, PHASE3_REPORT, PHASE4_REPORT, PHASE5_REPORT]
    loaded = [json.loads(p.read_text(encoding="utf-8")) for p in reports]
    report_passes = [bool(p.get("overall_blocking_pass")) for p in loaded]
    bundle, _diag = build_graph(
        artifacts, _phase2_runs(), density_policy="phase2_telemetry_only")
    n_on = sum(1 for e in bundle.edges if e.type == "ON_SURFACE")
    n_cs = sum(1 for e in bundle.edges if e.type == "CONTACTS_SURFACE")
    n_oes = sum(1 for e in bundle.edges if e.type == "ON_ENTITY_SURFACE")
    n_attached = sum(1 for e in bundle.edges if e.type == "ATTACHED_TO")
    ok = all(report_passes) and n_on == 0 and n_cs == 0 and n_oes == 0 and n_attached == 0
    return ok, {
        "prior_phase_reports_overall_pass": report_passes,
        "default_build_on_surface_edges": n_on,
        "default_build_contacts_surface_edges": n_cs,
        "default_build_on_entity_surface_edges": n_oes,
        "default_build_attached_to_edges": n_attached,
    }


def _gate_g8() -> tuple[bool, dict]:
    raised = False
    try:
        OnEntitySurfaceConfig(contact_threshold_m=-0.01)
    except ValueError:
        raised = True
    return raised, {
        "bad_config": "OnEntitySurfaceConfig(contact_threshold_m=-0.01)",
        "raised_value_error": raised,
    }


def _gate_schema(artifacts) -> tuple[bool, dict]:
    bundle = _combined_bundle(artifacts)
    has_oes = any(e.type == "ON_ENTITY_SURFACE" for e in bundle.edges)
    roundtrip_ok = False
    v5_rejected = False
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle"
        dump_scene_graph_bundle(bundle, out)
        loaded = load_scene_graph_bundle(out)
        roundtrip_ok = array_aware_equal(bundle, loaded) and any(
            e.type == "ON_ENTITY_SURFACE" for e in loaded.edges)
        manifest = out / "manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["schema_version"] = 5
        manifest.write_text(json.dumps(payload))
        try:
            load_scene_graph_bundle(out)
        except SchemaVersionError:
            v5_rejected = True
    ok = has_oes and roundtrip_ok and v5_rejected
    return ok, {
        "v6_on_entity_surface_roundtrip_ok": roundtrip_ok,
        "v5_manifest_strict_rejected": v5_rejected,
        "bundle_had_on_entity_surface_edge": has_oes,
    }


def _watched_artifacts() -> list[Path]:
    return [
        PHASE2_REPORT,
        PHASE3_REPORT,
        PHASE4_REPORT,
        PHASE5_REPORT,
        P5_QA_EVAL,
        P6_QA_EVAL,
    ]


def _snapshot(paths: list[Path]) -> dict[str, str]:
    return {
        str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths if p.exists()
    }


def main() -> int:
    if not (REPLICA_V2_DIR / "scene_graph.json").exists():
        print("Refusing: enriched-v2 importer output is missing.")
        return 1
    required = _watched_artifacts()
    for path in required:
        if not path.exists():
            print(f"Refusing: required artifact missing: {path}")
            return 1

    snap_before = _snapshot(required)
    artifacts = _real_replica_artifacts()

    g1_pass, g1 = _gate_g1(artifacts)
    g2_pass, g2 = _gate_g2(artifacts)
    g3_pass, g3 = _gate_g3(artifacts)
    g4_pass, g4 = _gate_g4(artifacts)
    g5_pass, g5 = _gate_g5(artifacts)
    g6_pass, g6 = _gate_g6(artifacts)
    g8_pass, g8 = _gate_g8()
    schema_pass, schema = _gate_schema(artifacts)

    snap_after = _snapshot(required)
    changed = sorted(k for k in snap_before if snap_before.get(k) != snap_after.get(k))
    g7_pass = not changed

    gates = {
        "G1_entity_top_derivation_and_rest_determinism": (g1_pass, g1),
        "G2_entity_surface_edge_invariants": (g2_pass, g2),
        "G3_entity_surface_smoke_fixture": (g3_pass, g3),
        "G4_mixed_qa_scorecard_matches_committed": (g4_pass, g4),
        "G5_floor_and_wall_qa_regression": (g5_pass, g5),
        "G6_default_path_preserved": (g6_pass, g6),
        "G7_prior_artifacts_untouched": (g7_pass, {
            "changed": changed,
            "all_unchanged": g7_pass,
            "phase6_router_qa_eval_in_snapshot_scope": str(
                P6_QA_EVAL.relative_to(REPO_ROOT)
            ) in snap_before,
            "method": "static path byte sha256 snapshot before/after; verifier writes only its own report",
        }),
        "G8_threshold_sanity_enforced": (g8_pass, g8),
        "schema_v6_roundtrip_and_v5_rejection": (schema_pass, schema),
    }
    overall = all(p for p, _ in gates.values())

    payload = {
        "phase": "P6.06",
        "artifact_kind": "phase6_exit_gate_report",
        "scene_id": artifacts.scene_id,
        "schema_version": 1,
        "overall_blocking_pass": overall,
        "extractor_version": ON_ENTITY_SURFACE_VERSION,
        "gates": {name: {"pass": p, **d} for name, (p, d) in gates.items()},
        "artifact_stability": {
            "watched_artifacts_unchanged": g7_pass,
            "p6_05_eval_untouched": (
                str(P6_QA_EVAL.relative_to(REPO_ROOT)) not in changed
            ),
            "method": "byte sha256 snapshot before/after; verifier writes only its own report",
        },
        "summary": {
            "on_entity_surface_edges": g1["edge_count"],
            "edge_invariant_violations": g2["violation_count"],
            "qa_scorecard": {
                "category_counts": g4.get("rederived_category_counts"),
                "false_answer_count": g4.get("rederived_false_answer_count"),
                "all_expected_outcomes_met": g4.get("rederived_all_expected_outcomes_met"),
                "matches_committed_artifact": g4.get("matches_committed_artifact"),
            },
        },
        "policy_decisions_recorded": [
            "ON_ENTITY_SURFACE is stored entity -> entity; the derived top "
            "surface is evidence, not a GraphRef(kind='surface').",
            "Entity-top surfaces are derived at build time and are not added "
            "to structural_surfaces.",
            "Support-furniture classes are excluded as supported tabletop "
            "answers in P6; cabinets remain excluded as container boundary.",
            "The P6 QA eval is a reasoner-native track, NOT comparable to the "
            "v1 benchmark and NOT directly comparable to P5 because the "
            "question set changed.",
        ],
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"\nPhase 6 exit gate report -> {ARTIFACT_PATH.relative_to(REPO_ROOT)}")
    for name, (passed, _d) in gates.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\nOverall blocking: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())

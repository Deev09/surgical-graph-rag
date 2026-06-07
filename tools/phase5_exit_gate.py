"""Phase 5 exit gate (P5.05).

Closeout verifier for wall contact (CONTACTS_SURFACE) + the first end-to-end
reasoner QA system test. Writes ONLY its own report:

  scenes/replica_room_0/eval/phase5_exit_gate_report.json

VERIFIER, not generator. It does NOT invoke any prior phase's gate/telemetry
main() (those have write side-effects), and it does NOT rewrite the P5.04 QA
eval artifact -- it re-derives the scorecard in memory and COMPARES to the
committed artifact (proving both that the code still produces 6/6 and that the
committed headline artifact is fresh, not stale).

Blocking gates (all on real Replica room_0):
  G1   wall-contact determinism: two ContactsSurfaceExtractor runs -> identical
       edge_id set AND edge keys.
  G2   subset: CONTACTS_SURFACE pairs subseteq polygon-mode NEAR_SURFACE pairs
       on wall surfaces (0 violations).
  G3   wall-contact smoke fixture: synthetic WS cases via the extractor
       (incl. WS5 non-wall skip) + real W1 positive + WN1/WN2/WN3 negatives.
  G4   mixed-QA scorecard re-derived in-memory using the SAME graph assembly
       as P5.04 (same bundle_hash), then compared to the committed
       phase5_router_qa_eval.json: all_expected_outcomes_met, false_answer_count
       == 0, category_counts match, per-question categories match. Fails clearly
       if the committed artifact is missing.
  G5   floor-QA regression: "what is on the floor?" through the combined graph
       still answers with obj_39 (stool).
  G6   default path preserved: committed P2/P3/P4 exit-gate reports pass
       (trusted, not re-derived); an in-memory default Phase 2 build has 0
       ON_SURFACE AND 0 CONTACTS_SURFACE edges.
  G7   prior artifacts untouched: byte snapshot of git-tracked eval JSON
       (except this gate's own report) unchanged before/after. The set includes
       phase5_router_qa_eval.json and the prior phase reports.
  G8   threshold-ordering guard: ContactsSurfaceConfig(contact_threshold_m=0.5)
       raises.
  schema  v4 CONTACTS_SURFACE round-trip + v3 manifest strict rejection (inline,
       temp dir).

Determinism: no timestamp; sorted keys + trailing newline; byte-identical on
rerun. Skip-on-missing: enriched-v2 importer output, committed P2/P3/P4
reports, and the committed P5.04 QA eval artifact must exist.

Run: python tools/phase5_exit_gate.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.equality import array_aware_equal
from common.serde import SchemaVersionError
from common.types import Plane, SceneFrame
from eval.router_qa import score_questions
from extractors.base import (
    EntityArtifact, EntityArtifacts, EntityIdentity, ExtractionDiagnostics,
    SemanticHypothesis, StructuralSurface,
)
from extractors.serde import CURRENT_SCHEMA_VERSION as ENT_SCHEMA_VERSION
from graph.builder import ExtractorRun, build_graph
from graph.relations.contacts_surface import (
    CONTACTS_SURFACE_VERSION, ContactsSurfaceConfig, ContactsSurfaceExtractor,
)
from graph.relations.surface import SurfaceProximityConfig, SurfaceProximityExtractor
from graph.schema import Edge, GraphRef
from graph.serde import dump_scene_graph_bundle, load_scene_graph_bundle
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer
# Reuse building blocks (read-only); never call a prior gate/tool main().
from tools.phase2_exit_gate import _phase2_runs, _real_replica_artifacts
from tools.phase5_router_qa_eval import _eval_runs, QUESTIONS_PATH


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
REPLICA_V2_DIR = REPLICA_SCENE_DIR / "enriched" / "v2"
EVAL_DIR = REPLICA_SCENE_DIR / "eval"
ARTIFACT_PATH = EVAL_DIR / "phase5_exit_gate_report.json"
PHASE2_REPORT = EVAL_DIR / "phase2_exit_gate_report.json"
PHASE3_REPORT = EVAL_DIR / "phase3_exit_gate_report.json"
PHASE4_REPORT = EVAL_DIR / "phase4_exit_gate_report.json"
P5_QA_EVAL = EVAL_DIR / "phase5_router_qa_eval.json"
WALL_FIXTURE = REPO_ROOT / "eval" / "questions" / "phase5_wall_contact_smoke.json"


def _oracle_ctx() -> ExecutionContext:
    return ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))


def _router() -> Router:
    return Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                  verbalizer=StandardVerbalizer())


def _combined_bundle(artifacts):
    bundle, _diag = build_graph(
        artifacts, _eval_runs(), density_policy="phase2_telemetry_only",
    )
    return bundle


# --- G1 determinism ------------------------------------------------------


def _gate_g1(artifacts) -> tuple[bool, dict]:
    ext = ContactsSurfaceExtractor()
    e1, _ = ext.extract(artifacts, ContactsSurfaceConfig())
    e2, _ = ext.extract(artifacts, ContactsSurfaceConfig())
    ids1 = sorted(e.edge_id for e in e1)
    ids2 = sorted(e.edge_id for e in e2)
    keys1 = sorted((e.source.uid, e.type, e.target.uid) for e in e1)
    keys2 = sorted((e.source.uid, e.type, e.target.uid) for e in e2)
    ok = ids1 == ids2 and keys1 == keys2
    return ok, {"two_run_edge_ids_match": ids1 == ids2,
                "two_run_edge_keys_match": keys1 == keys2, "edge_count": len(e1)}


# --- G2 subset -----------------------------------------------------------


def _gate_g2(artifacts) -> tuple[bool, dict]:
    cs, _ = ContactsSurfaceExtractor().extract(artifacts, ContactsSurfaceConfig())
    near, _ = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(use_polygon_clip=True))
    cs_pairs = {(e.source.uid, e.target.uid) for e in cs}
    near_pairs = {(e.source.uid, e.target.uid) for e in near}
    violations = sorted(cs_pairs - near_pairs)
    return not violations, {
        "contacts_surface_pairs": len(cs_pairs),
        "near_surface_polygon_pairs": len(near_pairs),
        "violation_count": len(violations),
        "violations": [{"entity_uid": e, "surface_uid": s} for e, s in violations[:10]],
    }


# --- G3 wall-contact smoke fixture ---------------------------------------


def _synth_artifacts(case, surface) -> EntityArtifacts:
    mn = case["entity_aabb"]["min"]
    mx = case["entity_aabb"]["max"]
    entity = EntityArtifact(
        identity=EntityIdentity(object_uid=case["id"], display_label=case["id"],
                                aliases=[], source_instance_ref=case["id"]),
        bbox_aabb=((mn[0], mn[1], mn[2]), (mx[0], mx[1], mx[2])),
        bbox_obb=None, centroid=tuple(case["entity_centroid"]),
        geometry_handle=None,
        semantic_hypotheses=[SemanticHypothesis(label=case["id"], confidence=1.0, source="fx")],
        embedding=None, extraction_diagnostics={})
    p = surface["plane"]
    sr = StructuralSurface(
        surface_uid=surface["surface_uid"], surface_type=surface["surface_type"],
        plane=Plane(a=p["a"], b=p["b"], c=p["c"], d=p["d"]),
        polygon=[(v[0], v[1], v[2]) for v in surface["polygon"]],
        confidence=1.0, source=surface["source"])
    return EntityArtifacts(
        schema_version=ENT_SCHEMA_VERSION, bundle_hash=f"synth_{case['id']}",
        scene_id=f"synth_{case['id']}",
        frame=SceneFrame(gravity=(0.0, 0.0, -1.0), canonical_forward=None,
                         canonical_right=None, units="meters", notes=""),
        representation_hash="rep", extractor_name="gate", extractor_version="0.0",
        entities=[entity], structural_surfaces=[sr], geometry_store_path=None,
        diagnostics=ExtractionDiagnostics(n_entities=1, n_structural_surfaces=1,
                                          runtime_seconds=0.0, coverage_score=None, notes=""),
        notes={})


def _gate_g3(artifacts) -> tuple[bool, dict]:
    fixture = json.loads(WALL_FIXTURE.read_text(encoding="utf-8"))
    surfaces = fixture["synthetic_surfaces"]
    ext = ContactsSurfaceExtractor()
    cfg = ContactsSurfaceConfig()
    failures: list[str] = []
    synthetic_checked = 0
    for case in fixture["cases"]:
        if not case["synthetic"]:
            continue
        synthetic_checked += 1
        surface = surfaces[case["surface_ref"]]
        arts = _synth_artifacts(case, surface)
        edges, _ = ext.extract(arts, cfg)
        emitted = len(edges) == 1
        if surface["surface_type"] != "wall":
            if emitted:
                failures.append(f"{case['id']}: non-wall surface must yield 0 edges")
        elif emitted != bool(case["expected_contacts_surface"]):
            failures.append(f"{case['id']}: emitted={emitted} exp={case['expected_contacts_surface']}")

    real_edges, _ = ext.extract(artifacts, cfg)
    pairs = {(e.source.uid, e.target.uid) for e in real_edges}
    w1 = next(c for c in fixture["cases"] if c["id"] == "W1")
    w1_ok = (w1["entity_uid"], w1["surface_uid"]) in pairs
    if not w1_ok:
        failures.append(f"W1 {w1['entity_uid']}/{w1['surface_uid']} not emitted")
    wn_ok = True
    for nid in ("WN1", "WN2", "WN3"):
        c = next(x for x in fixture["cases"] if x["id"] == nid)
        if (c["entity_uid"], c["surface_uid"]) in pairs:
            failures.append(f"{nid} {c['entity_uid']} wrongly emitted as wall contact")
            wn_ok = False
    return not failures, {
        "synthetic_cases_checked": synthetic_checked,
        "real_w1_present": w1_ok,
        "real_wn_negatives_excluded": wn_ok,
        "failures": failures,
    }


# --- G4 mixed-QA scorecard re-derived + compared to committed ------------


def _gate_g4(artifacts) -> tuple[bool, dict]:
    if not P5_QA_EVAL.exists():
        return False, {
            "error": "committed P5.04 QA eval artifact missing; run "
                     "tools/phase5_router_qa_eval.py first",
        }
    committed = json.loads(P5_QA_EVAL.read_text(encoding="utf-8"))
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
            f"committed={committed['graph']['bundle_hash']}")
    if d_agg["all_expected_outcomes_met"] != c_agg["all_expected_outcomes_met"]:
        mismatches.append("all_expected_outcomes_met differs")
    if d_agg["false_answer_count"] != c_agg["false_answer_count"]:
        mismatches.append("false_answer_count differs")
    if d_agg["category_counts"] != c_agg["category_counts"]:
        mismatches.append(
            f"category_counts: derived={d_agg['category_counts']} "
            f"committed={c_agg['category_counts']}")
    d_cats = {r["question_id"]: r["category"] for r in derived["per_question"]}
    c_cats = {r["question_id"]: r["category"] for r in c_eval["per_question"]}
    if d_cats != c_cats:
        mismatches.append(f"per-question categories: derived={d_cats} committed={c_cats}")

    headline_ok = (
        d_agg["all_expected_outcomes_met"]
        and d_agg["false_answer_count"] == 0
    )
    ok = headline_ok and not mismatches
    return ok, {
        "rederived_all_expected_outcomes_met": d_agg["all_expected_outcomes_met"],
        "rederived_false_answer_count": d_agg["false_answer_count"],
        "rederived_category_counts": d_agg["category_counts"],
        "matches_committed_artifact": not mismatches,
        "mismatches": mismatches,
    }


# --- G5 floor-QA regression ----------------------------------------------


def _gate_g5(artifacts) -> tuple[bool, dict]:
    bundle = _combined_bundle(artifacts)
    ans = _router().answer("what is on the floor?", bundle, _oracle_ctx())
    ok = ans.outcome == "bindings" and "obj_39" in ans.cited_uids
    return ok, {
        "outcome": ans.outcome,
        "stool_obj_39_present": "obj_39" in ans.cited_uids,
    }


# --- G6 default path preserved -------------------------------------------


def _gate_g6(artifacts) -> tuple[bool, dict]:
    p2 = json.loads(PHASE2_REPORT.read_text(encoding="utf-8"))
    p3 = json.loads(PHASE3_REPORT.read_text(encoding="utf-8"))
    p4 = json.loads(PHASE4_REPORT.read_text(encoding="utf-8"))
    p2_pass = bool(p2.get("overall_blocking_pass"))
    p3_pass = bool(p3.get("overall_blocking_pass"))
    p4_pass = bool(p4.get("overall_blocking_pass"))
    bundle, _diag = build_graph(
        artifacts, _phase2_runs(), density_policy="phase2_telemetry_only")
    n_on = sum(1 for e in bundle.edges if e.type == "ON_SURFACE")
    n_cs = sum(1 for e in bundle.edges if e.type == "CONTACTS_SURFACE")
    ok = p2_pass and p3_pass and p4_pass and n_on == 0 and n_cs == 0
    return ok, {
        "phase2_report_overall_pass": p2_pass,
        "phase3_report_overall_pass": p3_pass,
        "phase4_report_overall_pass": p4_pass,
        "default_build_on_surface_edges": n_on,
        "default_build_contacts_surface_edges": n_cs,
        "note": "P2/P3/P4 pass trusted from committed reports, not re-derived.",
    }


# --- G8 threshold-ordering guard -----------------------------------------


def _gate_g8() -> tuple[bool, dict]:
    raised = False
    try:
        ContactsSurfaceConfig(contact_threshold_m=0.5)
    except ValueError:
        raised = True
    return raised, {"bad_config": "ContactsSurfaceConfig(contact_threshold_m=0.5)",
                    "raised_value_error": raised}


# --- schema gate ---------------------------------------------------------


def _gate_schema(artifacts) -> tuple[bool, dict]:
    bundle = _combined_bundle(artifacts)
    has_cs = any(e.type == "CONTACTS_SURFACE" for e in bundle.edges)
    roundtrip_ok = False
    v3_rejected = False
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle"
        dump_scene_graph_bundle(bundle, out)
        loaded = load_scene_graph_bundle(out)
        roundtrip_ok = array_aware_equal(bundle, loaded) and any(
            e.type == "CONTACTS_SURFACE" for e in loaded.edges)
        manifest = out / "manifest.json"
        payload = json.loads(manifest.read_text())
        payload["schema_version"] = 3
        manifest.write_text(json.dumps(payload))
        try:
            load_scene_graph_bundle(out)
        except SchemaVersionError:
            v3_rejected = True
    ok = has_cs and roundtrip_ok and v3_rejected
    return ok, {"v4_contacts_surface_roundtrip_ok": roundtrip_ok,
                "v3_manifest_strict_rejected": v3_rejected,
                "bundle_had_contacts_surface_edge": has_cs}


# --- G7 helpers ----------------------------------------------------------


def _tracked_eval_json() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "scenes/replica_room_0/eval/*.json"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    paths = []
    for line in out.stdout.splitlines():
        rel = line.strip()
        if not rel or rel.endswith("phase5_exit_gate_report.json"):
            continue
        paths.append(REPO_ROOT / rel)
    return sorted(paths)


def _snapshot(paths) -> dict:
    return {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.exists()}


def main() -> int:
    if not (REPLICA_V2_DIR / "scene_graph.json").exists():
        print("Refusing: enriched-v2 importer output is missing.")
        return 1
    for rep in (PHASE2_REPORT, PHASE3_REPORT, PHASE4_REPORT):
        if not rep.exists():
            print(f"Refusing: committed report missing: {rep}")
            return 1
    if not P5_QA_EVAL.exists():
        print("Refusing: P5.04 QA eval artifact missing; run "
              "tools/phase5_router_qa_eval.py first.")
        return 1

    tracked = _tracked_eval_json()
    snap_before = _snapshot(tracked)

    artifacts = _real_replica_artifacts()

    g1_pass, g1 = _gate_g1(artifacts)
    g2_pass, g2 = _gate_g2(artifacts)
    g3_pass, g3 = _gate_g3(artifacts)
    g4_pass, g4 = _gate_g4(artifacts)
    g5_pass, g5 = _gate_g5(artifacts)
    g6_pass, g6 = _gate_g6(artifacts)
    g8_pass, g8 = _gate_g8()
    schema_pass, schema = _gate_schema(artifacts)

    snap_after = _snapshot(tracked)
    changed = sorted(k for k in snap_before if snap_before.get(k) != snap_after.get(k))
    g7_pass = not changed

    gates = {
        "G1_wall_contact_determinism": (g1_pass, g1),
        "G2_subset_of_polygon_near_surface": (g2_pass, g2),
        "G3_wall_contact_smoke_fixture": (g3_pass, g3),
        "G4_mixed_qa_scorecard_matches_committed": (g4_pass, g4),
        "G5_floor_qa_regression": (g5_pass, g5),
        "G6_default_path_preserved": (g6_pass, g6),
        # G7 records ONLY the claim (Option A): no dynamic list, no count --
        # both churn when later phases add tracked eval artifacts. The targeted
        # boolean is stable (it depends only on whether THAT specific file is
        # tracked, which stays true regardless of other additions) and
        # preserves the one audit fact that matters: the P5.04 QA eval artifact
        # is in the snapshot scope.
        "G7_prior_artifacts_untouched": (g7_pass, {
            "changed": changed, "all_unchanged": g7_pass,
            "phase5_router_qa_eval_in_snapshot_scope": any(
                p.name == "phase5_router_qa_eval.json" for p in tracked),
        }),
        "G8_threshold_ordering_enforced": (g8_pass, g8),
        "schema_v4_roundtrip_and_v3_rejection": (schema_pass, schema),
    }
    overall = all(p for p, _ in gates.values())

    payload = {
        "phase": "P5.05",
        "artifact_kind": "phase5_exit_gate_report",
        "scene_id": artifacts.scene_id,
        "schema_version": 1,
        "overall_blocking_pass": overall,
        "extractor_version": CONTACTS_SURFACE_VERSION,
        "gates": {name: {"pass": p, **d} for name, (p, d) in gates.items()},
        "artifact_stability": {
            "tracked_eval_json_unchanged": g7_pass,
            "p5_04_eval_untouched": (
                "scenes/replica_room_0/eval/phase5_router_qa_eval.json" not in changed),
            "method": "byte sha256 snapshot before/after; verifier writes only its own report",
        },
        "summary": {
            "contacts_surface_edges": g1["edge_count"],
            "subset_violations": g2["violation_count"],
            "qa_scorecard": {
                "category_counts": g4.get("rederived_category_counts"),
                "false_answer_count": g4.get("rederived_false_answer_count"),
                "all_expected_outcomes_met": g4.get("rederived_all_expected_outcomes_met"),
                "matches_committed_artifact": g4.get("matches_committed_artifact"),
            },
        },
        "policy_decisions_recorded": [
            "Verifier only: no prior-phase gate/tool main() invoked; the P5.04 "
            "QA eval artifact is re-derived in memory and COMPARED, never rewritten.",
            "Wall contact (CONTACTS_SURFACE) is isolated -- absent from any "
            "default builder run (G6: 0 ON_SURFACE and 0 CONTACTS_SURFACE).",
            "ATTACHED_TO is not emitted; 'attached to the wall?' defers (no "
            "fabricated attachment).",
            "The QA eval is a reasoner-native track, NOT comparable to the v1 "
            "benchmark.",
        ],
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")

    print(f"\nPhase 5 exit gate report -> {ARTIFACT_PATH.relative_to(REPO_ROOT)}")
    for name, (p, _d) in gates.items():
        print(f"  [{'PASS' if p else 'FAIL'}] {name}")
    print(f"\nOverall blocking: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())

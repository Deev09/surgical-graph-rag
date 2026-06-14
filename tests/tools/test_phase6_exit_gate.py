"""P6.06 exit gate tests.

Run: python tests/tools/test_phase6_exit_gate.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph.relations.on_entity_surface import ON_ENTITY_SURFACE_VERSION
from tools.phase6_exit_gate import (
    ARTIFACT_PATH,
    P5_QA_EVAL,
    P6_QA_EVAL,
    PHASE2_REPORT,
    PHASE3_REPORT,
    PHASE4_REPORT,
    PHASE5_REPORT,
    REPLICA_V2_DIR,
    main,
)


def _load() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def _inputs_present() -> bool:
    return (
        (REPLICA_V2_DIR / "scene_graph.json").exists()
        and PHASE2_REPORT.exists()
        and PHASE3_REPORT.exists()
        and PHASE4_REPORT.exists()
        and PHASE5_REPORT.exists()
        and P5_QA_EVAL.exists()
        and P6_QA_EVAL.exists()
    )


def test_report_exists_and_kind() -> None:
    if not ARTIFACT_PATH.exists():
        raise AssertionError(f"report missing at {ARTIFACT_PATH}; run the tool")
    p = _load()
    if p.get("artifact_kind") != "phase6_exit_gate_report" or p.get("phase") != "P6.06":
        raise AssertionError("kind/phase wrong")
    if p.get("extractor_version") != ON_ENTITY_SURFACE_VERSION:
        raise AssertionError("extractor_version drift")


def test_no_timestamp_keys() -> None:
    p = _load()
    forbidden = {"exported_at", "generated_at", "timestamp", "run_time", "time"}
    if forbidden & set(p.keys()):
        raise AssertionError("timestamp key leaked")


def test_overall_and_each_gate_pass() -> None:
    p = _load()
    if not p["overall_blocking_pass"]:
        failed = [n for n, g in p["gates"].items() if not g["pass"]]
        raise AssertionError(f"overall fail; failed: {failed}")
    expected = {
        "G1_entity_top_derivation_and_rest_determinism",
        "G2_entity_surface_edge_invariants",
        "G3_entity_surface_smoke_fixture",
        "G4_mixed_qa_scorecard_matches_committed",
        "G5_floor_and_wall_qa_regression",
        "G6_default_path_preserved",
        "G7_prior_artifacts_untouched",
        "G8_threshold_sanity_enforced",
        "schema_v5_roundtrip_and_v4_rejection",
    }
    if set(p["gates"].keys()) != expected:
        raise AssertionError(f"gate set drift: {set(p['gates'])} != {expected}")
    for name, g in p["gates"].items():
        if not g["pass"]:
            raise AssertionError(f"{name} did not pass: {g}")


def test_g3_smoke_fixture_real_pairs() -> None:
    g3 = _load()["gates"]["G3_entity_surface_smoke_fixture"]
    if g3["failures"]:
        raise AssertionError(f"G3 failures: {g3['failures']}")
    if g3["real_edge_count"] != 6:
        raise AssertionError(f"expected 6 real entity-surface edges; got {g3['real_edge_count']}")
    if ["obj_43", "obj_11"] in g3["real_positive_pairs"]:
        raise AssertionError("pot boundary must remain excluded")
    if ["obj_55", "obj_11"] in g3["real_positive_pairs"]:
        raise AssertionError("plant-stand must not be supported tabletop answer")
    if not g3["owner_provenance_invariant_ok"]:
        raise AssertionError("owner provenance invariant failed")


def test_g4_rederived_qa_matches_committed() -> None:
    g4 = _load()["gates"]["G4_mixed_qa_scorecard_matches_committed"]
    if not g4["rederived_all_expected_outcomes_met"]:
        raise AssertionError("G4 re-derived all_met must be True")
    if g4["rederived_false_answer_count"] != 0:
        raise AssertionError("G4 re-derived false_answer_count must be 0")
    if not g4["matches_committed_artifact"] or g4["mismatches"]:
        raise AssertionError(f"G4 must match committed artifact; {g4['mismatches']}")


def test_g5_and_g6_regression_isolation() -> None:
    p = _load()
    g5 = p["gates"]["G5_floor_and_wall_qa_regression"]
    if not g5["floor_obj_39_present"]:
        raise AssertionError("floor regression must still include obj_39")
    if not g5["wall_obj_6_present"] or not g5["wall_negatives_absent"]:
        raise AssertionError("wall regression drifted")
    if g5["attached_outcome"] != "abstain" or g5["attached_cited_uids"]:
        raise AssertionError("attachment must remain deferred with no citations")

    g6 = p["gates"]["G6_default_path_preserved"]
    if g6["default_build_on_surface_edges"] != 0:
        raise AssertionError("default build must have 0 ON_SURFACE")
    if g6["default_build_contacts_surface_edges"] != 0:
        raise AssertionError("default build must have 0 CONTACTS_SURFACE")
    if g6["default_build_on_entity_surface_edges"] != 0:
        raise AssertionError("default build must have 0 ON_ENTITY_SURFACE")


def test_schema_gate_passes() -> None:
    s = _load()["gates"]["schema_v5_roundtrip_and_v4_rejection"]
    if not s["v5_on_entity_surface_roundtrip_ok"]:
        raise AssertionError("v5 round-trip failed")
    if not s["v4_manifest_strict_rejected"]:
        raise AssertionError("v4 not rejected")


def test_artifact_stability_claims() -> None:
    p = _load()
    g7 = p["gates"]["G7_prior_artifacts_untouched"]
    if not g7["all_unchanged"] or g7["changed"]:
        raise AssertionError(f"G7 changed: {g7['changed']}")
    if not g7["phase6_router_qa_eval_in_snapshot_scope"]:
        raise AssertionError("P6 QA eval must be in snapshot scope")
    if not p["artifact_stability"]["p6_05_eval_untouched"]:
        raise AssertionError("p6_05_eval_untouched must be True")


def test_rerun_byte_identical_and_verifier_only() -> None:
    if not _inputs_present():
        print("  SKIP (inputs missing)")
        return
    watched = [P6_QA_EVAL, P5_QA_EVAL, PHASE2_REPORT, PHASE3_REPORT, PHASE4_REPORT, PHASE5_REPORT]
    report_before = ARTIFACT_PATH.read_bytes()
    watched_before = {p: p.read_bytes() for p in watched}
    if main() != 0:
        raise AssertionError("gate exited non-zero")
    if ARTIFACT_PATH.read_bytes() != report_before:
        raise AssertionError("report drifted on rerun")
    for p, data in watched_before.items():
        if p.read_bytes() != data:
            raise AssertionError(f"gate rewrote watched artifact {p}")


TESTS = [
    test_report_exists_and_kind,
    test_no_timestamp_keys,
    test_overall_and_each_gate_pass,
    test_g3_smoke_fixture_real_pairs,
    test_g4_rederived_qa_matches_committed,
    test_g5_and_g6_regression_isolation,
    test_schema_gate_passes,
    test_artifact_stability_claims,
    test_rerun_byte_identical_and_verifier_only,
]


def main_cli() -> int:
    failed = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
            print()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main_cli())

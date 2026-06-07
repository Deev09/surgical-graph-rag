"""P5.05 exit gate tests.

Reads the committed Phase 5 exit-gate report and asserts all gates pass +
structural invariants. Reruns the gate to confirm byte-stability and that the
gate does NOT rewrite the P5.04 QA eval artifact or prior-phase reports
(verifier, not generator).

Run: python tests/tools/test_phase5_exit_gate.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph.relations.contacts_surface import CONTACTS_SURFACE_VERSION
from tools.phase5_exit_gate import (
    ARTIFACT_PATH, P5_QA_EVAL, PHASE2_REPORT, PHASE3_REPORT, PHASE4_REPORT,
    REPLICA_V2_DIR, main,
)


def _load() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def _inputs_present() -> bool:
    return (
        (REPLICA_V2_DIR / "scene_graph.json").exists()
        and PHASE2_REPORT.exists() and PHASE3_REPORT.exists()
        and PHASE4_REPORT.exists() and P5_QA_EVAL.exists()
    )


def test_report_exists_and_kind() -> None:
    if not ARTIFACT_PATH.exists():
        raise AssertionError(f"report missing at {ARTIFACT_PATH}; run the tool")
    p = _load()
    if p.get("artifact_kind") != "phase5_exit_gate_report" or p.get("phase") != "P5.05":
        raise AssertionError("kind/phase wrong")
    if p.get("extractor_version") != CONTACTS_SURFACE_VERSION:
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
        "G1_wall_contact_determinism",
        "G2_subset_of_polygon_near_surface",
        "G3_wall_contact_smoke_fixture",
        "G4_mixed_qa_scorecard_matches_committed",
        "G5_floor_qa_regression",
        "G6_default_path_preserved",
        "G7_prior_artifacts_untouched",
        "G8_threshold_ordering_enforced",
        "schema_v4_roundtrip_and_v3_rejection",
    }
    if set(p["gates"].keys()) != expected:
        raise AssertionError(f"gate set drift: {set(p['gates'])} != {expected}")
    for name, g in p["gates"].items():
        if not g["pass"]:
            raise AssertionError(f"{name} did not pass: {g}")


def test_g2_zero_subset_violations() -> None:
    g2 = _load()["gates"]["G2_subset_of_polygon_near_surface"]
    if g2["violation_count"] != 0 or g2["violations"]:
        raise AssertionError(f"G2 violations: {g2}")


def test_g3_fixture_includes_w1_and_wn_negatives() -> None:
    g3 = _load()["gates"]["G3_wall_contact_smoke_fixture"]
    if g3["failures"]:
        raise AssertionError(f"G3 failures: {g3['failures']}")
    if not g3["real_w1_present"]:
        raise AssertionError("G3 must confirm W1 (lamp) present")
    if not g3["real_wn_negatives_excluded"]:
        raise AssertionError("G3 must confirm WN negatives excluded")
    if g3["synthetic_cases_checked"] < 6:
        raise AssertionError("G3 must check >=6 synthetic cases")


def test_g4_rederived_qa_matches_committed() -> None:
    g4 = _load()["gates"]["G4_mixed_qa_scorecard_matches_committed"]
    if not g4["rederived_all_expected_outcomes_met"]:
        raise AssertionError("G4 re-derived all_met must be True")
    if g4["rederived_false_answer_count"] != 0:
        raise AssertionError("G4 re-derived false_answer_count must be 0")
    if not g4["matches_committed_artifact"] or g4["mismatches"]:
        raise AssertionError(f"G4 must match committed artifact; {g4['mismatches']}")


def test_g6_default_path_preserved_isolation() -> None:
    g6 = _load()["gates"]["G6_default_path_preserved"]
    if not (g6["phase2_report_overall_pass"] and g6["phase3_report_overall_pass"]
            and g6["phase4_report_overall_pass"]):
        raise AssertionError("G6: P2/P3/P4 reports must pass")
    if g6["default_build_on_surface_edges"] != 0:
        raise AssertionError("G6: default build must have 0 ON_SURFACE")
    if g6["default_build_contacts_surface_edges"] != 0:
        raise AssertionError("G6: default build must have 0 CONTACTS_SURFACE")


def test_g7_tracked_includes_p5_04_and_unchanged() -> None:
    p = _load()
    g7 = p["gates"]["G7_prior_artifacts_untouched"]
    # G7 records the claim + a stable boolean, NOT the dynamic file list, so
    # the report stays byte-stable as later phases add tracked eval artifacts.
    if not g7["all_unchanged"] or g7["changed"]:
        raise AssertionError(f"G7 changed: {g7['changed']}")
    if "tracked_eval_json_checked" in g7:
        raise AssertionError("G7 must not persist the dynamic tracked-file list")
    if not g7.get("phase5_router_qa_eval_in_snapshot_scope"):
        raise AssertionError(
            "G7 must confirm phase5_router_qa_eval.json is in the snapshot scope"
        )
    if not p["artifact_stability"]["p5_04_eval_untouched"]:
        raise AssertionError("p5_04_eval_untouched must be True")


def test_schema_gate_passes() -> None:
    s = _load()["gates"]["schema_v4_roundtrip_and_v3_rejection"]
    if not s["v4_contacts_surface_roundtrip_ok"]:
        raise AssertionError("v4 round-trip failed")
    if not s["v3_manifest_strict_rejected"]:
        raise AssertionError("v3 not rejected")


def test_rerun_byte_identical() -> None:
    if not _inputs_present():
        print("  SKIP (inputs missing)")
        return
    before = ARTIFACT_PATH.read_bytes()
    if main() != 0:
        raise AssertionError("gate exited non-zero")
    if ARTIFACT_PATH.read_bytes() != before:
        raise AssertionError("report drifted on rerun")


def test_rerun_does_not_rewrite_p5_04_or_prior_reports() -> None:
    if not _inputs_present():
        print("  SKIP (inputs missing)")
        return
    watched = [P5_QA_EVAL, PHASE2_REPORT, PHASE3_REPORT, PHASE4_REPORT]
    before = {p: p.read_bytes() for p in watched}
    if main() != 0:
        raise AssertionError("gate exited non-zero")
    for p, b in before.items():
        if p.read_bytes() != b:
            raise AssertionError(f"gate rewrote {p}")


TESTS = [
    test_report_exists_and_kind,
    test_no_timestamp_keys,
    test_overall_and_each_gate_pass,
    test_g2_zero_subset_violations,
    test_g3_fixture_includes_w1_and_wn_negatives,
    test_g4_rederived_qa_matches_committed,
    test_g6_default_path_preserved_isolation,
    test_g7_tracked_includes_p5_04_and_unchanged,
    test_schema_gate_passes,
    test_rerun_byte_identical,
    test_rerun_does_not_rewrite_p5_04_or_prior_reports,
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

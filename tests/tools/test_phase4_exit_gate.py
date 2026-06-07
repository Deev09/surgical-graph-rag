"""P4.06 exit gate tests.

Reads the committed Phase 4 exit-gate report and asserts all blocking gates
pass + structural invariants. Reruns the gate to confirm:
  - the report is byte-stable;
  - the gate does NOT rewrite the P4.05 telemetry or any prior-phase
    artifact (verifier, not generator).

Run: python tests/tools/test_phase4_exit_gate.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph.relations.on_surface import ON_SURFACE_VERSION
from tools.phase4_exit_gate import (
    ARTIFACT_PATH, PHASE2_REPORT, PHASE3_REPORT, REPLICA_V2_DIR, main,
)

EVAL_DIR = REPO_ROOT / "scenes" / "replica_room_0" / "eval"
P4_TELEMETRY = EVAL_DIR / "phase4_on_surface_telemetry.json"


def _load() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def _inputs_present() -> bool:
    return (
        (REPLICA_V2_DIR / "scene_graph.json").exists()
        and PHASE2_REPORT.exists() and PHASE3_REPORT.exists()
    )


# --- shape ---------------------------------------------------------------


def test_report_exists_and_kind() -> None:
    if not ARTIFACT_PATH.exists():
        raise AssertionError(f"report missing at {ARTIFACT_PATH}; run the tool")
    p = _load()
    if p.get("artifact_kind") != "phase4_exit_gate_report":
        raise AssertionError(f"artifact_kind wrong: {p.get('artifact_kind')!r}")
    if p.get("phase") != "P4.06":
        raise AssertionError(f"phase wrong: {p.get('phase')!r}")
    if p.get("extractor_version") != ON_SURFACE_VERSION:
        raise AssertionError("extractor_version drift")


def test_no_timestamp_keys() -> None:
    p = _load()
    forbidden = {"exported_at", "generated_at", "timestamp", "run_time", "time"}
    leaked = forbidden & set(p.keys())
    if leaked:
        raise AssertionError(f"timestamp keys leaked: {leaked!r}")


# --- gates ---------------------------------------------------------------


def test_overall_and_each_gate_pass() -> None:
    p = _load()
    if not p["overall_blocking_pass"]:
        failed = [n for n, g in p["gates"].items() if not g["pass"]]
        raise AssertionError(f"overall fail; failed gates: {failed}")
    expected = {
        "G1_rest_contact_determinism",
        "G2_subset_of_polygon_near_surface",
        "G3_clean_inverse_supports",
        "G4_no_materialized_supports",
        "G5_phase4_smoke_fixture",
        "G6_default_path_preserved",
        "G7_prior_artifacts_untouched",
        "G8_threshold_ordering_enforced",
        "schema_v3_roundtrip_and_v2_rejection",
    }
    if set(p["gates"].keys()) != expected:
        raise AssertionError(
            f"gate set drift: missing={expected - set(p['gates'])} "
            f"extra={set(p['gates']) - expected}"
        )
    for name, g in p["gates"].items():
        if not g["pass"]:
            raise AssertionError(f"{name} did not pass: {g}")


def test_g2_zero_subset_violations() -> None:
    g2 = _load()["gates"]["G2_subset_of_polygon_near_surface"]
    if g2["violation_count"] != 0 or g2["violations"]:
        raise AssertionError(f"G2 subset violations: {g2}")


def test_g3_g4_clean_inverse_no_materialized() -> None:
    p = _load()
    g3 = p["gates"]["G3_clean_inverse_supports"]
    if g3["support_facts"] != g3["on_surface_edges"]:
        raise AssertionError("G3 clean inverse violated")
    g4 = p["gates"]["G4_no_materialized_supports"]
    if g4["materialized_supports"] != 0:
        raise AssertionError("G4 materialized SUPPORTS != 0")


def test_g5_fixture_and_real_f1() -> None:
    g5 = _load()["gates"]["G5_phase4_smoke_fixture"]
    if g5["failures"]:
        raise AssertionError(f"G5 failures: {g5['failures']}")
    if not g5["real_f1_present"]:
        raise AssertionError("G5 real F1 (stool) not present")
    if g5["synthetic_cases_checked"] < 8:
        raise AssertionError(f"G5 checked < 8 synthetic cases: {g5}")


def test_g6_default_path_preserved_and_isolated() -> None:
    g6 = _load()["gates"]["G6_default_path_preserved"]
    if not g6["phase2_report_overall_pass"] or not g6["phase3_report_overall_pass"]:
        raise AssertionError("G6: committed P2/P3 reports must report pass")
    if g6["default_build_on_surface_edges"] != 0:
        raise AssertionError(
            "G6: default build must contain 0 ON_SURFACE edges (isolation)"
        )


def test_g7_prior_artifacts_untouched() -> None:
    p = _load()
    g7 = p["gates"]["G7_prior_artifacts_untouched"]
    # G7 records the claim (all_unchanged / changed), NOT the dynamic file
    # list -- so the report stays stable as later phases add tracked eval
    # artifacts. The own-report exclusion is enforced inside the gate, not
    # asserted from a stored list.
    if not g7["all_unchanged"] or g7["changed"]:
        raise AssertionError(f"G7: tracked eval artifacts changed: {g7['changed']}")
    for churning in ("tracked_eval_json_checked", "tracked_eval_json_count"):
        if churning in g7:
            raise AssertionError(
                f"G7 must NOT persist {churning} -- it is repo-state-dependent "
                "and churns when later phases add tracked eval artifacts"
            )
    if not p["artifact_stability"]["telemetry_untouched"]:
        raise AssertionError("artifact_stability.telemetry_untouched must be True")


def test_schema_gate_v3_roundtrip_and_v2_rejection() -> None:
    s = _load()["gates"]["schema_v3_roundtrip_and_v2_rejection"]
    if not s["v3_on_surface_roundtrip_ok"]:
        raise AssertionError("schema: v3 ON_SURFACE round-trip failed")
    if not s["v2_manifest_strict_rejected"]:
        raise AssertionError("schema: v2 manifest was not strictly rejected")


# --- determinism + verifier-not-generator -------------------------------


def test_rerun_byte_identical_report() -> None:
    if not _inputs_present():
        print("  SKIP (inputs missing)")
        return
    before = ARTIFACT_PATH.read_bytes()
    rc = main()
    if rc != 0:
        raise AssertionError(f"gate exited non-zero: {rc}")
    after = ARTIFACT_PATH.read_bytes()
    if before != after:
        import difflib
        diff = "\n".join(difflib.unified_diff(
            before.decode().splitlines(), after.decode().splitlines(),
            lineterm="", n=2,
        ))
        raise AssertionError(f"report drifted on rerun:\n{diff[:1500]}")


def test_rerun_does_not_rewrite_prior_artifacts() -> None:
    """Verifier, not generator: rerunning the Phase 4 gate must not change
    the P4.05 telemetry or the Phase 2/3 gate reports."""
    if not _inputs_present():
        print("  SKIP (inputs missing)")
        return
    watched = [P4_TELEMETRY, PHASE2_REPORT, PHASE3_REPORT]
    before = {p: p.read_bytes() for p in watched if p.exists()}
    rc = main()
    if rc != 0:
        raise AssertionError(f"gate exited non-zero: {rc}")
    for p, b in before.items():
        if p.read_bytes() != b:
            raise AssertionError(f"gate rewrote prior artifact: {p}")


TESTS = [
    test_report_exists_and_kind,
    test_no_timestamp_keys,
    test_overall_and_each_gate_pass,
    test_g2_zero_subset_violations,
    test_g3_g4_clean_inverse_no_materialized,
    test_g5_fixture_and_real_f1,
    test_g6_default_path_preserved_and_isolated,
    test_g7_prior_artifacts_untouched,
    test_schema_gate_v3_roundtrip_and_v2_rejection,
    test_rerun_byte_identical_report,
    test_rerun_does_not_rewrite_prior_artifacts,
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

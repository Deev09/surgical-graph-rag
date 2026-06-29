"""P7.03 exit gate tests.

Run: python tests/tools/test_phase7_exit_gate.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph.relations.attached_to import ATTACHED_TO_VERSION
from tools.phase7_exit_gate import ARTIFACT_PATH, APARTMENT0, main


def _load() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def test_report_exists_and_kind() -> None:
    if not ARTIFACT_PATH.exists():
        raise AssertionError(f"report missing at {ARTIFACT_PATH}; run the tool")
    p = _load()
    if p.get("artifact_kind") != "phase7_exit_gate_report" or p.get("phase") != "P7.03":
        raise AssertionError("kind/phase wrong")
    if p.get("extractor_version") != ATTACHED_TO_VERSION:
        raise AssertionError("extractor_version drift")


def test_overall_and_gate_set_pass() -> None:
    p = _load()
    if not p["overall_blocking_pass"]:
        failed = [n for n, g in p["gates"].items() if not g["pass"]]
        raise AssertionError(f"overall fail; failed: {failed}")
    expected = {
        "G1_schema_v6_attached_to_roundtrip_and_v5_rejection",
        "G2_attachment_smoke_fixture",
        "G3_room0_honest_empty",
        "G4_compiler_p7_answer_and_p6_freeze",
        "G5_executor_answers_attached_to",
        "G6_phase6_exit_gate_still_passes",
        "G7_default_path_preserved",
        "G8_attached_to_determinism",
        "G9_apartment0_demo_plausibility",
    }
    if set(p["gates"]) != expected:
        raise AssertionError(f"gate set drift: {set(p['gates'])} != {expected}")


def test_core_phase7_semantics() -> None:
    gates = _load()["gates"]
    g3 = gates["G3_room0_honest_empty"]
    if g3["attached_to_edge_count"] != 0 or g3["answer_outcome"] != "empty":
        raise AssertionError(f"room_0 empty semantics drifted: {g3}")
    g4 = gates["G4_compiler_p7_answer_and_p6_freeze"]
    if g4["p7_edge_type"] != "ATTACHED_TO" or g4["p6_outcome"] != "out_of_schema":
        raise AssertionError(f"compiler freeze drifted: {g4}")
    g7 = gates["G7_default_path_preserved"]["default_build_edge_counts"]
    if g7["ATTACHED_TO"] != 0:
        raise AssertionError("default build must have 0 ATTACHED_TO")


def test_apartment_demo_gate_when_dataset_present() -> None:
    g9 = _load()["gates"]["G9_apartment0_demo_plausibility"]
    if not (APARTMENT0 / "habitat" / "info_semantic.json").exists():
        if not g9["skipped"]:
            raise AssertionError("apartment demo should be marked skipped when dataset absent")
        return
    if g9["skipped"]:
        raise AssertionError("apartment demo unexpectedly skipped")
    if g9["attached_to_entities"] != ["obj_176", "obj_260", "obj_309"]:
        raise AssertionError(f"apartment attached positives drifted: {g9['attached_to_entities']}")
    if not g9["aggregate"]["all_expected_outcomes_met"]:
        raise AssertionError("apartment demo scorecard did not meet plausibility labels")


def test_rerun_byte_identical() -> None:
    before = ARTIFACT_PATH.read_bytes()
    if main() != 0:
        raise AssertionError("gate exited non-zero")
    after = ARTIFACT_PATH.read_bytes()
    if before != after:
        raise AssertionError("phase7 gate report drifted on rerun")


TESTS = [
    test_report_exists_and_kind,
    test_overall_and_gate_set_pass,
    test_core_phase7_semantics,
    test_apartment_demo_gate_when_dataset_present,
    test_rerun_byte_identical,
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

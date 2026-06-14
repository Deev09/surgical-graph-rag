"""P6.05 tests: Router-native QA eval artifact.

Run: python tests/tools/test_phase6_router_qa_eval.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.phase6_router_qa_eval import ARTIFACT_PATH, REPLICA_V2_DIR, main


def _load() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def _replica_v2_present() -> bool:
    return (REPLICA_V2_DIR / "scene_graph.json").exists()


def test_artifact_exists_and_kind() -> None:
    if not ARTIFACT_PATH.exists():
        raise AssertionError(f"artifact missing at {ARTIFACT_PATH}; run the tool")
    p = _load()
    if p.get("artifact_kind") != "router_qa_eval" or p.get("phase") != "P6.05":
        raise AssertionError("artifact kind/phase wrong")


def test_artifact_no_timestamp_keys() -> None:
    p = _load()
    forbidden = {"exported_at", "generated_at", "timestamp", "run_time", "time"}
    if forbidden & set(p.keys()):
        raise AssertionError("timestamp key leaked")
    if not p["determinism"]["timestamp_free"]:
        raise AssertionError("timestamp_free must be True")


def test_artifact_scorecard_all_met_zero_false() -> None:
    agg = _load()["eval"]["aggregate"]
    if agg["total_questions"] != 7:
        raise AssertionError(f"expected 7 questions; got {agg['total_questions']}")
    if not agg["all_expected_outcomes_met"]:
        raise AssertionError("all_expected_outcomes_met must be True")
    if agg["false_answer_count"] != 0:
        raise AssertionError("false_answer_count must be 0")
    if agg["miss_count"] != 0 or agg["unexpected_count"] != 0:
        raise AssertionError("no miss/unexpected allowed")


def test_q3_table_answer_and_q7_empty() -> None:
    rows = {r["question_id"]: r for r in _load()["eval"]["per_question"]}
    q3 = rows["Q3"]
    expected = ["obj_12", "obj_59", "obj_87", "obj_90", "obj_92"]
    if q3["category"] != "true_answer":
        raise AssertionError(f"Q3 category drifted: {q3['category']}")
    if q3["cited_uids"] != expected:
        raise AssertionError(f"Q3 cited_uids drifted: {q3['cited_uids']}")
    for boundary in ("obj_43", "obj_55", "obj_39"):
        if boundary in q3["cited_uids"]:
            raise AssertionError(f"Q3 must not cite boundary object {boundary}")

    q7 = rows["Q7"]
    if q7["category"] != "true_empty" or q7["actual_outcome"] != "empty":
        raise AssertionError(f"Q7 must be true empty; got {q7}")
    if q7["deferred"] or q7["cited_uids"]:
        raise AssertionError("Q7 empty must not defer or cite entities")


def test_q6_attachment_still_defers() -> None:
    q6 = {r["question_id"]: r for r in _load()["eval"]["per_question"]}["Q6"]
    if q6["category"] != "correct_defer" or not q6["deferred"]:
        raise AssertionError(f"Q6 attachment must remain deferred; got {q6}")
    if q6["cited_uids"] or q6["cited_edges"]:
        raise AssertionError("deferred attachment must cite nothing")


def test_artifact_records_definition_change_and_limits() -> None:
    p = _load()
    if not p["phase6_eval_definition_change"]["not_comparable_to_p5"]:
        raise AssertionError("P6 artifact must mark P5 non-comparability")
    limits = " ".join(p["interpretation_limits"]).lower()
    if "not comparable to the v1 benchmark" not in limits:
        raise AssertionError("must disclaim v1 benchmark comparability")
    if "not directly comparable to the p5" not in limits:
        raise AssertionError("must disclaim P5 scorecard comparability")
    if "single scene" not in limits:
        raise AssertionError("must disclose single-scene limitation")


def test_tool_rerun_byte_identical() -> None:
    if not _replica_v2_present():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    before = ARTIFACT_PATH.read_bytes()
    rc = main()
    if rc != 0:
        raise AssertionError(f"tool exited non-zero: {rc}")
    after = ARTIFACT_PATH.read_bytes()
    if before != after:
        raise AssertionError("artifact drifted on rerun")


TESTS = [
    test_artifact_exists_and_kind,
    test_artifact_no_timestamp_keys,
    test_artifact_scorecard_all_met_zero_false,
    test_q3_table_answer_and_q7_empty,
    test_q6_attachment_still_defers,
    test_artifact_records_definition_change_and_limits,
    test_tool_rerun_byte_identical,
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

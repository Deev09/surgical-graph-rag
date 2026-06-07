"""P5.04 tests: Router-native QA eval scorer + artifact.

Unit-tests the pure scorer's classification rules (the pinned P5.04 scoring),
an end-to-end score_questions run on a tiny synthetic bundle through the real
Router, and the committed artifact (6/6 outcomes met, false_answer_count==0,
byte-identical rerun).

Run: python tests/tools/test_phase5_router_qa_eval.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.types import SceneFrame
from eval.router_qa import _classify, score_questions
from graph.schema import Edge, GraphRef, Node, Plane, SceneGraphBundle, SurfaceRecord
from graph.serde import CURRENT_SCHEMA_VERSION
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer
from tools.phase5_router_qa_eval import ARTIFACT_PATH, REPLICA_V2_DIR, main


def _ans(outcome, cited=()):
    return SimpleNamespace(outcome=outcome, cited_uids=list(cited), cited_edges=[])


def _q(expected, must_contain=(), must_not_contain=()):
    return {
        "question_id": "Qx", "question": "q?", "expected_outcome": expected,
        "expected_must_contain": list(must_contain),
        "expected_must_not_contain": list(must_not_contain),
    }


# --- unit: classification rules (the pinned P5.04 scoring) ----------------


def test_true_answer() -> None:
    cat = _classify(_q("answer", must_contain=["a"]), _ans("bindings", ["a", "b"]), False)
    if cat != "true_answer":
        raise AssertionError(cat)


def test_missing_must_contain_is_miss() -> None:
    cat = _classify(_q("answer", must_contain=["z"]), _ans("bindings", ["a", "b"]), False)
    if cat != "miss":
        raise AssertionError(cat)


def test_answer_not_bindings_is_miss() -> None:
    cat = _classify(_q("answer", must_contain=["a"]), _ans("empty", []), False)
    if cat != "miss":
        raise AssertionError(cat)


def test_must_not_contain_present_is_false_answer() -> None:
    cat = _classify(
        _q("answer", must_contain=["a"], must_not_contain=["bad"]),
        _ans("bindings", ["a", "bad"]), False,
    )
    if cat != "false_answer":
        raise AssertionError(cat)


def test_defer_deferred_is_correct_defer() -> None:
    cat = _classify(_q("defer"), _ans("abstain", []), True)
    if cat != "correct_defer":
        raise AssertionError(cat)


def test_defer_with_bindings_is_false_answer() -> None:
    cat = _classify(_q("defer"), _ans("bindings", ["x"]), False)
    if cat != "false_answer":
        raise AssertionError(cat)


def test_defer_not_deferred_is_unexpected() -> None:
    cat = _classify(_q("defer"), _ans("empty", []), False)
    if cat != "unexpected":
        raise AssertionError(cat)


def test_empty_is_true_empty() -> None:
    cat = _classify(_q("empty"), _ans("empty", []), False)
    if cat != "true_empty":
        raise AssertionError(cat)


def test_empty_with_bindings_is_false_answer() -> None:
    cat = _classify(_q("empty"), _ans("bindings", ["x"]), False)
    if cat != "false_answer":
        raise AssertionError(cat)


# --- end-to-end score_questions on a tiny synthetic bundle ---------------


def _wall_scene() -> SceneGraphBundle:
    wall = SurfaceRecord(
        uid="wall_1", surface_type="wall", plane=Plane(a=0.0, b=-1.0, c=0.0, d=2.0),
        polygon=[(0.0, 2.0, 0.0), (2.0, 2.0, 0.0), (2.0, 2.0, 2.0), (0.0, 2.0, 2.0)],
        source="habitat_label", confidence=1.0,
    )
    node = Node(
        id="obj_6", label="lamp", label_confidence=1.0, centroid=(0.0, 0.0, 0.0),
        bbox_aabb=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), bbox_obb=None,
        embedding_ref=None, attributes={"display_label": "lamp"}, provenance={},
    )
    edge = Edge(
        edge_id="c1", source=GraphRef(kind="entity", uid="obj_6"),
        type="CONTACTS_SURFACE", target=GraphRef(kind="surface", uid="wall_1"),
        frame="world", weight=1.0, confidence=1.0,
        extractor="contacts_surface", extractor_version="0.1", evidence={},
    )
    return SceneGraphBundle(
        schema_version=CURRENT_SCHEMA_VERSION, bundle_hash="h", scene_id="t",
        frame=SceneFrame(gravity=(0.0, 0.0, -1.0), canonical_forward=None,
                         canonical_right=None, units="meters", notes=""),
        entity_bundle_hash="e", nodes=[node], edges=[edge],
        structural_surface_refs=["wall_1"], structural_surfaces=[wall],
    )


def test_score_questions_end_to_end() -> None:
    scene = _wall_scene()
    router = Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                    verbalizer=StandardVerbalizer())
    ctx = ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))
    questions = [
        {"question_id": "A", "question": "what is against the wall?",
         "expected_outcome": "answer", "expected_must_contain": ["obj_6"],
         "expected_must_not_contain": []},
        {"question_id": "B", "question": "what is on the table?",
         "expected_outcome": "defer", "expected_must_contain": [],
         "expected_must_not_contain": []},
    ]
    sc = score_questions(questions, scene, router, ctx)
    cats = {r["question_id"]: r["category"] for r in sc["per_question"]}
    if cats != {"A": "true_answer", "B": "correct_defer"}:
        raise AssertionError(f"unexpected categories: {cats}")
    if not sc["aggregate"]["all_expected_outcomes_met"]:
        raise AssertionError("expected all_met")
    if sc["aggregate"]["false_answer_count"] != 0:
        raise AssertionError("expected 0 false answers")
    # rows sorted by question_id; cited sorted
    if [r["question_id"] for r in sc["per_question"]] != ["A", "B"]:
        raise AssertionError("rows must be sorted by question_id")


# --- committed artifact --------------------------------------------------


def _load() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def _replica_v2_present() -> bool:
    return (REPLICA_V2_DIR / "scene_graph.json").exists()


def test_artifact_exists_and_kind() -> None:
    if not ARTIFACT_PATH.exists():
        raise AssertionError(f"artifact missing at {ARTIFACT_PATH}; run the tool")
    p = _load()
    if p.get("artifact_kind") != "router_qa_eval" or p.get("phase") != "P5.04":
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
    if agg["total_questions"] != 6:
        raise AssertionError(f"expected 6 questions; got {agg['total_questions']}")
    if not agg["all_expected_outcomes_met"]:
        raise AssertionError("all_expected_outcomes_met must be True")
    if agg["false_answer_count"] != 0:
        raise AssertionError("false_answer_count must be 0")
    if agg["miss_count"] != 0 or agg["unexpected_count"] != 0:
        raise AssertionError("no miss/unexpected allowed")


def test_artifact_q2_lamp_no_false_attachment() -> None:
    rows = {r["question_id"]: r for r in _load()["eval"]["per_question"]}
    q2 = rows["Q2"]
    if q2["category"] != "true_answer":
        raise AssertionError(f"Q2 must be true_answer; got {q2['category']}")
    if "obj_6" not in q2["cited_uids"]:
        raise AssertionError("Q2 must cite obj_6 (lamp)")
    for neg in ("obj_63", "obj_84", "obj_17"):
        if neg in q2["cited_uids"]:
            raise AssertionError(f"Q2 must not cite {neg} (no false attachment)")


def test_artifact_disclaims_v1_benchmark() -> None:
    limits = " ".join(_load()["interpretation_limits"]).lower()
    if "not comparable to the v1 benchmark" not in limits:
        raise AssertionError("must disclaim v1 benchmark comparability")
    if "compile metadata" not in limits:
        raise AssertionError("must record the deferral-detection limitation")


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
        import difflib
        diff = "\n".join(difflib.unified_diff(
            before.decode().splitlines(), after.decode().splitlines(), lineterm="", n=2))
        raise AssertionError(f"artifact drifted on rerun:\n{diff[:1500]}")


TESTS = [
    test_true_answer,
    test_missing_must_contain_is_miss,
    test_answer_not_bindings_is_miss,
    test_must_not_contain_present_is_false_answer,
    test_defer_deferred_is_correct_defer,
    test_defer_with_bindings_is_false_answer,
    test_defer_not_deferred_is_unexpected,
    test_empty_is_true_empty,
    test_empty_with_bindings_is_false_answer,
    test_score_questions_end_to_end,
    test_artifact_exists_and_kind,
    test_artifact_no_timestamp_keys,
    test_artifact_scorecard_all_met_zero_false,
    test_artifact_q2_lamp_no_false_attachment,
    test_artifact_disclaims_v1_benchmark,
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

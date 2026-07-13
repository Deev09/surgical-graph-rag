"""Phase 8 E3 tests: scorecard math on a synthetic bundle (no dataset).

Covers: P/R math, the exhaustive/human_verified gating (no dishonest recall),
confusion matrix, defer-rate, and the headline/plausibility aggregate split.

Run: python tests/tools/test_scene_scorecard.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.types import Plane, SceneFrame
from graph.schema import Edge, GraphRef, Node, SceneGraphBundle, SurfaceRecord
from graph.serde import CURRENT_SCHEMA_VERSION
from tools.scene_scorecard import aggregate, prf_for_question, scorecard_for_key


def _node(uid: str, label: str) -> Node:
    return Node(id=uid, label=label, label_confidence=1.0,
                centroid=(0.0, 0.0, 1.5),
                bbox_aabb=((0.0, 0.0, 1.4), (0.2, 0.1, 1.6)), bbox_obb=None,
                embedding_ref=None, attributes={"display_label": label},
                provenance={})


def _attached_edge(uid: str, n: int) -> Edge:
    return Edge(edge_id=f"att_t_{n}", source=GraphRef(kind="entity", uid=uid),
                type="ATTACHED_TO", target=GraphRef(kind="surface", uid="wall_1"),
                frame="world", weight=1.0, confidence=1.0,
                extractor="attached_to", extractor_version="0.1",
                evidence={"floor_supported": False})


def _bundle() -> SceneGraphBundle:
    wall = SurfaceRecord(
        uid="wall_1", surface_type="wall",
        plane=Plane(a=0.0, b=1.0, c=0.0, d=0.0),
        polygon=[(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 2.0), (-1.0, 0.0, 2.0)],
        source="habitat_label", confidence=1.0)
    return SceneGraphBundle(
        schema_version=CURRENT_SCHEMA_VERSION, bundle_hash="phase8_sc_synth",
        scene_id="phase8_sc_synth",
        frame=SceneFrame(gravity=(0.0, 0.0, -1.0), canonical_forward=None,
                         canonical_right=None, units="meters", notes=""),
        entity_bundle_hash="ent_synth",
        nodes=[_node("obj_a", "vent"), _node("obj_b", "sconce")],
        edges=[_attached_edge("obj_a", 1), _attached_edge("obj_b", 2)],
        structural_surface_refs=["wall_1"], structural_surfaces=[wall])


def _key(answer_key_type: str, exhaustive: bool, must_contain: list[str]) -> dict:
    return {
        "scene_id": "phase8_sc_synth",
        "fixture_id": "t",
        "answer_key_type": answer_key_type,
        "questions": [
            {"question_id": "Q01", "question": "what is attached to the wall?",
             "relation": "ATTACHED_TO", "expected_outcome": "answer",
             "expected_must_contain": must_contain,
             "expected_must_not_contain": [], "exhaustive": exhaustive},
            {"question_id": "Q02", "question": "what is on the table?",
             "relation": "ON_ENTITY_SURFACE", "expected_outcome": "empty",
             "expected_must_contain": [], "expected_must_not_contain": [],
             "exhaustive": False},
        ],
    }


def test_prf_math():
    q = {"expected_outcome": "answer", "exhaustive": True,
         "expected_must_contain": ["obj_a", "obj_c"]}
    prf = prf_for_question(q, {"obj_a", "obj_b"}, "human_verified")
    if prf is None or prf["precision"] != 0.5 or prf["recall"] != 0.5:
        raise AssertionError(f"P/R math wrong: {prf}")
    if prf["f1"] != 0.5:
        raise AssertionError(f"F1 wrong: {prf}")


def test_prf_gating_blocks_dishonest_recall():
    q = {"expected_outcome": "answer", "exhaustive": False,
         "expected_must_contain": ["obj_a"]}
    if prf_for_question(q, {"obj_a"}, "human_verified") is not None:
        raise AssertionError("non-exhaustive rows must not get P/R")
    q["exhaustive"] = True
    if prf_for_question(q, {"obj_a"}, "plausibility_labels_not_ground_truth") is not None:
        raise AssertionError("unverified keys must not get P/R")


def test_scorecard_verified_exhaustive():
    # The bundle answers Q01 with BOTH obj_a and obj_b; the verified key says
    # the exhaustive truth is only obj_a -> obj_b is a false positive:
    # P=0.5, R=1.0 on the ATTACHED_TO rollup, and the row is a false_answer?
    # No - must_not_contain is empty, so router_qa calls it true_answer;
    # the P/R layer is exactly what catches the extra citation.
    card = scorecard_for_key(_key("human_verified", True, ["obj_a"]),
                             _bundle(), {})
    rel = card["per_relation"]["ATTACHED_TO"]
    if rel["precision"] != 0.5 or rel["recall"] != 1.0:
        raise AssertionError(f"per-relation P/R wrong: {rel}")
    if card["per_relation"]["ON_ENTITY_SURFACE"]["precision"] is not None:
        raise AssertionError("empty-expected relation must not get P/R")
    if card["defer_rate"] != 0.0:
        raise AssertionError(f"defer rate wrong: {card['defer_rate']}")
    conf = card["confusion"]
    if conf["answer"] != {"bindings": 1} or conf["empty"] != {"empty": 1}:
        raise AssertionError(f"confusion wrong: {conf}")
    for row in card["router_qa"]["per_question"]:
        if not isinstance(row["latency_ms"], float):
            raise AssertionError(f"latency missing: {row}")


def test_scorecard_plausibility_gets_no_prf():
    card = scorecard_for_key(
        _key("plausibility_labels_not_ground_truth", True, ["obj_a", "obj_b"]),
        _bundle(), {})
    for rel in card["per_relation"].values():
        if rel["precision"] is not None:
            raise AssertionError("plausibility keys must never yield P/R")


def test_aggregate_split():
    verified = scorecard_for_key(_key("human_verified", True, ["obj_a", "obj_b"]),
                                 _bundle(), {})
    plaus = scorecard_for_key(
        _key("plausibility_labels_not_ground_truth", False, ["obj_a", "obj_b"]),
        _bundle(), {})
    agg = aggregate([verified, plaus])
    if agg["headline_human_verified"]["scenes"] != ["phase8_sc_synth"]:
        raise AssertionError(f"headline wrong: {agg['headline_human_verified']}")
    if agg["plausibility_not_ground_truth"]["total_questions"] != 2:
        raise AssertionError(f"plausibility rollup wrong: {agg}")


TESTS = [
    test_prf_math,
    test_prf_gating_blocks_dishonest_recall,
    test_scorecard_verified_exhaustive,
    test_scorecard_plausibility_gets_no_prf,
    test_aggregate_split,
]


def main() -> int:
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
    sys.exit(main())

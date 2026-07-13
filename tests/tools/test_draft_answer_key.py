"""Phase 8 E2 tests: draft answer-key generation on a synthetic bundle.

Run: python tests/tools/test_draft_answer_key.py

No dataset dependency: uses a synthetic ATTACHED_TO bundle (the phase7 gate
pattern), which exercises draft_from_bundle end-to-end including the
documented circularity property.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.types import Plane, SceneFrame
from eval.router_qa import score_questions
from graph.schema import Edge, GraphRef, Node, SceneGraphBundle, SurfaceRecord
from graph.serde import CURRENT_SCHEMA_VERSION
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer
from tools.draft_answer_key import PLAUSIBILITY, battery_questions, draft_from_bundle

ROUTER_QA_FIELDS = ("question_id", "question", "expected_outcome",
                    "expected_must_contain", "expected_must_not_contain")


def _bundle() -> SceneGraphBundle:
    node = Node(
        id="obj_sconce", label="wall sconce", label_confidence=1.0,
        centroid=(0.0, 0.0, 1.5),
        bbox_aabb=((0.0, 0.0, 1.4), (0.2, 0.1, 1.6)), bbox_obb=None,
        embedding_ref=None, attributes={"display_label": "wall sconce"},
        provenance={},
    )
    edge = Edge(
        edge_id="att_t_1",
        source=GraphRef(kind="entity", uid="obj_sconce"),
        type="ATTACHED_TO",
        target=GraphRef(kind="surface", uid="wall_1"),
        frame="world", weight=1.0, confidence=1.0,
        extractor="attached_to", extractor_version="0.1",
        evidence={"floor_supported": False},
    )
    wall = SurfaceRecord(
        uid="wall_1", surface_type="wall",
        plane=Plane(a=0.0, b=1.0, c=0.0, d=0.0),
        polygon=[(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 2.0), (-1.0, 0.0, 2.0)],
        source="habitat_label", confidence=1.0,
    )
    return SceneGraphBundle(
        schema_version=CURRENT_SCHEMA_VERSION, bundle_hash="phase8_synth",
        scene_id="phase8_synth",
        frame=SceneFrame(gravity=(0.0, 0.0, -1.0), canonical_forward=None,
                         canonical_right=None, units="meters", notes=""),
        entity_bundle_hash="ent_synth",
        nodes=[node], edges=[edge],
        structural_surface_refs=["wall_1"], structural_surfaces=[wall],
    )


def _draft() -> dict:
    return draft_from_bundle(_bundle(), {"obj_sconce": "wall sconce"}, "phase8_synth")


def test_battery_ids_and_relations():
    rows = battery_questions()
    if len(rows) != 13:
        raise AssertionError(f"expected 13 battery questions, got {len(rows)}")
    if rows[0]["question_id"] != "Q01" or rows[-1]["question_id"] != "Q13":
        raise AssertionError(f"unstable ids: {rows[0]}, {rows[-1]}")
    by_q = {r["question"]: r["relation"] for r in rows}
    if by_q["what is attached to the wall?"] != "ATTACHED_TO":
        raise AssertionError(f"relation map broken: {by_q}")
    if by_q["what is on the table?"] != "ON_ENTITY_SURFACE":
        raise AssertionError(f"relation map broken: {by_q}")


def test_draft_shape_and_labeling():
    draft = _draft()
    if draft["answer_key_type"] != PLAUSIBILITY:
        raise AssertionError(f"draft must be plausibility-labeled: {draft['answer_key_type']}")
    if draft.get("circular_until_reviewed") is not True:
        raise AssertionError("draft must carry circular_until_reviewed=true")
    for q in draft["questions"]:
        for f in ROUTER_QA_FIELDS:
            if f not in q:
                raise AssertionError(f"missing router_qa field {f!r} in {q['question_id']}")
        if q["exhaustive"] is not False:
            raise AssertionError("drafts must never claim exhaustiveness")
        if q["review"]["status"] != "unreviewed":
            raise AssertionError("drafts must start unreviewed")


def test_draft_records_attached_answer():
    draft = _draft()
    attached = next(q for q in draft["questions"]
                    if q["question"] == "what is attached to the wall?")
    if attached["expected_outcome"] != "answer":
        raise AssertionError(f"synthetic attached edge should answer: {attached}")
    if attached["expected_must_contain"] != ["obj_sconce"]:
        raise AssertionError(f"unexpected citation: {attached}")
    if attached["candidate_labels"] != {"obj_sconce": "wall sconce"}:
        raise AssertionError(f"labels missing: {attached}")


def test_fresh_draft_is_circular():
    """A fresh draft scores 100% against the same bundle BY CONSTRUCTION —
    this is the documented reason drafts stay plausibility-labeled."""
    draft = _draft()
    router = Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                    verbalizer=StandardVerbalizer())
    ctx = ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))
    card = score_questions(draft["questions"], _bundle(), router, ctx)
    if not card["aggregate"]["all_expected_outcomes_met"]:
        raise AssertionError(f"fresh draft must be circular-perfect: {card['aggregate']}")


TESTS = [
    test_battery_ids_and_relations,
    test_draft_shape_and_labeling,
    test_draft_records_attached_answer,
    test_fresh_draft_is_circular,
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

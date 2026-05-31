"""P1.12 bathroom-wrapper smoke tests.

Run: python tests/fixtures/test_bathroom_wrapper.py

Deliberately narrow scope per the P1.12 spec:
  - GRAFFITI_BATHROOM converts to a SceneGraphBundle without error.
  - Object IDs survive (obj_1 .. obj_12).
  - Authored edge weights survive unchanged.
  - A couple of supported questions answer correctly:
      "What is left of the sink?" -> toilet
      "What is below the mirror?" -> sink

NOT in scope:
  - All 10 v1 bathroom questions. Several depend on zone matching or
    legacy ranking semantics the Phase 1 reasoner does not implement.
    Failing those is not a bug in the wrapper; bathroom-specific logic
    would be a wrong fix.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.equality import array_aware_equal
from eval.fixtures.bathroom_v1 import load_bathroom_bundle
from graph.serde import dump_scene_graph_bundle, load_scene_graph_bundle
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer


def _oracle_ctx() -> ExecutionContext:
    return ExecutionContext(
        completeness=CompletenessProfile(
            source="oracle", entity_recall_by_class={}, edge_recall_by_type={},
        ),
    )


def test_bundle_builds() -> None:
    bundle = load_bathroom_bundle()
    if bundle.scene_id != "graffiti_bathroom":
        raise AssertionError(f"scene_id mismatch: {bundle.scene_id}")
    if len(bundle.nodes) != 12:
        raise AssertionError(f"expected 12 nodes, got {len(bundle.nodes)}")
    if not bundle.edges:
        raise AssertionError("expected non-empty edges")


def test_object_ids_preserved() -> None:
    bundle = load_bathroom_bundle()
    uids = sorted(n.id for n in bundle.nodes)
    expected = sorted(f"obj_{i}" for i in range(1, 13))
    if uids != expected:
        raise AssertionError(f"node ids drifted: {uids}")


def test_authored_weights_preserved() -> None:
    """Authored relations carry a weight in the source dict (some are
    0.4, 0.85, 1.0). The wrapper must surface them on Edge.weight AND
    record the authored value in Edge.evidence for debug."""
    bundle = load_bathroom_bundle()
    # Pull a specific weighted edge from the authored graph: obj_1 BELOW obj_3 weight=0.4
    found = [
        e for e in bundle.edges
        if e.source.uid == "obj_1" and e.type == "BELOW" and e.target.uid == "obj_3"
    ]
    if len(found) != 1:
        raise AssertionError(f"expected exactly one BELOW(obj_1, obj_3); got {len(found)}")
    e = found[0]
    if abs(e.weight - 0.4) > 1e-9:
        raise AssertionError(f"weight drift: expected 0.4, got {e.weight}")
    if e.evidence.get("authored_weight") != 0.4:
        raise AssertionError(f"evidence.authored_weight missing or wrong: {e.evidence}")


def test_unweighted_relations_default_to_one() -> None:
    """obj_1 LEFT_OF obj_2 has no weight in the source — should default to 1.0."""
    bundle = load_bathroom_bundle()
    found = [
        e for e in bundle.edges
        if e.source.uid == "obj_1" and e.type == "LEFT_OF" and e.target.uid == "obj_2"
    ]
    if len(found) != 1:
        raise AssertionError(f"expected one LEFT_OF(obj_1, obj_2); got {len(found)}")
    if abs(found[0].weight - 1.0) > 1e-9:
        raise AssertionError(f"unweighted edge should default to 1.0; got {found[0].weight}")
    if "authored_weight" in found[0].evidence:
        raise AssertionError(
            "evidence.authored_weight should only appear for explicitly-weighted edges"
        )


def test_bundle_round_trips() -> None:
    """Serde through graph.serde must round-trip cleanly."""
    bundle = load_bathroom_bundle()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bath"
        dump_scene_graph_bundle(bundle, out)
        loaded = load_scene_graph_bundle(out)
    if not array_aware_equal(bundle, loaded):
        raise AssertionError("bathroom wrapper bundle did not round-trip")


def test_left_of_sink_returns_toilet() -> None:
    bundle = load_bathroom_bundle()
    router = Router(
        compiler=RulesCompiler(), executor=RulesExecutor(),
        verbalizer=StandardVerbalizer(),
    )
    ans = router.answer("What is left of the sink?", bundle, _oracle_ctx())
    if ans.outcome != "bindings":
        raise AssertionError(f"expected bindings; got {ans.outcome} ({ans.text!r})")
    if "toilet" not in ans.text.lower():
        raise AssertionError(f"expected 'toilet' in answer; got {ans.text!r}")
    if "obj_1" not in ans.cited_uids:
        raise AssertionError(f"expected obj_1 (toilet) in cited_uids; got {ans.cited_uids}")


def test_below_mirror_returns_sink() -> None:
    bundle = load_bathroom_bundle()
    router = Router(
        compiler=RulesCompiler(), executor=RulesExecutor(),
        verbalizer=StandardVerbalizer(),
    )
    ans = router.answer("What is below the mirror?", bundle, _oracle_ctx())
    if ans.outcome != "bindings":
        raise AssertionError(f"expected bindings; got {ans.outcome} ({ans.text!r})")
    # obj_2 (sink) and obj_1 (toilet) and obj_4 (floor drain) are all BELOW mirror
    # per the authored graph. Sink must be among the bindings.
    if "obj_2" not in ans.cited_uids:
        raise AssertionError(
            f"expected obj_2 (sink) in cited_uids for 'below the mirror'; got {ans.cited_uids}"
        )


TESTS = [
    test_bundle_builds,
    test_object_ids_preserved,
    test_authored_weights_preserved,
    test_unweighted_relations_default_to_one,
    test_bundle_round_trips,
    test_left_of_sink_returns_toilet,
    test_below_mirror_returns_sink,
]


def main() -> int:
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
            print()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

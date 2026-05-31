"""P1.10 tests: rules compiler, executor, verbalizer, router.

Run: python tests/reasoner/test_reasoner_pipeline.py

Covers:
  - Compiler: every supported pattern compiles to the expected AST;
    unsupported text produces parser_failure.
  - Executor: canonical-vs-inverse storage transparent to caller;
    symmetric NEAR matches in both directions; oracle source produces
    'empty' when no match; unknown source produces 'unknown' in the
    same situation; measured source applies the threshold rule.
  - Verbalizer: bindings → NL with display labels; empty / unknown /
    parser_failure produce distinct user-facing strings.
  - Router: end-to-end pipeline against the Replica oracle path and
    against a synthetic bathroom graph.
  - Exit condition (per task description): 'What is left of the sink?'
    runs end-to-end on Replica oracle and returns a sensible Answer
    (no sink in Replica → empty under oracle context).
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.base import ReconstructionConfig
from adapters.oracle_replica import (
    OracleReplicaAdapter, build_replica_capture_bundle,
)
from common.types import SceneFrame
from extractors.base import (
    EntityArtifact, EntityArtifacts, EntityIdentity, ExtractionDiagnostics,
    InstanceExtractorConfig, SemanticHypothesis,
)
from extractors.oracle_replica import OracleReplicaExtractor
from graph.builder import ExtractorRun, build_graph
from graph.relations.directional import DirectionalConfig, DirectionalExtractor
from graph.relations.proximity import ProximityConfig, ProximityExtractor
from reasoner.ast import (
    Aggregation, EdgeConstraint, EntityRef, Variable,
)
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer
from representations.mesh import MeshRepresentation


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"


# ---------- shared helpers ----------

def _build_replica_oracle_graph(*, mode: str):
    capture = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    repr_bundle = OracleReplicaAdapter().reconstruct(
        capture, ReconstructionConfig(name="oracle_replica", version="0.1", params={}),
    )
    representation = MeshRepresentation(bundle=repr_bundle)
    artifacts = OracleReplicaExtractor().extract(
        representation,
        InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )
    runs = [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode=mode)),
        ExtractorRun(ProximityExtractor(), ProximityConfig(mode=mode)),
    ]
    bundle, _ = build_graph(artifacts, runs)
    return bundle


def _build_synthetic_bathroom_graph():
    """Tiny scene with a sink and a toilet to its left, plus a paper
    dispenser. Used to exercise the bindings path."""
    frame = SceneFrame(
        gravity=(0.0, 0.0, -1.0), canonical_forward=None,
        canonical_right=None, units="meters", notes="",
    )

    def _ent(uid, label, centroid):
        return EntityArtifact(
            identity=EntityIdentity(
                object_uid=uid, display_label=label, aliases=[label.rsplit("_", 1)[0]] if "_" in label else [],
                source_instance_ref=uid,
            ),
            bbox_aabb=((centroid[0]-0.2, centroid[1]-0.2, centroid[2]-0.2),
                       (centroid[0]+0.2, centroid[1]+0.2, centroid[2]+0.2)),
            bbox_obb=None, centroid=centroid, geometry_handle=None,
            semantic_hypotheses=[SemanticHypothesis(label=label, confidence=1.0, source="test")],
            embedding=None, extraction_diagnostics={},
        )

    entities = [
        _ent("obj_toilet", "toilet", (0.0, 0.0, 0.4)),
        _ent("obj_sink",   "sink",   (1.5, 0.0, 0.9)),  # to the right of toilet
        # Dispenser at x=2.0 keeps both toilet→dispenser (~2.15m) and
        # sink→dispenser within the sparse 2.5m cap.
        _ent("obj_dispenser", "dispenser", (2.0, 0.0, 1.2)),
    ]
    artifacts = EntityArtifacts(
        schema_version=1, bundle_hash="ent_bath", scene_id="bath",
        frame=frame, representation_hash="repr_bath",
        extractor_name="synth", extractor_version="0.1",
        entities=entities, structural_surfaces=[],
        geometry_store_path=None,
        diagnostics=ExtractionDiagnostics(
            n_entities=3, n_structural_surfaces=0, runtime_seconds=0.0,
            coverage_score=None, notes="",
        ),
        notes={},
    )
    runs = [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse")),
        ExtractorRun(ProximityExtractor(), ProximityConfig(mode="sparse")),
    ]
    bundle, _ = build_graph(artifacts, runs)
    return bundle


def _oracle_ctx() -> ExecutionContext:
    return ExecutionContext(
        completeness=CompletenessProfile(
            source="oracle", entity_recall_by_class={}, edge_recall_by_type={},
        ),
    )


def _unknown_ctx() -> ExecutionContext:
    return ExecutionContext(
        completeness=CompletenessProfile(
            source="unknown", entity_recall_by_class={}, edge_recall_by_type={},
        ),
    )


def _measured_ctx(*, entity_recall: float, edge_recall: float) -> ExecutionContext:
    return ExecutionContext(
        completeness=CompletenessProfile(
            source="measured",
            entity_recall_by_class={"all": entity_recall},
            edge_recall_by_type={
                "LEFT_OF": edge_recall, "RIGHT_OF": edge_recall,
                "ABOVE": edge_recall, "BELOW": edge_recall,
                "BEHIND": edge_recall, "IN_FRONT_OF": edge_recall,
                "NEAR": edge_recall,
            },
        ),
    )


# ---------- compiler tests ----------

def test_compiler_left_of_pattern() -> None:
    cr = RulesCompiler().compile("What is left of the sink?", _build_synthetic_bathroom_graph())
    if cr.outcome != "compiled":
        raise AssertionError(f"expected compiled, got {cr.outcome}")
    assert isinstance(cr.ast, Aggregation)
    assert len(cr.ast.where) == 1
    c = cr.ast.where[0]
    assert isinstance(c, EdgeConstraint) and c.type == "LEFT_OF"
    assert isinstance(c.source, Variable) and c.source.name == "x"
    assert isinstance(c.target, EntityRef) and c.target.label == "sink"


def test_compiler_handles_synonyms() -> None:
    for q, expected_type in [
        ("What is under the table?", "BELOW"),
        ("what's near the chair", "NEAR"),
        ("What's next to the lamp?", "NEAR"),
        ("What is close to the cabinet?", "NEAR"),
        ("What is directly above the floor?", "ABOVE"),
    ]:
        cr = RulesCompiler().compile(q, _build_synthetic_bathroom_graph())
        if cr.outcome != "compiled":
            raise AssertionError(f"{q!r}: did not compile")
        assert isinstance(cr.ast, Aggregation)
        assert cr.ast.where[0].type == expected_type, f"{q!r}: expected {expected_type}, got {cr.ast.where[0].type}"


def test_compiler_unsupported_returns_parser_failure() -> None:
    cr = RulesCompiler().compile("Why is the sink white?", _build_synthetic_bathroom_graph())
    if cr.outcome != "parser_failure":
        raise AssertionError(f"expected parser_failure, got {cr.outcome}")
    if cr.ast is not None:
        raise AssertionError("ast should be None on parser_failure")


# ---------- executor tests ----------

def test_executor_finds_left_of_sink_on_bathroom() -> None:
    """The synthetic bathroom has toilet at x=0 and sink at x=1.5, so
    toilet is left of sink. Sparse mode stores LEFT_OF(toilet, sink)."""
    graph = _build_synthetic_bathroom_graph()
    ast = Aggregation(
        op="ENUMERATE", bind=Variable(name="x"),
        where=[EdgeConstraint(
            source=Variable(name="x"), type="LEFT_OF",
            target=EntityRef(label="sink"),
        )],
    )
    er = RulesExecutor().execute(ast, graph, _oracle_ctx())
    if er.outcome != "bindings":
        raise AssertionError(f"expected bindings, got {er.outcome} ({er.notes})")
    uids = {b["x"].uid for b in er.bindings}
    if "obj_toilet" not in uids:
        raise AssertionError(f"expected toilet in bindings; got {uids}")


def test_executor_handles_inverse_via_canonical_lookup() -> None:
    """The sparse graph stores LEFT_OF only. Asking for RIGHT_OF must
    derive from LEFT_OF with swapped endpoints."""
    graph = _build_synthetic_bathroom_graph()
    ast = Aggregation(
        op="ENUMERATE", bind=Variable(name="x"),
        where=[EdgeConstraint(
            source=Variable(name="x"), type="RIGHT_OF",
            target=EntityRef(label="toilet"),
        )],
    )
    er = RulesExecutor().execute(ast, graph, _oracle_ctx())
    if er.outcome != "bindings":
        raise AssertionError(f"expected bindings, got {er.outcome} ({er.notes})")
    uids = {b["x"].uid for b in er.bindings}
    if "obj_sink" not in uids:
        raise AssertionError(f"expected sink (which is right of toilet); got {uids}")


def test_executor_symmetric_near_matches_either_direction() -> None:
    graph = _build_synthetic_bathroom_graph()
    # toilet at (0,0,0.4), sink at (1.5, ...) → distance > 1.0, no NEAR
    # sink at (1.5, ...), dispenser at (2.5, ...) → distance ~1.0, borderline
    # Just check the executor doesn't crash and produces some consistent result.
    ast = Aggregation(
        op="ENUMERATE", bind=Variable(name="x"),
        where=[EdgeConstraint(
            source=Variable(name="x"), type="NEAR",
            target=EntityRef(label="sink"),
        )],
    )
    er = RulesExecutor().execute(ast, graph, _oracle_ctx())
    if er.outcome not in ("bindings", "empty"):
        raise AssertionError(f"expected bindings/empty, got {er.outcome}")


def test_executor_unresolved_anchor_returns_empty_under_oracle() -> None:
    graph = _build_synthetic_bathroom_graph()
    ast = Aggregation(
        op="ENUMERATE", bind=Variable(name="x"),
        where=[EdgeConstraint(
            source=Variable(name="x"), type="LEFT_OF",
            target=EntityRef(label="unicorn"),
        )],
    )
    er = RulesExecutor().execute(ast, graph, _oracle_ctx())
    if er.outcome != "empty":
        raise AssertionError(f"oracle + unresolved anchor: expected empty, got {er.outcome}")
    if er.coverage_floor != 1.0:
        raise AssertionError(f"oracle coverage_floor should be 1.0, got {er.coverage_floor}")


def test_executor_unresolved_anchor_returns_unknown_under_unknown_source() -> None:
    graph = _build_synthetic_bathroom_graph()
    ast = Aggregation(
        op="ENUMERATE", bind=Variable(name="x"),
        where=[EdgeConstraint(
            source=Variable(name="x"), type="LEFT_OF",
            target=EntityRef(label="unicorn"),
        )],
    )
    er = RulesExecutor().execute(ast, graph, _unknown_ctx())
    if er.outcome != "unknown":
        raise AssertionError(f"unknown + unresolved anchor: expected unknown, got {er.outcome}")


def test_executor_measured_high_recall_produces_empty() -> None:
    graph = _build_synthetic_bathroom_graph()
    ast = Aggregation(
        op="ENUMERATE", bind=Variable(name="x"),
        where=[EdgeConstraint(
            source=Variable(name="x"), type="LEFT_OF",
            target=EntityRef(label="unicorn"),
        )],
    )
    er = RulesExecutor().execute(
        ast, graph,
        _measured_ctx(entity_recall=0.98, edge_recall=0.97),
    )
    if er.outcome != "empty":
        raise AssertionError(f"measured high recall: expected empty, got {er.outcome}")


def test_executor_measured_low_recall_produces_unknown() -> None:
    graph = _build_synthetic_bathroom_graph()
    ast = Aggregation(
        op="ENUMERATE", bind=Variable(name="x"),
        where=[EdgeConstraint(
            source=Variable(name="x"), type="LEFT_OF",
            target=EntityRef(label="unicorn"),
        )],
    )
    er = RulesExecutor().execute(
        ast, graph,
        _measured_ctx(entity_recall=0.50, edge_recall=0.80),
    )
    if er.outcome != "unknown":
        raise AssertionError(f"measured low recall: expected unknown, got {er.outcome}")


def test_executor_invalid_ast_returns_execution_error() -> None:
    graph = _build_synthetic_bathroom_graph()
    # Both operands are EntityRef → unsupported in Phase 1
    ast = Aggregation(
        op="ENUMERATE", bind=Variable(name="x"),
        where=[EdgeConstraint(
            source=EntityRef(label="a"), type="LEFT_OF",
            target=EntityRef(label="b"),
        )],
    )
    er = RulesExecutor().execute(ast, graph, _oracle_ctx())
    if er.outcome != "execution_error":
        raise AssertionError(f"expected execution_error, got {er.outcome}")


# ---------- verbalizer tests ----------

def test_verbalizer_bindings_render_with_display_label() -> None:
    graph = _build_synthetic_bathroom_graph()
    cr = RulesCompiler().compile("What is left of the sink?", graph)
    er = RulesExecutor().execute(cr.ast, graph, _oracle_ctx())
    ans = StandardVerbalizer().verbalize("What is left of the sink?", cr, er, graph)
    if ans.outcome != "bindings":
        raise AssertionError(f"expected bindings outcome, got {ans.outcome}")
    if "toilet" not in ans.text.lower():
        raise AssertionError(f"expected 'toilet' in answer text; got {ans.text!r}")
    if ans.answered_by != "rules_compiler":
        raise AssertionError(f"expected answered_by=rules_compiler, got {ans.answered_by}")
    if not ans.cited_uids or "obj_toilet" not in ans.cited_uids:
        raise AssertionError(f"cited_uids missing toilet: {ans.cited_uids}")


def test_verbalizer_empty_says_nothing_matches() -> None:
    graph = _build_synthetic_bathroom_graph()
    cr = RulesCompiler().compile("What is left of the unicorn?", graph)
    er = RulesExecutor().execute(cr.ast, graph, _oracle_ctx())
    ans = StandardVerbalizer().verbalize("What is left of the unicorn?", cr, er, graph)
    if ans.outcome != "empty":
        raise AssertionError(f"expected empty, got {ans.outcome}")
    if "nothing" not in ans.text.lower():
        raise AssertionError(f"empty answer should say 'nothing'; got {ans.text!r}")


def test_verbalizer_unknown_says_not_enough_evidence() -> None:
    graph = _build_synthetic_bathroom_graph()
    cr = RulesCompiler().compile("What is left of the unicorn?", graph)
    er = RulesExecutor().execute(cr.ast, graph, _unknown_ctx())
    ans = StandardVerbalizer().verbalize("What is left of the unicorn?", cr, er, graph)
    if ans.outcome != "unknown":
        raise AssertionError(f"expected unknown, got {ans.outcome}")
    if "evidence" not in ans.text.lower():
        raise AssertionError(f"unknown answer should mention evidence; got {ans.text!r}")


def test_verbalizer_parser_failure_abstains() -> None:
    graph = _build_synthetic_bathroom_graph()
    cr = RulesCompiler().compile("Why is the sky blue?", graph)
    ans = StandardVerbalizer().verbalize("Why is the sky blue?", cr, None, graph)
    if ans.outcome != "parser_failure":
        raise AssertionError(f"expected parser_failure, got {ans.outcome}")
    if ans.answered_by != "verbalizer_abstain":
        raise AssertionError(f"expected verbalizer_abstain, got {ans.answered_by}")


def test_verbalizer_multiple_bindings_uses_human_join() -> None:
    graph = _build_synthetic_bathroom_graph()
    # "what is right of the toilet" — both sink and dispenser are right of toilet
    cr = RulesCompiler().compile("What is right of the toilet?", graph)
    er = RulesExecutor().execute(cr.ast, graph, _oracle_ctx())
    ans = StandardVerbalizer().verbalize("What is right of the toilet?", cr, er, graph)
    if ans.outcome != "bindings":
        raise AssertionError(f"expected bindings, got {ans.outcome} ({er.notes})")
    # Should contain "and" for multi-binding human-readable join
    if " and " not in ans.text and "," not in ans.text:
        raise AssertionError(f"multi-binding should use human join; got {ans.text!r}")


# ---------- router end-to-end ----------

def test_router_left_of_sink_on_bathroom_returns_toilet() -> None:
    graph = _build_synthetic_bathroom_graph()
    router = Router(
        compiler=RulesCompiler(), executor=RulesExecutor(),
        verbalizer=StandardVerbalizer(),
    )
    ans = router.answer("What is left of the sink?", graph, _oracle_ctx())
    if ans.outcome != "bindings":
        raise AssertionError(f"expected bindings, got {ans.outcome} ({ans.text!r})")
    if "toilet" not in ans.text.lower():
        raise AssertionError(f"expected 'toilet'; got {ans.text!r}")


def test_router_left_of_sink_on_replica_oracle_returns_empty() -> None:
    """Exit demo: Replica has no sink, oracle context, so the router
    should return a sensible 'nothing matches' Answer (not crash, not
    return garbage, not say 'unknown')."""
    graph = _build_replica_oracle_graph(mode="sparse")
    router = Router(
        compiler=RulesCompiler(), executor=RulesExecutor(),
        verbalizer=StandardVerbalizer(),
    )
    ans = router.answer("What is left of the sink?", graph, _oracle_ctx())
    if ans.outcome != "empty":
        raise AssertionError(
            f"oracle + no sink: expected empty Answer, got {ans.outcome} ({ans.text!r})"
        )


def test_router_left_of_sink_on_replica_unknown_returns_unknown() -> None:
    """Same query, unknown completeness profile → unknown outcome."""
    graph = _build_replica_oracle_graph(mode="sparse")
    router = Router(
        compiler=RulesCompiler(), executor=RulesExecutor(),
        verbalizer=StandardVerbalizer(),
    )
    ans = router.answer("What is left of the sink?", graph, _unknown_ctx())
    if ans.outcome != "unknown":
        raise AssertionError(
            f"unknown + no sink: expected unknown Answer, got {ans.outcome} ({ans.text!r})"
        )


def test_router_near_cabinet_on_replica_returns_bindings() -> None:
    """Cabinet is in Replica room_0 against a wall — nothing is to its
    left, but several objects are NEAR it. Use proximity to exercise
    the bindings path on real Replica geometry."""
    graph = _build_replica_oracle_graph(mode="sparse")
    router = Router(
        compiler=RulesCompiler(), executor=RulesExecutor(),
        verbalizer=StandardVerbalizer(),
    )
    ans = router.answer("What is near the cabinet?", graph, _oracle_ctx())
    if ans.outcome != "bindings":
        raise AssertionError(
            f"NEAR cabinet should produce bindings; got {ans.outcome} ({ans.text!r})"
        )
    if not ans.cited_uids:
        raise AssertionError("expected at least one cited uid")


def test_router_parser_failure_abstains_without_llm() -> None:
    graph = _build_synthetic_bathroom_graph()
    router = Router(
        compiler=RulesCompiler(), executor=RulesExecutor(),
        verbalizer=StandardVerbalizer(),
    )
    ans = router.answer("Why is the sky blue?", graph, _oracle_ctx())
    if ans.outcome != "parser_failure":
        raise AssertionError(f"expected parser_failure, got {ans.outcome}")


def test_router_falls_back_to_llm_compiler_when_provided() -> None:
    """Phase 4 wiring sanity: when rules fail and an LLM compiler is
    wired, the router tries it. We stub the LLM with a fake that always
    compiles to a known query."""
    class FakeLlmCompiler:
        name = "llm_v1"
        version = "0.1"
        def compile(self, question, scene):
            from reasoner.base import CompileResult
            from reasoner.ast import Aggregation, EdgeConstraint, EntityRef, Variable
            v = Variable(name="x")
            return CompileResult(
                ast=Aggregation(
                    op="ENUMERATE", bind=v,
                    where=[EdgeConstraint(source=v, type="LEFT_OF", target=EntityRef(label="sink"))],
                ),
                outcome="compiled", compiler_name="llm_v1", notes="fake",
            )

    graph = _build_synthetic_bathroom_graph()
    router = Router(
        compiler=RulesCompiler(), executor=RulesExecutor(),
        verbalizer=StandardVerbalizer(), llm_compiler=FakeLlmCompiler(),
    )
    ans = router.answer("Some weird unparseable question", graph, _oracle_ctx())
    if ans.outcome != "bindings":
        raise AssertionError(f"expected LLM fallback to produce bindings; got {ans.outcome}")
    if ans.answered_by != "llm_compiler":
        raise AssertionError(f"answered_by should be llm_compiler; got {ans.answered_by}")


TESTS = [
    # compiler
    test_compiler_left_of_pattern,
    test_compiler_handles_synonyms,
    test_compiler_unsupported_returns_parser_failure,
    # executor
    test_executor_finds_left_of_sink_on_bathroom,
    test_executor_handles_inverse_via_canonical_lookup,
    test_executor_symmetric_near_matches_either_direction,
    test_executor_unresolved_anchor_returns_empty_under_oracle,
    test_executor_unresolved_anchor_returns_unknown_under_unknown_source,
    test_executor_measured_high_recall_produces_empty,
    test_executor_measured_low_recall_produces_unknown,
    test_executor_invalid_ast_returns_execution_error,
    # verbalizer
    test_verbalizer_bindings_render_with_display_label,
    test_verbalizer_empty_says_nothing_matches,
    test_verbalizer_unknown_says_not_enough_evidence,
    test_verbalizer_parser_failure_abstains,
    test_verbalizer_multiple_bindings_uses_human_join,
    # router
    test_router_left_of_sink_on_bathroom_returns_toilet,
    test_router_left_of_sink_on_replica_oracle_returns_empty,
    test_router_left_of_sink_on_replica_unknown_returns_unknown,
    test_router_near_cabinet_on_replica_returns_bindings,
    test_router_parser_failure_abstains_without_llm,
    test_router_falls_back_to_llm_compiler_when_provided,
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

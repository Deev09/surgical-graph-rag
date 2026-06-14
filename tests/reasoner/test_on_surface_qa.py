"""P4.04 tests: "what is on the floor?" QA over the support view.

Proves the normal compiler -> executor -> verbalizer path answers the floor
support query from graph structure (the SUPPORTS derived view over
ON_SURFACE edges). P6 extends SUPPORTS to entity surfaces, so table/chair
queries compile now; unsupported support classes still defer.

Run: python tests/reasoner/test_on_surface_qa.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.base import ReconstructionConfig
from adapters.oracle_replica import OracleReplicaAdapter, build_replica_capture_bundle
from common.types import Plane, SceneFrame
from extractors.base import InstanceExtractorConfig
from extractors.oracle_replica import OracleReplicaExtractor
from graph.builder import ExtractorRun, build_graph
from graph.relations.on_surface import OnSurfaceConfig, OnSurfaceExtractor
from graph.schema import (
    Edge, GraphRef, Node, SceneGraphBundle, SurfaceRecord,
)
from graph.serde import CURRENT_SCHEMA_VERSION
from reasoner.ast import Aggregation, EdgeConstraint, SurfaceRef, Variable
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer
from representations.mesh import MeshRepresentation


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
REPLICA_V2_DIR = REPLICA_SCENE_DIR / "enriched" / "v2"


def _frame() -> SceneFrame:
    return SceneFrame(
        gravity=(0.0, 0.0, -1.0), canonical_forward=None,
        canonical_right=None, units="meters", notes="",
    )


def _oracle_ctx() -> ExecutionContext:
    return ExecutionContext(
        completeness=CompletenessProfile(
            source="oracle", entity_recall_by_class={}, edge_recall_by_type={},
        ),
    )


def _node(uid, label) -> Node:
    return Node(
        id=uid, label=label, label_confidence=1.0,
        centroid=(0.0, 0.0, 0.0), bbox_aabb=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        bbox_obb=None, embedding_ref=None,
        attributes={"display_label": label}, provenance={},
    )


def _on_surface_edge(edge_id, entity_uid, surface_uid) -> Edge:
    return Edge(
        edge_id=edge_id,
        source=GraphRef(kind="entity", uid=entity_uid),
        type="ON_SURFACE",
        target=GraphRef(kind="surface", uid=surface_uid),
        frame="world", weight=1.0, confidence=1.0,
        extractor="on_surface", extractor_version="0.1",
        evidence={"bottom_gap_m": -0.01, "contact": True},
    )


def _floor_surface(uid="floor_1") -> SurfaceRecord:
    return SurfaceRecord(
        uid=uid, surface_type="floor",
        plane=Plane(a=0.0, b=0.0, c=1.0, d=0.0),
        polygon=[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)],
        source="habitat_label", confidence=1.0,
    )


def _bundle(nodes, edges, surfaces) -> SceneGraphBundle:
    return SceneGraphBundle(
        schema_version=CURRENT_SCHEMA_VERSION,
        bundle_hash="h", scene_id="t", frame=_frame(),
        entity_bundle_hash="e", nodes=list(nodes), edges=list(edges),
        structural_surface_refs=[s.uid for s in surfaces],
        structural_surfaces=list(surfaces),
    )


def _floor_scene_with_two_on_floor() -> SceneGraphBundle:
    return _bundle(
        nodes=[_node("obj_1", "stool"), _node("obj_2", "basket"), _node("obj_3", "lamp")],
        edges=[
            _on_surface_edge("e_on_1", "obj_1", "floor_1"),
            _on_surface_edge("e_on_2", "obj_2", "floor_1"),
        ],
        surfaces=[_floor_surface()],
    )


def _replica_v2_present() -> bool:
    return (REPLICA_V2_DIR / "scene_graph.json").exists()


# --- compile ------------------------------------------------------------


def test_floor_query_compiles() -> None:
    cr = RulesCompiler().compile("what is on the floor?", _floor_scene_with_two_on_floor())
    if cr.outcome != "compiled" or cr.ast is None:
        raise AssertionError(f"floor query must compile; got {cr.outcome} ({cr.notes})")
    agg = cr.ast
    if not isinstance(agg, Aggregation) or len(agg.where) != 1:
        raise AssertionError("expected single-constraint Aggregation")
    c = agg.where[0]
    if not isinstance(c, EdgeConstraint) or c.type != "SUPPORTS":
        raise AssertionError(f"expected SUPPORTS constraint; got {c}")
    if not isinstance(c.source, SurfaceRef) or c.source.surface_type != "floor":
        raise AssertionError(f"expected SurfaceRef(floor) source; got {c.source}")
    if not isinstance(c.target, Variable):
        raise AssertionError("expected Variable target")


# --- execute: bindings from the support view ----------------------------


def test_floor_query_returns_bindings_from_support_view() -> None:
    scene = _floor_scene_with_two_on_floor()
    cr = RulesCompiler().compile("what is on the floor?", scene)
    er = RulesExecutor().execute(cr.ast, scene, _oracle_ctx())
    if er.outcome != "bindings":
        raise AssertionError(f"expected bindings; got {er.outcome} ({er.notes})")
    bound_uids = sorted(b["x"].uid for b in er.bindings)
    if bound_uids != ["obj_1", "obj_2"]:
        raise AssertionError(f"expected obj_1,obj_2; got {bound_uids}")
    # evidence must be the stored ON_SURFACE edges, never a SUPPORTS edge
    if any(e.type != "ON_SURFACE" for e in er.evidence):
        raise AssertionError("evidence must cite ON_SURFACE edges only")
    if len(er.evidence) != 2:
        raise AssertionError(f"expected 2 evidence edges; got {len(er.evidence)}")


def test_cited_evidence_points_to_on_surface_edges() -> None:
    scene = _floor_scene_with_two_on_floor()
    router = Router(
        compiler=RulesCompiler(), executor=RulesExecutor(),
        verbalizer=StandardVerbalizer(),
    )
    ans = router.answer("what is on the floor?", scene, _oracle_ctx())
    if ans.outcome != "bindings":
        raise AssertionError(f"expected bindings answer; got {ans.outcome}")
    on_ids = {e.edge_id for e in scene.edges if e.type == "ON_SURFACE"}
    if not ans.cited_edges or any(eid not in on_ids for eid in ans.cited_edges):
        raise AssertionError(
            f"cited_edges must all be ON_SURFACE edge ids; got {ans.cited_edges}"
        )
    if sorted(ans.cited_uids) != ["obj_1", "obj_2"]:
        raise AssertionError(f"cited_uids wrong: {ans.cited_uids}")


def test_isolation_no_on_surface_edges_yields_empty_not_bindings() -> None:
    """ON_SURFACE only appears when the extractor is explicitly run. A graph
    with a floor surface but no ON_SURFACE edges must yield empty/unknown,
    proving the QA path does not invent support from nowhere."""
    scene = _bundle(
        nodes=[_node("obj_1", "stool")], edges=[], surfaces=[_floor_surface()],
    )
    cr = RulesCompiler().compile("what is on the floor?", scene)
    er = RulesExecutor().execute(cr.ast, scene, _oracle_ctx())
    if er.outcome not in ("empty", "unknown"):
        raise AssertionError(
            f"no ON_SURFACE edges must give empty/unknown; got {er.outcome}"
        )
    if er.bindings:
        raise AssertionError("must not fabricate bindings without ON_SURFACE edges")


# --- P6 entity-support compile + unsupported deferral --------------------


def _deferred_answer(question: str):
    scene = _floor_scene_with_two_on_floor()
    cr = RulesCompiler().compile(question, scene)
    router = Router(
        compiler=RulesCompiler(), executor=RulesExecutor(),
        verbalizer=StandardVerbalizer(),
    )
    ans = router.answer(question, scene, _oracle_ctx())
    return cr, ans


def test_table_query_compiles_in_p6() -> None:
    cr, ans = _deferred_answer("what is on the table?")
    if cr.outcome != "compiled" or "entity_class=table" not in cr.notes:
        raise AssertionError(f"table must compile in P6; got {cr.outcome} ({cr.notes})")
    if ans.outcome not in ("empty", "unknown"):
        raise AssertionError(f"floor-only synthetic scene has no table support; got {ans.outcome}")


def test_chair_query_compiles_in_p6() -> None:
    cr, ans = _deferred_answer("what is on the chair?")
    if cr.outcome != "compiled" or "entity_class=chair" not in cr.notes:
        raise AssertionError(f"chair must compile in P6; got {cr.outcome} ({cr.notes})")
    if ans.outcome not in ("empty", "unknown"):
        raise AssertionError(f"floor-only synthetic scene has no chair support; got {ans.outcome}")


def test_unsupported_on_query_defers() -> None:
    cr, ans = _deferred_answer("what is on the cabinet?")
    if cr.outcome != "out_of_schema" or not cr.notes.startswith("deferred:"):
        raise AssertionError(f"cabinet must defer; got {cr.outcome} ({cr.notes})")
    if ans.outcome != "abstain" or ans.cited_uids:
        raise AssertionError("unsupported support deferral must abstain with no citations")


def test_wall_query_no_longer_defers_in_p5() -> None:
    """BEHAVIOR CHANGE (P4 -> P5): "what is against the wall?" used to defer
    (wall contact not in P4). P5.03 makes it answerable -- it now compiles to
    CONTACTS_SURFACE(?x, SurfaceRef("wall")). On this floor-only synthetic
    scene there is no wall surface, so it executes to empty/unknown (not a
    deferral, not bindings). The deferred-surface coverage now lives in
    tests/reasoner/test_wall_contact_qa.py."""
    scene = _floor_scene_with_two_on_floor()
    cr = RulesCompiler().compile("what is against the wall?", scene)
    if cr.outcome != "compiled":
        raise AssertionError(
            f"against-the-wall must compile in P5 (no longer deferred); got {cr.outcome}"
        )
    c = cr.ast.where[0]
    if c.type != "CONTACTS_SURFACE":
        raise AssertionError(f"expected CONTACTS_SURFACE; got {c.type}")
    er = RulesExecutor().execute(cr.ast, scene, _oracle_ctx())
    if er.outcome not in ("empty", "unknown") or er.bindings:
        raise AssertionError(
            f"wall-less scene must yield empty/unknown; got {er.outcome}"
        )


def test_deferred_is_not_parser_failure() -> None:
    """Deferred must be out_of_schema, distinct from parser_failure (which
    is reserved for genuinely unparseable questions)."""
    scene = _floor_scene_with_two_on_floor()
    deferred = RulesCompiler().compile("what is on the cabinet?", scene)
    if deferred.outcome == "parser_failure":
        raise AssertionError("deferred must not be parser_failure")
    nonsense = RulesCompiler().compile("how many wugs frobnicate?", scene)
    if nonsense.outcome != "parser_failure":
        raise AssertionError(f"unparseable must be parser_failure; got {nonsense.outcome}")


# --- real Replica F1 -----------------------------------------------------


def test_real_replica_floor_answer_includes_stool() -> None:
    if not _replica_v2_present():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    capture = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    representation = MeshRepresentation(
        bundle=OracleReplicaAdapter().reconstruct(
            capture,
            ReconstructionConfig(name="oracle_replica", version="0.1", params={}),
        ),
    )
    artifacts = OracleReplicaExtractor(enriched_v2_path=REPLICA_V2_DIR).extract(
        representation,
        InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )
    bundle, _diag = build_graph(
        artifacts,
        [ExtractorRun(OnSurfaceExtractor(), OnSurfaceConfig())],
        density_policy="phase2_telemetry_only",
    )
    router = Router(
        compiler=RulesCompiler(), executor=RulesExecutor(),
        verbalizer=StandardVerbalizer(),
    )
    ans = router.answer("what is on the floor?", bundle, _oracle_ctx())
    if ans.outcome != "bindings":
        raise AssertionError(f"expected bindings on real Replica; got {ans.outcome}")
    if "obj_39" not in ans.cited_uids:
        raise AssertionError(
            f"F1 stool obj_39 must be in the on-floor answer; cited {ans.cited_uids}"
        )
    # all cited edges are ON_SURFACE
    on_ids = {e.edge_id for e in bundle.edges if e.type == "ON_SURFACE"}
    if any(eid not in on_ids for eid in ans.cited_edges):
        raise AssertionError("real-data citations must all be ON_SURFACE edges")


TESTS = [
    test_floor_query_compiles,
    test_floor_query_returns_bindings_from_support_view,
    test_cited_evidence_points_to_on_surface_edges,
    test_isolation_no_on_surface_edges_yields_empty_not_bindings,
    test_table_query_compiles_in_p6,
    test_chair_query_compiles_in_p6,
    test_unsupported_on_query_defers,
    test_wall_query_no_longer_defers_in_p5,
    test_deferred_is_not_parser_failure,
    test_real_replica_floor_answer_includes_stool,
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

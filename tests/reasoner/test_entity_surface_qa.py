"""P6.04 tests: QA over EntitySurface support.

Run: python tests/reasoner/test_entity_surface_qa.py
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
from common.types import SceneFrame
from extractors.base import InstanceExtractorConfig
from extractors.oracle_replica import OracleReplicaExtractor
from graph.builder import ExtractorRun, build_graph
from graph.relations.on_entity_surface import (
    OnEntitySurfaceConfig,
    OnEntitySurfaceExtractor,
)
from graph.schema import Edge, GraphRef, Node, SceneGraphBundle
from graph.serde import CURRENT_SCHEMA_VERSION
from reasoner.ast import Aggregation, EdgeConstraint, EntityClassRef, Variable
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
        centroid=(0.0, 0.0, 0.0),
        bbox_aabb=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        bbox_obb=None, embedding_ref=None,
        attributes={"display_label": label}, provenance={},
    )


def _on_entity_surface_edge(edge_id, supported_uid, supporter_uid) -> Edge:
    return Edge(
        edge_id=edge_id,
        source=GraphRef(kind="entity", uid=supported_uid),
        type="ON_ENTITY_SURFACE",
        target=GraphRef(kind="entity", uid=supporter_uid),
        frame="world",
        weight=1.0,
        confidence=1.0,
        extractor="on_entity_surface",
        extractor_version="0.1",
        evidence={
            "owner_entity_uid": supporter_uid,
            "entity_surface_uid": f"ent_surf_{supporter_uid}_top",
            "owner_class": "table",
            "bottom_gap_m": 0.0,
            "contact": True,
        },
    )


def _bundle(nodes, edges) -> SceneGraphBundle:
    return SceneGraphBundle(
        schema_version=CURRENT_SCHEMA_VERSION,
        bundle_hash="h",
        scene_id="t",
        frame=_frame(),
        entity_bundle_hash="e",
        nodes=list(nodes),
        edges=list(edges),
        structural_surface_refs=[],
        structural_surfaces=[],
    )


def _scene() -> SceneGraphBundle:
    return _bundle(
        nodes=[
            _node("book_1", "book_1"),
            _node("lamp_1", "lamp_1"),
            _node("table_1", "table_1"),
            _node("chair_1", "chair_1"),
        ],
        edges=[
            _on_entity_surface_edge("oes_1", "book_1", "table_1"),
            _on_entity_surface_edge("oes_2", "lamp_1", "table_1"),
        ],
    )


def _router() -> Router:
    return Router(
        compiler=RulesCompiler(), executor=RulesExecutor(),
        verbalizer=StandardVerbalizer(),
    )


def _replica_v2_present() -> bool:
    return (REPLICA_V2_DIR / "scene_graph.json").exists()


def test_table_query_compiles_to_entity_class_ref() -> None:
    cr = RulesCompiler().compile("what is on the table?", _scene())
    if cr.outcome != "compiled" or cr.ast is None:
        raise AssertionError(f"table query must compile; got {cr.outcome} ({cr.notes})")
    agg = cr.ast
    if not isinstance(agg, Aggregation) or len(agg.where) != 1:
        raise AssertionError("expected single-constraint Aggregation")
    c = agg.where[0]
    if not isinstance(c, EdgeConstraint) or c.type != "SUPPORTS":
        raise AssertionError(f"expected SUPPORTS constraint; got {c}")
    if not isinstance(c.source, EntityClassRef) or c.source.entity_class != "table":
        raise AssertionError(f"expected EntityClassRef(table); got {c.source}")
    if not isinstance(c.target, Variable):
        raise AssertionError("expected Variable target")


def test_table_query_returns_entity_surface_bindings() -> None:
    scene = _scene()
    cr = RulesCompiler().compile("what is on the table?", scene)
    er = RulesExecutor().execute(cr.ast, scene, _oracle_ctx())
    if er.outcome != "bindings":
        raise AssertionError(f"expected bindings; got {er.outcome} ({er.notes})")
    if sorted(b["x"].uid for b in er.bindings) != ["book_1", "lamp_1"]:
        raise AssertionError(f"wrong bindings: {er.bindings}")
    if any(e.type != "ON_ENTITY_SURFACE" for e in er.evidence):
        raise AssertionError("evidence must cite ON_ENTITY_SURFACE edges")


def test_chair_query_returns_empty_not_defer() -> None:
    scene = _scene()
    router = _router()
    cr = RulesCompiler().compile("what is on the chair?", scene)
    if cr.outcome != "compiled":
        raise AssertionError(f"chair must compile in P6; got {cr.outcome}")
    ans = router.answer("what is on the chair?", scene, _oracle_ctx())
    if ans.outcome != "empty":
        raise AssertionError(f"chair must be true empty, got {ans.outcome}: {ans.text}")
    if ans.cited_uids or ans.cited_edges:
        raise AssertionError("empty answer must cite nothing")


def test_unknown_support_class_still_defers() -> None:
    cr = RulesCompiler().compile("what is on the cabinet?", _scene())
    if cr.outcome != "out_of_schema" or not cr.notes.startswith("deferred:"):
        raise AssertionError(f"cabinet should defer; got {cr.outcome} ({cr.notes})")


def test_real_replica_table_answer_and_chair_empty() -> None:
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
        [ExtractorRun(OnEntitySurfaceExtractor(), OnEntitySurfaceConfig())],
        density_policy="phase2_telemetry_only",
    )
    router = _router()
    table = router.answer("what is on the table?", bundle, _oracle_ctx())
    if table.outcome != "bindings":
        raise AssertionError(f"expected table bindings; got {table.outcome}")
    expected = ["obj_12", "obj_59", "obj_87", "obj_90", "obj_92"]
    if sorted(table.cited_uids) != expected:
        raise AssertionError(f"table answer drifted: {sorted(table.cited_uids)}")
    if "obj_43" in table.cited_uids or "obj_55" in table.cited_uids:
        raise AssertionError("table answer included pot or plant-stand boundary case")

    chair = router.answer("what is on the chair?", bundle, _oracle_ctx())
    if chair.outcome != "empty" or chair.cited_uids:
        raise AssertionError(f"chair should be true empty; got {chair.outcome}")


TESTS = [
    test_table_query_compiles_to_entity_class_ref,
    test_table_query_returns_entity_surface_bindings,
    test_chair_query_returns_empty_not_defer,
    test_unknown_support_class_still_defers,
    test_real_replica_table_answer_and_chair_empty,
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

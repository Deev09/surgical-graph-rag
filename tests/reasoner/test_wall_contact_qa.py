"""P5.03 tests: wall-contact QA over the SurfaceRef-anchored stored helper.

Proves the reasoner glue: "against the wall?" -> CONTACTS_SURFACE,
"near the wall?" -> NEAR_SURFACE, and "attached to the wall?" -> ATTACHED_TO
all compile to RELATION(?x, SurfaceRef) and execute through ONE
parameterized stored-edge branch; the Phase6RulesCompiler freeze still defers
attachment; "near the lamp" still falls through to the entity NEAR path;
malformed surface-relation shapes return execution_error. Real Replica:
"against the wall?" includes obj_6 (lamp) and excludes obj_63/obj_84/obj_17.

Run: python tests/reasoner/test_wall_contact_qa.py
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
from graph.relations.attached_to import AttachedToConfig, AttachedToExtractor
from graph.relations.contacts_surface import (
    ContactsSurfaceConfig, ContactsSurfaceExtractor,
)
from graph.relations.surface import SurfaceProximityConfig, SurfaceProximityExtractor
from graph.schema import Edge, GraphRef, Node, Plane, SceneGraphBundle, SurfaceRecord
from graph.serde import CURRENT_SCHEMA_VERSION
from reasoner.ast import Aggregation, EdgeConstraint, EntityRef, SurfaceRef, Variable
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import Phase6RulesCompiler, RulesCompiler
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
        id=uid, label=label, label_confidence=1.0, centroid=(0.0, 0.0, 0.0),
        bbox_aabb=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), bbox_obb=None,
        embedding_ref=None, attributes={"display_label": label}, provenance={},
    )


def _surf_edge(eid, entity_uid, surface_uid, etype) -> Edge:
    return Edge(
        edge_id=eid, source=GraphRef(kind="entity", uid=entity_uid),
        type=etype, target=GraphRef(kind="surface", uid=surface_uid),
        frame="world", weight=1.0, confidence=1.0,
        extractor="x", extractor_version="0.1", evidence={},
    )


def _wall_surface(uid="wall_1") -> SurfaceRecord:
    return SurfaceRecord(
        uid=uid, surface_type="wall", plane=Plane(a=0.0, b=-1.0, c=0.0, d=2.0),
        polygon=[(0.0, 2.0, 0.0), (2.0, 2.0, 0.0), (2.0, 2.0, 2.0), (0.0, 2.0, 2.0)],
        source="habitat_label", confidence=1.0,
    )


def _bundle(nodes, edges, surfaces) -> SceneGraphBundle:
    return SceneGraphBundle(
        schema_version=CURRENT_SCHEMA_VERSION, bundle_hash="h", scene_id="t",
        frame=_frame(), entity_bundle_hash="e", nodes=list(nodes),
        edges=list(edges), structural_surface_refs=[s.uid for s in surfaces],
        structural_surfaces=list(surfaces),
    )


def _wall_scene() -> SceneGraphBundle:
    return _bundle(
        nodes=[_node("obj_6", "lamp"), _node("obj_2", "cabinet")],
        edges=[
            _surf_edge("c1", "obj_6", "wall_1", "CONTACTS_SURFACE"),
            _surf_edge("n1", "obj_2", "wall_1", "NEAR_SURFACE"),
        ],
        surfaces=[_wall_surface()],
    )


def _attached_scene() -> SceneGraphBundle:
    return _bundle(
        nodes=[_node("obj_8", "wall sconce")],
        edges=[_surf_edge("a1", "obj_8", "wall_1", "ATTACHED_TO")],
        surfaces=[_wall_surface()],
    )


def _replica_v2_present() -> bool:
    return (REPLICA_V2_DIR / "scene_graph.json").exists()


def _router() -> Router:
    return Router(
        compiler=RulesCompiler(), executor=RulesExecutor(),
        verbalizer=StandardVerbalizer(),
    )


# --- compile shapes ------------------------------------------------------


def test_against_wall_compiles_to_contacts_surface() -> None:
    cr = RulesCompiler().compile("what is against the wall?", _wall_scene())
    if cr.outcome != "compiled" or cr.ast is None:
        raise AssertionError(f"must compile; got {cr.outcome}")
    c = cr.ast.where[0]
    if not isinstance(c, EdgeConstraint) or c.type != "CONTACTS_SURFACE":
        raise AssertionError(f"expected CONTACTS_SURFACE; got {c}")
    if not isinstance(c.source, Variable) or not isinstance(c.target, SurfaceRef):
        raise AssertionError("expected Variable -> SurfaceRef shape")
    if c.target.surface_type != "wall":
        raise AssertionError("expected SurfaceRef(wall)")


def test_near_wall_compiles_to_near_surface() -> None:
    cr = RulesCompiler().compile("what is near the wall?", _wall_scene())
    if cr.outcome != "compiled" or cr.ast is None:
        raise AssertionError(f"must compile; got {cr.outcome}")
    c = cr.ast.where[0]
    if c.type != "NEAR_SURFACE":
        raise AssertionError(f"expected NEAR_SURFACE; got {c.type}")
    if not isinstance(c.source, Variable) or not isinstance(c.target, SurfaceRef):
        raise AssertionError("expected Variable -> SurfaceRef shape")
    if c.target.surface_type != "wall":
        raise AssertionError("expected SurfaceRef(wall)")


def test_attached_to_wall_compiles_to_attached_to() -> None:
    cr = RulesCompiler().compile("what is attached to the wall?", _wall_scene())
    if cr.outcome != "compiled" or cr.ast is None:
        raise AssertionError(f"must compile; got {cr.outcome}")
    c = cr.ast.where[0]
    if c.type != "ATTACHED_TO":
        raise AssertionError(f"expected ATTACHED_TO; got {c.type}")
    if not isinstance(c.source, Variable) or not isinstance(c.target, SurfaceRef):
        raise AssertionError("expected Variable -> SurfaceRef shape")
    if c.target.surface_type != "wall":
        raise AssertionError("expected SurfaceRef(wall)")


def test_phase6_compiler_freeze_keeps_attached_to_wall_deferred() -> None:
    cr = Phase6RulesCompiler().compile("what is attached to the wall?", _wall_scene())
    if cr.outcome != "out_of_schema" or not cr.notes.startswith("deferred:"):
        raise AssertionError(f"P6 attached must defer; got {cr.outcome} ({cr.notes})")
    if "ATTACHED_TO" not in cr.notes:
        raise AssertionError("deferral note should mention ATTACHED_TO")


def test_near_entity_still_falls_through_to_entity_near() -> None:
    """"near the lamp" must NOT be captured by the surface path; it stays on
    the generic NEAR(?x, EntityRef) entity pattern."""
    cr = RulesCompiler().compile("what is near the lamp?", _wall_scene())
    if cr.outcome != "compiled":
        raise AssertionError(f"near-entity must compile; got {cr.outcome}")
    c = cr.ast.where[0]
    if c.type != "NEAR" or not isinstance(c.target, EntityRef):
        raise AssertionError(f"expected NEAR(?x, EntityRef); got {c}")


# --- execute: synthetic bindings -----------------------------------------


def test_contact_query_binds_from_contacts_surface() -> None:
    scene = _wall_scene()
    cr = RulesCompiler().compile("what is against the wall?", scene)
    er = RulesExecutor().execute(cr.ast, scene, _oracle_ctx())
    if er.outcome != "bindings":
        raise AssertionError(f"expected bindings; got {er.outcome}")
    if sorted(b["x"].uid for b in er.bindings) != ["obj_6"]:
        raise AssertionError(f"expected obj_6; got {er.bindings}")
    if [e.edge_id for e in er.evidence] != ["c1"]:
        raise AssertionError("must cite the stored CONTACTS_SURFACE edge")
    if any(e.type != "CONTACTS_SURFACE" for e in er.evidence):
        raise AssertionError("evidence must be CONTACTS_SURFACE edges")


def test_near_query_binds_from_near_surface() -> None:
    scene = _wall_scene()
    cr = RulesCompiler().compile("what is near the wall?", scene)
    er = RulesExecutor().execute(cr.ast, scene, _oracle_ctx())
    if er.outcome != "bindings":
        raise AssertionError(f"expected bindings; got {er.outcome}")
    if sorted(b["x"].uid for b in er.bindings) != ["obj_2"]:
        raise AssertionError(f"expected obj_2; got {er.bindings}")
    if any(e.type != "NEAR_SURFACE" for e in er.evidence):
        raise AssertionError("evidence must be NEAR_SURFACE edges")


def test_attached_query_binds_from_attached_to() -> None:
    scene = _attached_scene()
    cr = RulesCompiler().compile("what is attached to the wall?", scene)
    er = RulesExecutor().execute(cr.ast, scene, _oracle_ctx())
    if er.outcome != "bindings":
        raise AssertionError(f"expected bindings; got {er.outcome}")
    if sorted(b["x"].uid for b in er.bindings) != ["obj_8"]:
        raise AssertionError(f"expected obj_8; got {er.bindings}")
    if [e.edge_id for e in er.evidence] != ["a1"]:
        raise AssertionError("must cite the stored ATTACHED_TO edge")
    if any(e.type != "ATTACHED_TO" for e in er.evidence):
        raise AssertionError("evidence must be ATTACHED_TO edges")


def test_no_wall_surface_yields_empty_not_bindings() -> None:
    scene = _bundle(nodes=[_node("obj_6", "lamp")], edges=[], surfaces=[])
    cr = RulesCompiler().compile("what is against the wall?", scene)
    er = RulesExecutor().execute(cr.ast, scene, _oracle_ctx())
    if er.outcome not in ("empty", "unknown") or er.bindings:
        raise AssertionError(f"no wall -> empty/unknown; got {er.outcome}")


def test_malformed_surface_relation_shape_execution_error() -> None:
    """A surface relation with the wrong operand shape returns a clear
    surface-specific execution_error (not the generic EntityRef error)."""
    scene = _wall_scene()
    bad = Aggregation(
        op="ENUMERATE", bind=Variable("x"),
        where=[EdgeConstraint(
            source=SurfaceRef(surface_type="wall"),
            type="CONTACTS_SURFACE", target=Variable("x"),
        )],
    )
    er = RulesExecutor().execute(bad, scene, _oracle_ctx())
    if er.outcome != "execution_error":
        raise AssertionError(f"expected execution_error; got {er.outcome}")
    if "surface relation" not in er.notes.lower():
        raise AssertionError(f"note should be surface-specific; got {er.notes!r}")


# --- real Replica end-to-end ---------------------------------------------


def _build_replica_wall_bundle() -> SceneGraphBundle:
    capture = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    representation = MeshRepresentation(
        bundle=OracleReplicaAdapter().reconstruct(
            capture, ReconstructionConfig(name="oracle_replica", version="0.1", params={}),
        ),
    )
    artifacts = OracleReplicaExtractor(enriched_v2_path=REPLICA_V2_DIR).extract(
        representation,
        InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )
    # mixed graph: NEAR_SURFACE + CONTACTS_SURFACE
    bundle, _diag = build_graph(
        artifacts,
        [
            ExtractorRun(SurfaceProximityExtractor(), SurfaceProximityConfig()),
            ExtractorRun(ContactsSurfaceExtractor(), ContactsSurfaceConfig()),
        ],
        density_policy="phase2_telemetry_only",
    )
    return bundle


def _build_replica_attached_bundle() -> SceneGraphBundle:
    capture = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    representation = MeshRepresentation(
        bundle=OracleReplicaAdapter().reconstruct(
            capture, ReconstructionConfig(name="oracle_replica", version="0.1", params={}),
        ),
    )
    artifacts = OracleReplicaExtractor(enriched_v2_path=REPLICA_V2_DIR).extract(
        representation,
        InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )
    bundle, _diag = build_graph(
        artifacts,
        [ExtractorRun(AttachedToExtractor(), AttachedToConfig())],
        density_policy="phase2_telemetry_only",
    )
    return bundle


def test_real_against_wall_includes_lamp_excludes_negatives() -> None:
    if not _replica_v2_present():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    bundle = _build_replica_wall_bundle()
    ans = _router().answer("what is against the wall?", bundle, _oracle_ctx())
    if ans.outcome != "bindings":
        raise AssertionError(f"expected bindings; got {ans.outcome}")
    if "obj_6" not in ans.cited_uids:
        raise AssertionError(f"obj_6 (lamp) must be in the answer; got {ans.cited_uids}")
    for neg in ("obj_63", "obj_84", "obj_17"):
        if neg in ans.cited_uids:
            raise AssertionError(
                f"{neg} (picture/pillar/window) must NOT be a wall contact"
            )
    # all citations are CONTACTS_SURFACE edges
    cs_ids = {e.edge_id for e in bundle.edges if e.type == "CONTACTS_SURFACE"}
    if any(eid not in cs_ids for eid in ans.cited_edges):
        raise AssertionError("citations must all be CONTACTS_SURFACE edges")


def test_real_phase6_freeze_attached_to_wall_defers() -> None:
    if not _replica_v2_present():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    bundle = _build_replica_wall_bundle()
    router = Router(
        compiler=Phase6RulesCompiler(), executor=RulesExecutor(),
        verbalizer=StandardVerbalizer(),
    )
    ans = router.answer("what is attached to the wall?", bundle, _oracle_ctx())
    if ans.cited_uids or ans.cited_edges:
        raise AssertionError("attached-to-wall must cite nothing (deferred)")
    if "can't answer that yet" not in ans.text:
        raise AssertionError(f"deferred text expected; got {ans.text!r}")


def test_real_room0_attached_to_wall_empty_with_p7_extractor() -> None:
    if not _replica_v2_present():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    bundle = _build_replica_attached_bundle()
    ans = _router().answer("what is attached to the wall?", bundle, _oracle_ctx())
    if ans.outcome != "empty":
        raise AssertionError(f"room_0 has no honest ATTACHED_TO edges; got {ans.outcome}")
    if ans.cited_uids or ans.cited_edges:
        raise AssertionError("empty room_0 attachment answer must cite nothing")


TESTS = [
    test_against_wall_compiles_to_contacts_surface,
    test_near_wall_compiles_to_near_surface,
    test_attached_to_wall_compiles_to_attached_to,
    test_phase6_compiler_freeze_keeps_attached_to_wall_deferred,
    test_near_entity_still_falls_through_to_entity_near,
    test_contact_query_binds_from_contacts_surface,
    test_near_query_binds_from_near_surface,
    test_attached_query_binds_from_attached_to,
    test_no_wall_surface_yields_empty_not_bindings,
    test_malformed_surface_relation_shape_execution_error,
    test_real_against_wall_includes_lamp_excludes_negatives,
    test_real_phase6_freeze_attached_to_wall_defers,
    test_real_room0_attached_to_wall_empty_with_p7_extractor,
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

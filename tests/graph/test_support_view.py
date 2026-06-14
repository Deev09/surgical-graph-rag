"""P4.03 tests: SUPPORTS derived view (graph/views/support.py).

Proves SUPPORTS is a read-side projection, never a stored edge:
  - one SupportFact per ON_SURFACE edge (clean inverse, no role filter);
  - direction inverted (supporter=surface, supported=entity);
  - derived_from_edge_id references a real ON_SURFACE edge;
  - no evidence dict copied into the view (SupportFact has no evidence field);
  - SupportFact is not an Edge and is immutable;
  - the view is strict: a materialized SUPPORTS edge raises;
  - real Replica clean-inverse via a single-extractor build_graph run.

Run: python tests/graph/test_support_view.py
"""
from __future__ import annotations

import sys
import traceback
from dataclasses import FrozenInstanceError
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
from graph.relations.on_surface import OnSurfaceConfig, OnSurfaceExtractor
from graph.schema import Edge, Node, SceneGraphBundle, SurfaceRecord, GraphRef
from graph.serde import CURRENT_SCHEMA_VERSION
from graph.views.support import SupportFact, entity_support_facts, support_facts
from representations.mesh import MeshRepresentation


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
REPLICA_V2_DIR = REPLICA_SCENE_DIR / "enriched" / "v2"


def _frame() -> SceneFrame:
    return SceneFrame(
        gravity=(0.0, 0.0, -1.0), canonical_forward=None,
        canonical_right=None, units="meters", notes="",
    )


def _edge(edge_id, src_kind, src_uid, etype, tgt_kind, tgt_uid, evidence=None) -> Edge:
    return Edge(
        edge_id=edge_id,
        source=GraphRef(kind=src_kind, uid=src_uid),
        type=etype,
        target=GraphRef(kind=tgt_kind, uid=tgt_uid),
        frame="world", weight=1.0, confidence=1.0,
        extractor="test", extractor_version="0.1",
        evidence=evidence or {},
    )


def _bundle(edges) -> SceneGraphBundle:
    return SceneGraphBundle(
        schema_version=CURRENT_SCHEMA_VERSION,
        bundle_hash="h", scene_id="t", frame=_frame(),
        entity_bundle_hash="e", nodes=[], edges=list(edges),
        structural_surface_refs=[], structural_surfaces=[],
    )


def _on_surface_edge(edge_id, entity_uid, surface_uid, evidence=None) -> Edge:
    # ON_SURFACE direction: source=entity, target=surface
    return _edge(
        edge_id, "entity", entity_uid, "ON_SURFACE", "surface", surface_uid,
        evidence=evidence or {"bottom_gap_m": -0.01, "contact": True},
    )


def _on_entity_surface_edge(edge_id, supported_uid, supporter_uid, evidence=None) -> Edge:
    return _edge(
        edge_id,
        "entity",
        supported_uid,
        "ON_ENTITY_SURFACE",
        "entity",
        supporter_uid,
        evidence=evidence or {
            "owner_entity_uid": supporter_uid,
            "entity_surface_uid": f"ent_surf_{supporter_uid}_top",
            "contact": True,
        },
    )


def _replica_v2_present() -> bool:
    return (REPLICA_V2_DIR / "scene_graph.json").exists()


# --- clean inverse: count + direction ------------------------------------


def test_one_fact_per_on_surface_edge() -> None:
    edges = [
        _on_surface_edge("e1", "obj_1", "floor_1"),
        _on_surface_edge("e2", "obj_2", "floor_1"),
        _on_surface_edge("e3", "obj_3", "table_1"),
        # non-ON_SURFACE edges must be ignored
        _edge("n1", "entity", "obj_1", "NEAR", "entity", "obj_2"),
        _edge("n2", "entity", "obj_1", "NEAR_SURFACE", "surface", "floor_1"),
    ]
    facts = support_facts(_bundle(edges))
    on_count = sum(1 for e in edges if e.type == "ON_SURFACE")
    if len(facts) != on_count:
        raise AssertionError(
            f"SUPPORTS_count {len(facts)} != ON_SURFACE_count {on_count}"
        )


def test_direction_inverted_surface_supports_entity() -> None:
    facts = support_facts(_bundle([_on_surface_edge("e1", "obj_1", "floor_1")]))
    f = facts[0]
    if f.supporter.kind != "surface" or f.supporter.uid != "floor_1":
        raise AssertionError(f"supporter must be the surface; got {f.supporter}")
    if f.supported.kind != "entity" or f.supported.uid != "obj_1":
        raise AssertionError(f"supported must be the entity; got {f.supported}")
    if f.relation != "SUPPORTS":
        raise AssertionError(f"relation must be 'SUPPORTS'; got {f.relation!r}")


def test_empty_bundle_yields_no_facts() -> None:
    if support_facts(_bundle([])) != []:
        raise AssertionError("empty bundle must yield no support facts")


# --- evidence by reference, no copy --------------------------------------


def test_derived_from_edge_id_points_to_on_surface_edge() -> None:
    edges = [
        _on_surface_edge("e1", "obj_1", "floor_1"),
        _on_surface_edge("e2", "obj_2", "table_1"),
    ]
    bundle = _bundle(edges)
    on_ids = {e.edge_id for e in bundle.edges if e.type == "ON_SURFACE"}
    for f in support_facts(bundle):
        if f.derived_from_edge_id not in on_ids:
            raise AssertionError(
                f"derived_from_edge_id {f.derived_from_edge_id!r} is not an "
                "ON_SURFACE edge id"
            )


def test_no_evidence_copied_into_view() -> None:
    """The view must reference evidence by edge id, not copy the dict."""
    edge = _on_surface_edge("e1", "obj_1", "floor_1", evidence={"secret": 123})
    f = support_facts(_bundle([edge]))[0]
    if hasattr(f, "evidence"):
        raise AssertionError("SupportFact must not carry an evidence field")
    # the only link to evidence is the edge id
    if f.derived_from_edge_id != "e1":
        raise AssertionError("derived_from_edge_id must reference the source edge")


# --- SupportFact is not an Edge, and is immutable ------------------------


def test_support_fact_is_not_an_edge() -> None:
    f = support_facts(_bundle([_on_surface_edge("e1", "obj_1", "floor_1")]))[0]
    if isinstance(f, Edge):
        raise AssertionError("SupportFact must not be an Edge instance")


def test_support_fact_is_frozen() -> None:
    f = support_facts(_bundle([_on_surface_edge("e1", "obj_1", "floor_1")]))[0]
    try:
        f.supporter = GraphRef(kind="entity", uid="x")  # type: ignore[misc]
    except FrozenInstanceError:
        return
    raise AssertionError("SupportFact must be immutable (frozen)")


# --- strict: materialized SUPPORTS edge is a contract violation ----------


def test_materialized_supports_edge_raises() -> None:
    edges = [
        _on_surface_edge("e1", "obj_1", "floor_1"),
        _edge("s1", "surface", "floor_1", "SUPPORTS", "entity", "obj_1"),
    ]
    try:
        support_facts(_bundle(edges))
    except ValueError:
        return
    raise AssertionError(
        "support_facts must raise on a materialized SUPPORTS edge"
    )


def test_malformed_on_surface_endpoints_raise() -> None:
    """ON_SURFACE is defined entity -> surface. A role-swapped edge
    (surface -> entity) must raise rather than project a bad SupportFact."""
    swapped = _edge(
        "bad1", "surface", "floor_1", "ON_SURFACE", "entity", "obj_1",
    )
    try:
        support_facts(_bundle([swapped]))
    except ValueError:
        pass
    else:
        raise AssertionError("swapped-endpoint ON_SURFACE edge must raise")

    # also reject entity -> entity and surface -> surface ON_SURFACE edges
    ee = _edge("bad2", "entity", "obj_1", "ON_SURFACE", "entity", "obj_2")
    try:
        support_facts(_bundle([ee]))
    except ValueError:
        pass
    else:
        raise AssertionError("entity->entity ON_SURFACE edge must raise")

    ss = _edge("bad3", "surface", "floor_1", "ON_SURFACE", "surface", "floor_2")
    try:
        support_facts(_bundle([ss]))
    except ValueError:
        return
    raise AssertionError("surface->surface ON_SURFACE edge must raise")


def test_no_materialized_supports_in_clean_bundle() -> None:
    edges = [_on_surface_edge("e1", "obj_1", "floor_1")]
    bundle = _bundle(edges)
    if any(e.type == "SUPPORTS" for e in bundle.edges):
        raise AssertionError("clean bundle should contain no SUPPORTS edges")
    # view works on a clean bundle
    if len(support_facts(bundle)) != 1:
        raise AssertionError("expected 1 support fact from clean bundle")


# --- entity support view (P6) -------------------------------------------


def test_entity_support_fact_direction() -> None:
    facts = entity_support_facts(_bundle([
        _on_entity_surface_edge("oes1", "book_1", "table_1"),
    ]))
    if len(facts) != 1:
        raise AssertionError(f"expected one entity support fact; got {len(facts)}")
    f = facts[0]
    if f.supporter.kind != "entity" or f.supporter.uid != "table_1":
        raise AssertionError(f"supporter must be owner entity; got {f.supporter}")
    if f.supported.kind != "entity" or f.supported.uid != "book_1":
        raise AssertionError(f"supported must be resting entity; got {f.supported}")
    if f.derived_from_edge_id != "oes1":
        raise AssertionError("derived_from_edge_id must cite ON_ENTITY_SURFACE edge")


def test_entity_support_rejects_materialized_supports() -> None:
    edges = [
        _on_entity_surface_edge("oes1", "book_1", "table_1"),
        _edge("s1", "entity", "table_1", "SUPPORTS", "entity", "book_1"),
    ]
    try:
        entity_support_facts(_bundle(edges))
    except ValueError:
        return
    raise AssertionError("entity_support_facts must raise on materialized SUPPORTS")


def test_malformed_on_entity_surface_endpoints_raise() -> None:
    bad = _edge("bad1", "entity", "book_1", "ON_ENTITY_SURFACE", "surface", "top_1")
    try:
        entity_support_facts(_bundle([bad]))
    except ValueError:
        pass
    else:
        raise AssertionError("surface-target ON_ENTITY_SURFACE must raise")

    bad_owner = _on_entity_surface_edge(
        "bad2", "book_1", "table_1",
        evidence={"owner_entity_uid": "other_table"},
    )
    try:
        entity_support_facts(_bundle([bad_owner]))
    except ValueError:
        return
    raise AssertionError("owner_entity_uid mismatch must raise")


# --- real Replica clean-inverse ------------------------------------------


def test_real_replica_clean_inverse() -> None:
    """Build a real ON_SURFACE-only bundle on Replica room_0 and assert the
    derived view is a clean inverse: one SupportFact per ON_SURFACE edge,
    each referencing a real ON_SURFACE edge, no materialized SUPPORTS."""
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
    on_edges = [e for e in bundle.edges if e.type == "ON_SURFACE"]
    facts = support_facts(bundle)
    if len(facts) != len(on_edges):
        raise AssertionError(
            f"clean inverse violated: {len(facts)} facts != "
            f"{len(on_edges)} ON_SURFACE edges"
        )
    if not facts:
        raise AssertionError("expected at least one support fact on Replica")
    on_ids = {e.edge_id for e in on_edges}
    for f in facts:
        if f.derived_from_edge_id not in on_ids:
            raise AssertionError("support fact references a non-ON_SURFACE edge")
        if f.supporter.kind != "surface" or f.supported.kind != "entity":
            raise AssertionError("support fact direction wrong on real data")
    if any(e.type == "SUPPORTS" for e in bundle.edges):
        raise AssertionError("real bundle must contain zero materialized SUPPORTS")


TESTS = [
    test_one_fact_per_on_surface_edge,
    test_direction_inverted_surface_supports_entity,
    test_empty_bundle_yields_no_facts,
    test_derived_from_edge_id_points_to_on_surface_edge,
    test_no_evidence_copied_into_view,
    test_support_fact_is_not_an_edge,
    test_support_fact_is_frozen,
    test_materialized_supports_edge_raises,
    test_malformed_on_surface_endpoints_raise,
    test_no_materialized_supports_in_clean_bundle,
    test_entity_support_fact_direction,
    test_entity_support_rejects_materialized_supports,
    test_malformed_on_entity_surface_endpoints_raise,
    test_real_replica_clean_inverse,
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

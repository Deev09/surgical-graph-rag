"""Round-trip serde tests for every frozen-bundle dataclass.

Run: python tests/schema/test_round_trip.py

Stdlib only (no pytest dependency). Uses array-aware equality so
np.ndarray fields are compared by dtype + shape + values.
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from common.equality import array_aware_equal
from common.serde import SchemaVersionError
from common.types import CameraPose, OrientedBBox, Plane, SceneFrame
from representations.base import (
    GeometryHandle, ReconstructionDiagnostics, RepresentationCapabilities,
    SceneRepresentationBundle,
)
from representations.serde import (
    CURRENT_SCHEMA_VERSION as REPR_SCHEMA_VERSION,
    dump_scene_repr_bundle, load_scene_repr_bundle,
)
from extractors.base import (
    BundleCorrespondence, EntityArtifact, EntityArtifacts, EntityIdentity,
    ExtractionDiagnostics, SemanticHypothesis, StructuralSurface,
)
from extractors.serde import (
    CURRENT_SCHEMA_VERSION as ENT_SCHEMA_VERSION,
    dump_bundle_correspondence, dump_entity_artifacts,
    load_bundle_correspondence, load_entity_artifacts,
)
from graph.schema import (
    BuildDiagnostics, Edge, EdgeRejection, GraphRef, Node, SceneGraphBundle,
    SurfaceRecord,
)
from graph.serde import (
    CURRENT_SCHEMA_VERSION as GRAPH_SCHEMA_VERSION,
    dump_build_diagnostics, dump_scene_graph_bundle,
    load_build_diagnostics, load_scene_graph_bundle,
)
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.serde import (
    dump_completeness_profile, dump_execution_context,
    load_completeness_profile, load_execution_context,
)


def make_scene_frame() -> SceneFrame:
    return SceneFrame(
        gravity=(0.0, 0.0, -1.0),
        canonical_forward=(0.0, 1.0, 0.0),
        canonical_right=(1.0, 0.0, 0.0),
        units="meters",
        notes="test",
    )


def make_repr_bundle() -> SceneRepresentationBundle:
    return SceneRepresentationBundle(
        schema_version=REPR_SCHEMA_VERSION,
        representation_hash="r_abc",
        scene_id="scene_test",
        frame=make_scene_frame(),
        capabilities=RepresentationCapabilities(
            renderable_channels=frozenset(["rgb", "depth"]),
            supports_arbitrary_pose=True,
            deterministic=True,
            typical_render_ms=50,
        ),
        geometry_handle=GeometryHandle(
            kind="mesh_file", uri="meshes/test.ply", notes={"src": "test"},
        ),
        poses=[
            CameraPose(
                camera_id="cam0",
                position=(1.0, 2.0, 3.0),
                rotation_quat=(0.0, 0.0, 0.0, 1.0),
                intrinsics=(500.0, 500.0, 320.0, 240.0),
                width=640,
                height=480,
            )
        ],
        diagnostics=ReconstructionDiagnostics(
            loss=0.01, coverage=0.95, pose_rmse=0.02,
            runtime_seconds=12.3, notes="ok",
        ),
        notes={"k": "v"},
    )


def make_entity_artifacts() -> EntityArtifacts:
    return EntityArtifacts(
        schema_version=ENT_SCHEMA_VERSION,
        bundle_hash="e_abc",
        scene_id="scene_test",
        frame=make_scene_frame(),
        representation_hash="r_abc",
        extractor_name="oracle_test",
        extractor_version="0.1",
        entities=[
            EntityArtifact(
                identity=EntityIdentity(
                    object_uid="obj_1",
                    display_label="chair_1",
                    aliases=["chair"],
                    source_instance_ref="42",
                ),
                bbox_aabb=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                bbox_obb=OrientedBBox(
                    center=(0.5, 0.5, 0.5),
                    extents=(0.5, 0.5, 0.5),
                    rotation_quat=(0.0, 0.0, 0.0, 1.0),
                ),
                centroid=(0.5, 0.5, 0.5),
                geometry_handle=None,
                semantic_hypotheses=[
                    SemanticHypothesis(label="chair", confidence=1.0, source="test"),
                ],
                embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
                extraction_diagnostics={"coverage": 1.0},
            ),
            EntityArtifact(
                identity=EntityIdentity(
                    object_uid="obj_2",
                    display_label="table",
                    aliases=[],
                    source_instance_ref="43",
                ),
                bbox_aabb=((2.0, 0.0, 0.0), (3.0, 1.0, 0.7)),
                bbox_obb=None,
                centroid=(2.5, 0.5, 0.35),
                geometry_handle=None,
                semantic_hypotheses=[],
                embedding=None,
                extraction_diagnostics={},
            ),
        ],
        structural_surfaces=[
            StructuralSurface(
                surface_uid="surf_floor",
                surface_type="floor",
                plane=Plane(a=0.0, b=0.0, c=1.0, d=0.0),
                polygon=[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 5.0, 0.0), (0.0, 5.0, 0.0)],
                confidence=1.0,
                source="habitat_label",
            ),
            StructuralSurface(
                surface_uid="surf_wall_n",
                surface_type="wall",
                plane=Plane(a=0.0, b=1.0, c=0.0, d=-5.0),
                polygon=None,
                confidence=0.9,
                source="habitat_label",
            ),
        ],
        geometry_store_path=Path("geom"),
        diagnostics=ExtractionDiagnostics(
            n_entities=2,
            n_structural_surfaces=2,
            runtime_seconds=1.5,
            coverage_score=0.98,
            notes="test",
        ),
        notes={"src": "fixture"},
    )


def make_bundle_correspondence() -> BundleCorrespondence:
    return BundleCorrespondence(
        source_bundle_hash="src",
        target_bundle_hash="tgt",
        entity_pairs=[("obj_1", "obj_a"), ("obj_2", "obj_b")],
        surface_pairs=[("surf_floor", "surf_f1")],
        method="iou_match",
        score={"entity:obj_1->obj_a": 0.95, "surface:surf_floor->surf_f1": 1.0},
        unmatched_source_entities=["obj_3"],
        unmatched_target_entities=[],
        unmatched_source_surfaces=[],
        unmatched_target_surfaces=["surf_extra"],
    )


def make_scene_graph_bundle() -> SceneGraphBundle:
    return SceneGraphBundle(
        schema_version=GRAPH_SCHEMA_VERSION,
        bundle_hash="g_abc",
        scene_id="scene_test",
        frame=make_scene_frame(),
        entity_bundle_hash="e_abc",
        nodes=[
            Node(
                id="obj_1",
                label="chair",
                label_confidence=1.0,
                centroid=(0.5, 0.5, 0.5),
                bbox_aabb=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                bbox_obb=None,
                embedding_ref=None,
                attributes={"color": "red"},
                provenance={"extractor": "oracle"},
            ),
        ],
        edges=[
            Edge(
                edge_id="e1",
                source=GraphRef(kind="entity", uid="obj_1"),
                type="ABOVE",
                target=GraphRef(kind="surface", uid="surf_floor"),
                frame="world",
                weight=1.0,
                confidence=1.0,
                extractor="directional",
                extractor_version="0.1",
                evidence={"dz": 0.5},
            ),
        ],
        structural_surface_refs=["surf_floor"],
        structural_surfaces=[
            SurfaceRecord(
                uid="surf_floor",
                surface_type="floor",
                plane=Plane(a=0.0, b=0.0, c=1.0, d=0.0),
                polygon=[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 5.0, 0.0), (0.0, 5.0, 0.0)],
                source="habitat_label",
                confidence=1.0,
            ),
        ],
    )


def make_build_diagnostics() -> BuildDiagnostics:
    from graph.relations.base import RelationExtractorDiagnostics
    return BuildDiagnostics(
        extractor_versions={"directional": "0.1", "proximity": "0.1"},
        edges_emitted_per_type={"ABOVE": 1, "NEAR": 0},
        rejections_per_type={"BELOW": 2},
        rejection_samples=[
            EdgeRejection(
                source=GraphRef(kind="entity", uid="obj_1"),
                type="BELOW",
                target=GraphRef(kind="entity", uid="obj_2"),
                extractor="directional",
                rejected_reason="dominant_axis_not_z",
                evidence={"dz": 0.1, "dx": 0.5},
            ),
        ],
        runtime_ms_per_extractor={"directional": 4, "proximity": 1},
        per_extractor=[
            RelationExtractorDiagnostics(
                extractor="directional",
                version="0.1",
                mode="sparse",
                physical_edges_per_type={"ABOVE": 1},
                physical_edges_total=1,
                logical_edges_total=1,
                rejections_per_type={"BELOW": 2},
                rejection_samples=[],
                runtime_ms=4,
            ),
        ],
        physical_edges_total=1,
        logical_edges_total=1,
        mode="sparse",
    )


def make_completeness_profile() -> CompletenessProfile:
    return CompletenessProfile(
        source="measured",
        entity_recall_by_class={"furniture": 0.92, "small": 0.41},
        edge_recall_by_type={"ABOVE": 0.85, "NEAR": 0.90},
        calibration_dataset="replica_room_0_v1",
    )


def make_execution_context() -> ExecutionContext:
    return ExecutionContext(
        completeness=make_completeness_profile(),
        empty_recall_threshold=0.95,
    )


def _assert_equal(original, loaded, name: str) -> None:
    if not array_aware_equal(original, loaded):
        raise AssertionError(f"{name}: round-trip mismatch")


def test_scene_repr_bundle_roundtrip() -> None:
    original = make_repr_bundle()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle"
        dump_scene_repr_bundle(original, out)
        loaded = load_scene_repr_bundle(out)
    _assert_equal(original, loaded, "SceneRepresentationBundle")


def test_entity_artifacts_roundtrip() -> None:
    original = make_entity_artifacts()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle"
        dump_entity_artifacts(original, out)
        loaded = load_entity_artifacts(out)
    _assert_equal(original, loaded, "EntityArtifacts")


def test_bundle_correspondence_roundtrip() -> None:
    original = make_bundle_correspondence()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "corr.json"
        dump_bundle_correspondence(original, out)
        loaded = load_bundle_correspondence(out)
    _assert_equal(original, loaded, "BundleCorrespondence")


def test_scene_graph_bundle_roundtrip() -> None:
    original = make_scene_graph_bundle()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle"
        dump_scene_graph_bundle(original, out)
        loaded = load_scene_graph_bundle(out)
    _assert_equal(original, loaded, "SceneGraphBundle")


def test_build_diagnostics_roundtrip() -> None:
    original = make_build_diagnostics()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "diag.json"
        dump_build_diagnostics(original, out)
        loaded = load_build_diagnostics(out)
    _assert_equal(original, loaded, "BuildDiagnostics")


def test_completeness_profile_roundtrip() -> None:
    original = make_completeness_profile()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "cp.json"
        dump_completeness_profile(original, out)
        loaded = load_completeness_profile(out)
    _assert_equal(original, loaded, "CompletenessProfile")


def test_execution_context_roundtrip() -> None:
    original = make_execution_context()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ctx.json"
        dump_execution_context(original, out)
        loaded = load_execution_context(out)
    _assert_equal(original, loaded, "ExecutionContext")


def test_schema_version_mismatch_raises_repr() -> None:
    original = make_repr_bundle()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle"
        dump_scene_repr_bundle(original, out)
        m = out / "manifest.json"
        payload = json.loads(m.read_text())
        payload["schema_version"] = 999
        m.write_text(json.dumps(payload))
        try:
            load_scene_repr_bundle(out)
        except SchemaVersionError:
            return
        raise AssertionError("expected SchemaVersionError")


def test_schema_version_mismatch_raises_entity() -> None:
    original = make_entity_artifacts()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle"
        dump_entity_artifacts(original, out)
        m = out / "manifest.json"
        payload = json.loads(m.read_text())
        payload["schema_version"] = 999
        m.write_text(json.dumps(payload))
        try:
            load_entity_artifacts(out)
        except SchemaVersionError:
            return
        raise AssertionError("expected SchemaVersionError")


def test_schema_version_mismatch_raises_graph() -> None:
    original = make_scene_graph_bundle()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle"
        dump_scene_graph_bundle(original, out)
        m = out / "manifest.json"
        payload = json.loads(m.read_text())
        payload["schema_version"] = 999
        m.write_text(json.dumps(payload))
        try:
            load_scene_graph_bundle(out)
        except SchemaVersionError:
            return
        raise AssertionError("expected SchemaVersionError")


def test_on_surface_edge_roundtrips_under_v3() -> None:
    """P4.02: ON_SURFACE is accepted by graph serde (EdgeType addition).
    A bundle carrying an ON_SURFACE edge round-trips under schema v3."""
    from dataclasses import replace
    base = make_scene_graph_bundle()
    on_edge = Edge(
        edge_id="e_on_surface_1",
        source=GraphRef(kind="entity", uid="obj_1"),
        type="ON_SURFACE",
        target=GraphRef(kind="surface", uid="surf_floor"),
        frame="world",
        weight=1.0,
        confidence=1.0,
        extractor="on_surface",
        extractor_version="0.1",
        evidence={"bottom_gap_m": -0.01, "contact": True, "polygon_clip_required": True},
    )
    original = replace(base, edges=list(base.edges) + [on_edge])
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle"
        dump_scene_graph_bundle(original, out)
        loaded = load_scene_graph_bundle(out)
    _assert_equal(original, loaded, "SceneGraphBundle(ON_SURFACE)")
    if not any(e.type == "ON_SURFACE" for e in loaded.edges):
        raise AssertionError("ON_SURFACE edge lost in round-trip")


def test_graph_schema_v2_rejected_under_v3() -> None:
    """P4.02 / D5: strict v3 loader rejects a manually-downgraded v2 graph
    manifest. No migration path; old graph bundles are not silently coerced."""
    original = make_scene_graph_bundle()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle"
        dump_scene_graph_bundle(original, out)
        m = out / "manifest.json"
        payload = json.loads(m.read_text())
        payload["schema_version"] = 2  # the pre-ON_SURFACE graph serde version
        m.write_text(json.dumps(payload))
        try:
            load_scene_graph_bundle(out)
        except SchemaVersionError:
            return
        raise AssertionError("expected SchemaVersionError for v2 under strict v3")


def test_contacts_surface_edge_roundtrips_under_v4() -> None:
    """P5.02: CONTACTS_SURFACE is accepted by graph serde (EdgeType addition).
    A bundle carrying a CONTACTS_SURFACE edge round-trips under schema v4."""
    from dataclasses import replace
    base = make_scene_graph_bundle()
    cs_edge = Edge(
        edge_id="e_contacts_surface_1",
        source=GraphRef(kind="entity", uid="obj_1"),
        type="CONTACTS_SURFACE",
        target=GraphRef(kind="surface", uid="surf_floor"),
        frame="world",
        weight=1.0,
        confidence=1.0,
        extractor="contacts_surface",
        extractor_version="0.1",
        evidence={"wall_gap_m": 0.01, "contact": True, "up": [0.0, 0.0, 1.0],
                  "polygon_clip_required": True},
    )
    original = replace(base, edges=list(base.edges) + [cs_edge])
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle"
        dump_scene_graph_bundle(original, out)
        loaded = load_scene_graph_bundle(out)
    _assert_equal(original, loaded, "SceneGraphBundle(CONTACTS_SURFACE)")
    if not any(e.type == "CONTACTS_SURFACE" for e in loaded.edges):
        raise AssertionError("CONTACTS_SURFACE edge lost in round-trip")


def test_on_entity_surface_edge_roundtrips_under_v5() -> None:
    """P6.02: ON_ENTITY_SURFACE is accepted by graph serde.
    The stored edge is entity -> entity; derived-top details live in evidence."""
    from dataclasses import replace
    base = make_scene_graph_bundle()
    node = Node(
        id="obj_2",
        label="table",
        label_confidence=1.0,
        centroid=(2.5, 0.5, 0.35),
        bbox_aabb=((2.0, 0.0, 0.0), (3.0, 1.0, 0.7)),
        bbox_obb=None,
        embedding_ref=None,
        attributes={"display_label": "table_1"},
        provenance={},
    )
    edge = Edge(
        edge_id="e_on_entity_surface_1",
        source=GraphRef(kind="entity", uid="obj_1"),
        type="ON_ENTITY_SURFACE",
        target=GraphRef(kind="entity", uid="obj_2"),
        frame="world",
        weight=1.0,
        confidence=1.0,
        extractor="on_entity_surface",
        extractor_version="0.1",
        evidence={
            "owner_entity_uid": "obj_2",
            "entity_surface_uid": "ent_surf_obj_2_top",
            "owner_class": "table",
            "bottom_gap_m": 0.0,
            "contact": True,
        },
    )
    original = replace(base, nodes=list(base.nodes) + [node], edges=list(base.edges) + [edge])
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle"
        dump_scene_graph_bundle(original, out)
        loaded = load_scene_graph_bundle(out)
    _assert_equal(original, loaded, "SceneGraphBundle(ON_ENTITY_SURFACE)")
    loaded_edge = next(e for e in loaded.edges if e.type == "ON_ENTITY_SURFACE")
    if loaded_edge.target.kind != "entity":
        raise AssertionError("ON_ENTITY_SURFACE target must remain entity ref")


def test_graph_schema_v3_rejected_under_v4() -> None:
    """P5.02 / D4: strict v4 loader rejects a manually-downgraded v3 graph
    manifest. No migration path; old v3 graph bundles are not silently
    coerced. (The earlier v2-rejection test still holds: any non-current
    version is rejected.)"""
    original = make_scene_graph_bundle()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle"
        dump_scene_graph_bundle(original, out)
        m = out / "manifest.json"
        payload = json.loads(m.read_text())
        payload["schema_version"] = 3  # the pre-CONTACTS_SURFACE graph serde version
        m.write_text(json.dumps(payload))
        try:
            load_scene_graph_bundle(out)
        except SchemaVersionError:
            return
        raise AssertionError("expected SchemaVersionError for v3 under strict v4")


def test_graph_schema_v4_rejected_under_v5() -> None:
    """P6.02: strict v5 loader rejects a manually-downgraded v4 graph."""
    original = make_scene_graph_bundle()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle"
        dump_scene_graph_bundle(original, out)
        m = out / "manifest.json"
        payload = json.loads(m.read_text())
        payload["schema_version"] = 4
        m.write_text(json.dumps(payload))
        try:
            load_scene_graph_bundle(out)
        except SchemaVersionError:
            return
        raise AssertionError("expected SchemaVersionError for v4 under strict v5")


def test_embedding_npy_sidecar_written_and_typed() -> None:
    original = make_entity_artifacts()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle"
        dump_entity_artifacts(original, out)
        npy_path = out / "embeddings" / "obj_1.npy"
        if not npy_path.exists():
            raise AssertionError(f"expected .npy sidecar at {npy_path}")
        loaded = np.load(npy_path)
        if loaded.dtype != np.float32:
            raise AssertionError(f"dtype mismatch: {loaded.dtype}")
        if loaded.shape != (3,):
            raise AssertionError(f"shape mismatch: {loaded.shape}")
        if not np.array_equal(loaded, np.array([0.1, 0.2, 0.3], dtype=np.float32)):
            raise AssertionError("values mismatch")


def test_array_aware_equality_dtype_sensitive() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    if array_aware_equal(a, b):
        raise AssertionError("expected dtype mismatch to fail equality")


def test_array_aware_equality_shape_sensitive() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0])
    if array_aware_equal(a, b):
        raise AssertionError("expected shape mismatch to fail equality")


def test_array_aware_equality_tuple_vs_list() -> None:
    """Schema correctness: a field typed as tuple must not equal the same
    values stored as list. Catches serde bugs that lose tuple typing."""
    if array_aware_equal((1, 2, 3), [1, 2, 3]):
        raise AssertionError("expected tuple-vs-list to fail equality")


def test_graph_ref_uses_correct_kind() -> None:
    """Loading an Edge whose target is a surface ref preserves kind='surface'."""
    original = make_scene_graph_bundle()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle"
        dump_scene_graph_bundle(original, out)
        loaded = load_scene_graph_bundle(out)
        edge = loaded.edges[0]
        if edge.target.kind != "surface":
            raise AssertionError(f"surface ref kind lost: {edge.target.kind!r}")
        if edge.target.uid != "surf_floor":
            raise AssertionError(f"surface ref uid lost: {edge.target.uid!r}")


def _mutate_graph_manifest_and_expect_value_error(mutate) -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle"
        dump_scene_graph_bundle(make_scene_graph_bundle(), out)
        manifest = out / "manifest.json"
        payload = json.loads(manifest.read_text())
        mutate(payload)
        manifest.write_text(json.dumps(payload))
        try:
            load_scene_graph_bundle(out)
        except ValueError:
            return
        raise AssertionError("expected ValueError for invalid SceneGraphBundle manifest")


def test_graph_load_rejects_mismatched_surface_refs() -> None:
    _mutate_graph_manifest_and_expect_value_error(
        lambda payload: payload.update({"structural_surface_refs": []}),
    )


def test_graph_load_rejects_duplicate_surface_records() -> None:
    def mutate(payload) -> None:
        payload["structural_surfaces"].append(dict(payload["structural_surfaces"][0]))
        payload["structural_surface_refs"].append("surf_floor")
    _mutate_graph_manifest_and_expect_value_error(mutate)


def test_graph_load_rejects_unknown_edge_surface_ref() -> None:
    def mutate(payload) -> None:
        payload["edges"][0]["target"]["uid"] = "ghost_surface"
    _mutate_graph_manifest_and_expect_value_error(mutate)


TESTS = [
    test_scene_repr_bundle_roundtrip,
    test_entity_artifacts_roundtrip,
    test_bundle_correspondence_roundtrip,
    test_scene_graph_bundle_roundtrip,
    test_build_diagnostics_roundtrip,
    test_completeness_profile_roundtrip,
    test_execution_context_roundtrip,
    test_schema_version_mismatch_raises_repr,
    test_schema_version_mismatch_raises_entity,
    test_schema_version_mismatch_raises_graph,
    test_on_surface_edge_roundtrips_under_v3,
    test_graph_schema_v2_rejected_under_v3,
    test_contacts_surface_edge_roundtrips_under_v4,
    test_graph_schema_v3_rejected_under_v4,
    test_on_entity_surface_edge_roundtrips_under_v5,
    test_graph_schema_v4_rejected_under_v5,
    test_embedding_npy_sidecar_written_and_typed,
    test_array_aware_equality_dtype_sensitive,
    test_array_aware_equality_shape_sensitive,
    test_array_aware_equality_tuple_vs_list,
    test_graph_ref_uses_correct_kind,
    test_graph_load_rejects_mismatched_surface_refs,
    test_graph_load_rejects_duplicate_surface_records,
    test_graph_load_rejects_unknown_edge_surface_ref,
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

"""End-to-end tests for the Replica oracle vertical slice.

Run: python tests/oracle_replica/test_oracle_pipeline.py

Covers P1.04 (adapter + mesh representation) and P1.05 (instance
extractor) as a single batch. Exit condition under test: the current
Replica scene deterministically yields 73 distinct oracle entities with
stable IDs and a replayable serialized bundle.

Stdlib only (no pytest dependency).
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

from adapters.base import ReconstructionConfig
from adapters.oracle_replica import (
    OracleReplicaAdapter, build_replica_capture_bundle,
)
from common.equality import array_aware_equal
from common.types import CameraPose
from extractors.base import InstanceExtractorConfig
from extractors.oracle_replica import OracleReplicaExtractor
from extractors.serde import dump_entity_artifacts, load_entity_artifacts
from representations.base import GeometryQuery, RenderRequest
from representations.mesh import MeshRepresentation, load_mesh_representation
from representations.serde import dump_scene_repr_bundle, load_scene_repr_bundle


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"


def _adapter_config() -> ReconstructionConfig:
    return ReconstructionConfig(
        name="oracle_replica", version="0.1", params={},
    )


def _extractor_config() -> InstanceExtractorConfig:
    return InstanceExtractorConfig(
        name="oracle_replica", version="0.1", params={},
    )


def _run_full_pipeline(scene_dir: Path):
    capture = build_replica_capture_bundle(scene_dir)
    adapter = OracleReplicaAdapter()
    repr_bundle = adapter.reconstruct(capture, _adapter_config())
    representation = MeshRepresentation(bundle=repr_bundle)
    extractor = OracleReplicaExtractor()
    artifacts = extractor.extract(representation, _extractor_config())
    return capture, repr_bundle, representation, artifacts


def _write_synthetic_scene(td: Path, objects: list[dict], scene_id: str = "test_scene") -> Path:
    sg = {"scene": scene_id, "objects": objects, "relations": []}
    cm = {
        "scene_id": scene_id,
        "source": "test",
        "axis_convention": {"up_axis": "+z", "gravity_dir_raw": [0.0, 0.0, -1.0]},
        "units": "meters",
        "room_bbox": [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]],
        "object_count": len(objects),
        "authored_relation_count": 0,
        "import_notes": {
            "z_translation_applied": 0.0,
            "dropped_undefined": 0,
            "dropped_structural": 0,
            "keep_structural": False,
            "zone_field": "always_null_for_this_scene",
            "abb_center_rotated_by_orientation_quat": True,
            "bbox_sizes_kept_in_local_frame": True,
        },
    }
    (td / "scene_graph.json").write_text(json.dumps(sg))
    (td / "capture_meta.json").write_text(json.dumps(cm))
    return td


# ---------- capability validation ----------

def test_adapter_capabilities() -> None:
    cap = OracleReplicaAdapter().capabilities()
    if not cap.produces_mesh:
        raise AssertionError("adapter must advertise produces_mesh=True")
    if cap.requires_gpu:
        raise AssertionError("oracle adapter must not require GPU")
    if cap.estimates_poses:
        raise AssertionError("oracle adapter does not estimate poses")
    if cap.typical_runtime_minutes != 0:
        raise AssertionError(f"expected zero runtime, got {cap.typical_runtime_minutes}")


def test_extractor_capabilities() -> None:
    cap = OracleReplicaExtractor().capabilities()
    if cap.label_vocab is not None:
        raise AssertionError("oracle extractor is open-vocab; label_vocab must be None")
    if cap.provides_embeddings:
        raise AssertionError("oracle extractor does not produce embeddings")
    if cap.provides_oriented_bboxes:
        raise AssertionError("oracle extractor does not produce OBBs in Phase 1")
    if cap.provides_structural_surfaces:
        raise AssertionError(
            "oracle extractor must declare provides_structural_surfaces=False; "
            "Phase 2 work"
        )
    if cap.extractor_class_hint != "all":
        raise AssertionError(f"expected extractor_class_hint='all', got {cap.extractor_class_hint!r}")


def test_representation_capabilities_declare_no_render() -> None:
    if not REPLICA_SCENE_DIR.exists():
        raise AssertionError(f"Replica scene dir missing: {REPLICA_SCENE_DIR}")
    _, repr_bundle, _, _ = _run_full_pipeline(REPLICA_SCENE_DIR)
    if repr_bundle.capabilities.renderable_channels:
        raise AssertionError(
            "oracle representation must declare empty renderable_channels; "
            f"got {sorted(repr_bundle.capabilities.renderable_channels)}"
        )
    if not repr_bundle.capabilities.deterministic:
        raise AssertionError("oracle representation must be deterministic")
    if repr_bundle.capabilities.supports_arbitrary_pose:
        raise AssertionError("oracle representation cannot render arbitrary poses")


# ---------- deterministic hashes ----------

def test_capture_bundle_hash_is_deterministic() -> None:
    c1 = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    c2 = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    if c1.bundle_hash != c2.bundle_hash:
        raise AssertionError(
            f"capture bundle hash drifted: {c1.bundle_hash!r} vs {c2.bundle_hash!r}"
        )


def test_representation_hash_is_deterministic() -> None:
    adapter = OracleReplicaAdapter()
    cap = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    b1 = adapter.reconstruct(cap, _adapter_config())
    b2 = adapter.reconstruct(cap, _adapter_config())
    if b1.representation_hash != b2.representation_hash:
        raise AssertionError(
            f"representation hash drifted: {b1.representation_hash!r} vs {b2.representation_hash!r}"
        )
    if not array_aware_equal(b1, b2):
        raise AssertionError("representation bundles are not deeply equal across runs")


def test_entity_bundle_hash_is_deterministic() -> None:
    _, repr_bundle, repr_rt, _ = _run_full_pipeline(REPLICA_SCENE_DIR)
    extractor = OracleReplicaExtractor()
    e1 = extractor.extract(repr_rt, _extractor_config())
    e2 = extractor.extract(repr_rt, _extractor_config())
    if e1.bundle_hash != e2.bundle_hash:
        raise AssertionError(
            f"entity bundle hash drifted: {e1.bundle_hash!r} vs {e2.bundle_hash!r}"
        )
    if [e.identity.object_uid for e in e1.entities] != [e.identity.object_uid for e in e2.entities]:
        raise AssertionError("entity uid ordering drifted across runs")


# ---------- exit condition: 73 distinct oracle entities ----------

def test_replica_scene_yields_73_distinct_entities() -> None:
    _, _, _, artifacts = _run_full_pipeline(REPLICA_SCENE_DIR)
    if len(artifacts.entities) != 73:
        raise AssertionError(
            f"expected 73 entities, got {len(artifacts.entities)}"
        )
    uids = {e.identity.object_uid for e in artifacts.entities}
    if len(uids) != 73:
        raise AssertionError(
            f"expected 73 distinct object_uids, got {len(uids)} (collisions present)"
        )
    if artifacts.structural_surfaces:
        raise AssertionError(
            "Phase 1 oracle path must emit no structural surfaces; "
            f"got {len(artifacts.structural_surfaces)}"
        )
    if artifacts.scene_id != "replica_room_0":
        raise AssertionError(f"unexpected scene_id: {artifacts.scene_id!r}")


# ---------- duplicate-label identity ----------

def test_duplicate_label_identity_keeps_uids_distinct() -> None:
    """Two 'chair' instances with different ids get different uids and
    both share 'chair' as an alias."""
    objects = [
        {
            "id": "obj_1", "label": "chair_1", "zone": None,
            "xyz": [1.0, 1.0, 0.5],
            "attributes": {"type": "chair_1", "bbox_sizes": [0.5, 0.5, 1.0]},
        },
        {
            "id": "obj_2", "label": "chair_2", "zone": None,
            "xyz": [2.0, 1.0, 0.5],
            "attributes": {"type": "chair_2", "bbox_sizes": [0.5, 0.5, 1.0]},
        },
        {
            "id": "obj_3", "label": "table", "zone": None,
            "xyz": [3.0, 1.0, 0.5],
            "attributes": {"type": "table", "bbox_sizes": [1.0, 1.0, 0.7]},
        },
    ]
    with tempfile.TemporaryDirectory() as td:
        scene_dir = _write_synthetic_scene(Path(td), objects, scene_id="test_dup")
        _, _, _, artifacts = _run_full_pipeline(scene_dir)
    e1, e2, e3 = artifacts.entities
    if e1.identity.object_uid == e2.identity.object_uid:
        raise AssertionError("duplicate-label entities collapsed to same uid")
    if e1.identity.display_label != "chair_1" or e2.identity.display_label != "chair_2":
        raise AssertionError(
            "display_labels lost: got "
            f"{e1.identity.display_label!r}, {e2.identity.display_label!r}"
        )
    if "chair" not in e1.identity.aliases or "chair" not in e2.identity.aliases:
        raise AssertionError(
            "base label 'chair' missing from aliases: "
            f"{e1.identity.aliases}, {e2.identity.aliases}"
        )
    # The unsuffixed 'table' should have NO base-label alias.
    if e3.identity.aliases:
        raise AssertionError(
            f"unsuffixed label should have empty aliases; got {e3.identity.aliases}"
        )
    # source_instance_ref strips the 'obj_' prefix.
    if e1.identity.source_instance_ref != "1" or e2.identity.source_instance_ref != "2":
        raise AssertionError(
            "source_instance_ref incorrect: "
            f"{e1.identity.source_instance_ref!r}, {e2.identity.source_instance_ref!r}"
        )


def test_duplicate_uid_in_input_raises() -> None:
    objects = [
        {
            "id": "obj_1", "label": "chair_1", "zone": None,
            "xyz": [1.0, 1.0, 0.5],
            "attributes": {"type": "chair_1", "bbox_sizes": [0.5, 0.5, 1.0]},
        },
        {
            "id": "obj_1", "label": "chair_2", "zone": None,
            "xyz": [2.0, 1.0, 0.5],
            "attributes": {"type": "chair_2", "bbox_sizes": [0.5, 0.5, 1.0]},
        },
    ]
    with tempfile.TemporaryDirectory() as td:
        scene_dir = _write_synthetic_scene(Path(td), objects, scene_id="test_collide")
        try:
            _run_full_pipeline(scene_dir)
        except ValueError as e:
            if "object_uid collision" not in str(e):
                raise AssertionError(f"unexpected ValueError text: {e}")
            return
    raise AssertionError("expected ValueError for duplicate uid input")


# ---------- mesh representation behavior ----------

def test_mesh_representation_render_raises() -> None:
    _, repr_bundle, representation, _ = _run_full_pipeline(REPLICA_SCENE_DIR)
    cam = CameraPose(
        camera_id="c0", position=(0.0, 0.0, 0.0),
        rotation_quat=(0.0, 0.0, 0.0, 1.0),
        intrinsics=(500.0, 500.0, 320.0, 240.0),
        width=640, height=480,
    )
    req = RenderRequest(
        request_hash="r0", camera=cam, width=640, height=480,
        channels=frozenset(["rgb"]), feature_extractor=None,
    )
    try:
        representation.render_view(req)
    except NotImplementedError:
        return
    raise AssertionError("MeshRepresentation.render_view must raise NotImplementedError in Phase 1")


def test_mesh_representation_query_geometry_describes_handle() -> None:
    _, repr_bundle, representation, _ = _run_full_pipeline(REPLICA_SCENE_DIR)
    result = representation.query_geometry(GeometryQuery(kind="describe_handle"))
    if result.kind != "handle_description":
        raise AssertionError(f"unexpected result.kind: {result.kind!r}")
    if result.payload["geometry_kind"] != repr_bundle.geometry_handle.kind:
        raise AssertionError("geometry kind not propagated through query_geometry")
    if result.payload["uri"] != repr_bundle.geometry_handle.uri:
        raise AssertionError("geometry uri not propagated through query_geometry")


def test_mesh_representation_query_geometry_unknown_kind_raises() -> None:
    _, _, representation, _ = _run_full_pipeline(REPLICA_SCENE_DIR)
    try:
        representation.query_geometry(GeometryQuery(kind="raycast"))
    except NotImplementedError:
        return
    raise AssertionError("expected NotImplementedError for unsupported query kind")


# ---------- replayable serialized bundle ----------

def test_full_pipeline_serializes_and_replays() -> None:
    _, repr_bundle, _, artifacts = _run_full_pipeline(REPLICA_SCENE_DIR)
    with tempfile.TemporaryDirectory() as td:
        repr_dir = Path(td) / "repr"
        ent_dir = Path(td) / "ent"
        dump_scene_repr_bundle(repr_bundle, repr_dir)
        dump_entity_artifacts(artifacts, ent_dir)

        loaded_repr = load_scene_repr_bundle(repr_dir)
        loaded_artifacts = load_entity_artifacts(ent_dir)

        if not array_aware_equal(repr_bundle, loaded_repr):
            raise AssertionError("repr bundle round-trip lost data")
        if not array_aware_equal(artifacts, loaded_artifacts):
            raise AssertionError("entity artifacts round-trip lost data")

        # Replayability: load_mesh_representation from disk → re-extract →
        # the new extractor output bundle_hash must equal the original.
        replayed_repr = load_mesh_representation(repr_dir)
        re_extracted = OracleReplicaExtractor().extract(replayed_repr, _extractor_config())
        if re_extracted.bundle_hash != artifacts.bundle_hash:
            raise AssertionError(
                "re-extracted bundle_hash differs from original: "
                f"{re_extracted.bundle_hash!r} vs {artifacts.bundle_hash!r}"
            )
        if [e.identity.object_uid for e in re_extracted.entities] != \
           [e.identity.object_uid for e in artifacts.entities]:
            raise AssertionError("re-extracted entity uid ordering differs from original")


# ---------- adapter / extractor boundary discipline ----------

def test_extractor_does_not_renormalize_frame() -> None:
    """SceneFrame on EntityArtifacts must be the same SceneFrame the
    adapter emitted on the SceneRepresentationBundle. The extractor must
    not reinterpret it."""
    _, repr_bundle, _, artifacts = _run_full_pipeline(REPLICA_SCENE_DIR)
    if artifacts.frame != repr_bundle.frame:
        raise AssertionError("extractor altered the SceneFrame; boundary violation")


def test_extractor_uses_representation_hash_from_bundle() -> None:
    _, repr_bundle, _, artifacts = _run_full_pipeline(REPLICA_SCENE_DIR)
    if artifacts.representation_hash != repr_bundle.representation_hash:
        raise AssertionError(
            "extractor did not propagate representation_hash from bundle"
        )


TESTS = [
    test_adapter_capabilities,
    test_extractor_capabilities,
    test_representation_capabilities_declare_no_render,
    test_capture_bundle_hash_is_deterministic,
    test_representation_hash_is_deterministic,
    test_entity_bundle_hash_is_deterministic,
    test_replica_scene_yields_73_distinct_entities,
    test_duplicate_label_identity_keeps_uids_distinct,
    test_duplicate_uid_in_input_raises,
    test_mesh_representation_render_raises,
    test_mesh_representation_query_geometry_describes_handle,
    test_mesh_representation_query_geometry_unknown_kind_raises,
    test_full_pipeline_serializes_and_replays,
    test_extractor_does_not_renormalize_frame,
    test_extractor_uses_representation_hash_from_bundle,
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

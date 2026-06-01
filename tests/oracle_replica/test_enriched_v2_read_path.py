"""P2.06 tests: OracleReplicaExtractor enriched-v2 read path.

Covers two contracts:

  1. Default mode (`enriched_v2_path=None`) preserves Phase 1 behavior
     bit-identically. The bundle_hash payload retains the historical
     4-key shape so the P1.08 compat reproduction gate is unaffected.

  2. Enriched mode (`enriched_v2_path` set) consumes the importer's
     schema-v2 output: tight world-frame AABBs from the rotated OBB,
     populated bbox_obb, structural_surfaces with `source` provenance,
     capability flags True.

Run: python tests/oracle_replica/test_enriched_v2_read_path.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.base import ReconstructionConfig
from adapters.oracle_replica import OracleReplicaAdapter, build_replica_capture_bundle
from extractors.base import InstanceExtractorConfig
from extractors.oracle_replica import OracleReplicaExtractor, _bundle_hash
from importers.replica import import_replica
from representations.mesh import MeshRepresentation


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
REPLICA_V2_DIR = REPLICA_SCENE_DIR / "enriched" / "v2"


def _adapter_config() -> ReconstructionConfig:
    return ReconstructionConfig(name="oracle_replica", version="0.1", params={})


def _extractor_config() -> InstanceExtractorConfig:
    return InstanceExtractorConfig(name="oracle_replica", version="0.1", params={})


def _write_raw_replica_scene(root: Path) -> Path:
    scene_dir = root / "raw"
    habitat = scene_dir / "habitat"
    habitat.mkdir(parents=True)
    s = math.sqrt(0.5)
    identity = [0.0, 0.0, 0.0, 1.0]
    info = {
        "gravity_dir": [0.0, 0.0, -1.0],
        "objects": [
            {"id": 1, "class_name": "table",
             "oriented_bbox": {"abb": {"center": [1.0, 0.0, -1.0], "sizes": [2.0, 4.0, 6.0]},
                               "orientation": {"rotation": [0.0, 0.0, s, s]}}},
            {"id": 100, "class_name": "floor",
             "oriented_bbox": {"abb": {"center": [0.0, 0.0, -3.95], "sizes": [10.0, 8.0, 0.1]},
                               "orientation": {"rotation": identity}}},
            {"id": 101, "class_name": "ceiling",
             "oriented_bbox": {"abb": {"center": [0.0, 0.0, 2.0], "sizes": [10.0, 8.0, 0.1]},
                               "orientation": {"rotation": identity}}},
            {"id": 102, "class_name": "wall",
             "oriented_bbox": {"abb": {"center": [4.95, 0.0, -1.0], "sizes": [0.1, 8.0, 6.0]},
                               "orientation": {"rotation": identity}}},
            {"id": 103, "class_name": "wall",
             "oriented_bbox": {"abb": {"center": [-4.95, 0.0, -1.0], "sizes": [0.1, 8.0, 6.0]},
                               "orientation": {"rotation": identity}}},
        ],
    }
    (habitat / "info_semantic.json").write_text(json.dumps(info), encoding="utf-8")
    return scene_dir


def _setup_synthetic_pipeline(td: Path) -> tuple[Path, MeshRepresentation]:
    """Write a synthetic raw scene, run both importer modes, build the
    adapter+representation off the legacy output. Returns (v2_dir,
    representation)."""
    raw = _write_raw_replica_scene(td)
    out = td / "out"
    import_replica(raw, "synthetic", out, keep_structural=False)
    import_replica(raw, "synthetic", out, keep_structural=False, enriched_v2=True)
    legacy_dir = out / "synthetic"
    v2_dir = legacy_dir / "enriched" / "v2"
    capture = build_replica_capture_bundle(legacy_dir)
    repr_bundle = OracleReplicaAdapter().reconstruct(capture, _adapter_config())
    return v2_dir, MeshRepresentation(bundle=repr_bundle)


# --- default-mode invariants (Phase 1 byte-equal) -------------------------


def test_default_mode_bundle_hash_payload_has_exact_phase1_shape() -> None:
    """Unit-test the hash payload: default mode includes exactly the 4
    Phase 1 keys, no extras. Adding a key would invalidate Phase 1
    compat reproduction."""
    rep_h = "rep_test_xxx"
    expected_payload = json.dumps({
        "representation_hash": rep_h,
        "extractor_name": "oracle_replica",
        "extractor_version": "0.1",
        "config_params": {},
    }, sort_keys=True)
    expected = f"ent_oracle_replica_{hashlib.sha256(expected_payload.encode()).hexdigest()[:16]}"
    actual = _bundle_hash(rep_h, "oracle_replica", "0.1", {})
    if actual != expected:
        raise AssertionError(
            f"default-mode bundle_hash drifted: actual={actual} expected={expected}"
        )


def test_default_mode_emits_empty_structural_surfaces() -> None:
    with tempfile.TemporaryDirectory() as td:
        _v2, representation = _setup_synthetic_pipeline(Path(td))
        artifacts = OracleReplicaExtractor().extract(representation, _extractor_config())
    if artifacts.structural_surfaces != []:
        raise AssertionError(
            f"default mode must emit empty structural_surfaces; got "
            f"{len(artifacts.structural_surfaces)} surfaces"
        )


def test_default_mode_entities_have_no_obb_and_approximate_diagnostic() -> None:
    with tempfile.TemporaryDirectory() as td:
        _v2, representation = _setup_synthetic_pipeline(Path(td))
        artifacts = OracleReplicaExtractor().extract(representation, _extractor_config())
    for e in artifacts.entities:
        if e.bbox_obb is not None:
            raise AssertionError(f"entity {e.identity.object_uid} unexpectedly has bbox_obb")
        if e.extraction_diagnostics.get("bbox_aabb_is_approximate") is not True:
            raise AssertionError(
                f"entity {e.identity.object_uid} missing approximation diagnostic"
            )


def test_default_mode_capabilities_phase1() -> None:
    cap = OracleReplicaExtractor().capabilities()
    if cap.provides_oriented_bboxes is not False:
        raise AssertionError("default capabilities.provides_oriented_bboxes must stay False")
    if cap.provides_structural_surfaces is not False:
        raise AssertionError("default capabilities.provides_structural_surfaces must stay False")


# --- enriched-v2 mode contracts -------------------------------------------


def test_enriched_v2_mode_populates_obb_on_every_entity() -> None:
    with tempfile.TemporaryDirectory() as td:
        v2_dir, representation = _setup_synthetic_pipeline(Path(td))
        artifacts = OracleReplicaExtractor(enriched_v2_path=v2_dir).extract(
            representation, _extractor_config(),
        )
    for e in artifacts.entities:
        if e.bbox_obb is None:
            raise AssertionError(f"entity {e.identity.object_uid} missing bbox_obb on v2 path")
        if "bbox_aabb_is_approximate" in e.extraction_diagnostics:
            raise AssertionError(
                f"entity {e.identity.object_uid} still has approximation diagnostic on v2 path"
            )


def test_enriched_v2_mode_populates_structural_surfaces_with_source() -> None:
    with tempfile.TemporaryDirectory() as td:
        v2_dir, representation = _setup_synthetic_pipeline(Path(td))
        artifacts = OracleReplicaExtractor(enriched_v2_path=v2_dir).extract(
            representation, _extractor_config(),
        )
    if len(artifacts.structural_surfaces) == 0:
        raise AssertionError("v2 path emitted zero structural_surfaces")
    for s in artifacts.structural_surfaces:
        if s.source != "habitat_label":
            raise AssertionError(
                f"surface {s.surface_uid} source={s.source!r}, expected 'habitat_label'"
            )
    counts = Counter(s.surface_type for s in artifacts.structural_surfaces)
    if counts != {"floor": 1, "wall": 2, "ceiling": 1}:
        raise AssertionError(f"synthetic surface counts wrong: {counts!r}")


def test_enriched_v2_mode_capabilities_flip() -> None:
    with tempfile.TemporaryDirectory() as td:
        v2_dir, _r = _setup_synthetic_pipeline(Path(td))
        cap = OracleReplicaExtractor(enriched_v2_path=v2_dir).capabilities()
    if cap.provides_oriented_bboxes is not True:
        raise AssertionError("v2 capabilities.provides_oriented_bboxes must be True")
    if cap.provides_structural_surfaces is not True:
        raise AssertionError("v2 capabilities.provides_structural_surfaces must be True")


def test_enriched_v2_diagnostics_record_surface_count() -> None:
    with tempfile.TemporaryDirectory() as td:
        v2_dir, representation = _setup_synthetic_pipeline(Path(td))
        artifacts = OracleReplicaExtractor(enriched_v2_path=v2_dir).extract(
            representation, _extractor_config(),
        )
    if artifacts.diagnostics.n_structural_surfaces != len(artifacts.structural_surfaces):
        raise AssertionError(
            f"n_structural_surfaces diagnostic mismatch: "
            f"{artifacts.diagnostics.n_structural_surfaces} vs "
            f"{len(artifacts.structural_surfaces)}"
        )


def test_default_and_v2_bundle_hashes_differ() -> None:
    with tempfile.TemporaryDirectory() as td:
        v2_dir, representation = _setup_synthetic_pipeline(Path(td))
        default = OracleReplicaExtractor().extract(representation, _extractor_config())
        v2 = OracleReplicaExtractor(enriched_v2_path=v2_dir).extract(
            representation, _extractor_config(),
        )
    if default.bundle_hash == v2.bundle_hash:
        raise AssertionError("default and v2 builds must produce different bundle_hashes")


def test_enriched_v2_mode_is_deterministic() -> None:
    with tempfile.TemporaryDirectory() as td:
        v2_dir, representation = _setup_synthetic_pipeline(Path(td))
        ext = OracleReplicaExtractor(enriched_v2_path=v2_dir)
        a = ext.extract(representation, _extractor_config())
        b = ext.extract(representation, _extractor_config())
    if a.bundle_hash != b.bundle_hash:
        raise AssertionError("v2 mode produced non-deterministic bundle_hash")
    if len(a.structural_surfaces) != len(b.structural_surfaces):
        raise AssertionError("v2 mode produced different surface counts")


def test_enriched_v2_bundle_hash_is_path_independent() -> None:
    """The v2 bundle is addressed by enriched content, not a local path."""
    with tempfile.TemporaryDirectory() as td_a, tempfile.TemporaryDirectory() as td_b:
        v2_a, representation_a = _setup_synthetic_pipeline(Path(td_a))
        v2_b, representation_b = _setup_synthetic_pipeline(Path(td_b))
        a = OracleReplicaExtractor(enriched_v2_path=v2_a).extract(
            representation_a, _extractor_config(),
        )
        b = OracleReplicaExtractor(enriched_v2_path=v2_b).extract(
            representation_b, _extractor_config(),
        )
    if a.representation_hash != b.representation_hash:
        raise AssertionError("synthetic representation hashes unexpectedly differ")
    if a.bundle_hash != b.bundle_hash:
        raise AssertionError(
            f"identical v2 content at different paths changed bundle_hash: "
            f"{a.bundle_hash!r} vs {b.bundle_hash!r}"
        )


def test_enriched_v2_bundle_hash_changes_when_content_changes() -> None:
    """Editing enriched input bytes must invalidate the downstream bundle hash."""
    with tempfile.TemporaryDirectory() as td:
        v2_dir, representation = _setup_synthetic_pipeline(Path(td))
        extractor = OracleReplicaExtractor(enriched_v2_path=v2_dir)
        before = extractor.extract(representation, _extractor_config())
        scene_path = v2_dir / "scene_graph.json"
        payload = json.loads(scene_path.read_text(encoding="utf-8"))
        payload["objects"][0]["label"] = "renamed_table"
        scene_path.write_text(json.dumps(payload), encoding="utf-8")
        after = extractor.extract(representation, _extractor_config())
    if before.bundle_hash == after.bundle_hash:
        raise AssertionError("v2 content drift did not invalidate bundle_hash")


def test_enriched_v2_rejects_schema_v1_input() -> None:
    """If the v2 dir somehow contains a v1-shaped scene_graph (no
    schema_version), the extractor must fail explicitly rather than
    silently misinterpret."""
    with tempfile.TemporaryDirectory() as td:
        v2_dir, representation = _setup_synthetic_pipeline(Path(td))
        scene_path = v2_dir / "scene_graph.json"
        sg = json.loads(scene_path.read_text(encoding="utf-8"))
        sg.pop("schema_version", None)
        scene_path.write_text(json.dumps(sg), encoding="utf-8")
        raised = False
        try:
            OracleReplicaExtractor(enriched_v2_path=v2_dir).extract(
                representation, _extractor_config(),
            )
        except ValueError:
            raised = True
    if not raised:
        raise AssertionError("expected ValueError on missing schema_version")


def test_enriched_v2_rejects_scene_id_mismatch() -> None:
    with tempfile.TemporaryDirectory() as td:
        v2_dir, representation = _setup_synthetic_pipeline(Path(td))
        scene_path = v2_dir / "scene_graph.json"
        sg = json.loads(scene_path.read_text(encoding="utf-8"))
        sg["scene"] = "wrong_scene"
        scene_path.write_text(json.dumps(sg), encoding="utf-8")
        try:
            OracleReplicaExtractor(enriched_v2_path=v2_dir).extract(
                representation, _extractor_config(),
            )
        except ValueError as e:
            if "wrong_scene" not in str(e) or "synthetic" not in str(e):
                raise AssertionError(f"scene-id mismatch error lacks context: {e}")
            return
    raise AssertionError("expected ValueError on enriched-v2 scene-id mismatch")


# --- real Replica room_0 (skip if v2 artifact not present) ----------------


def test_real_replica_v2_pipeline_matches_expected_counts() -> None:
    if not (REPLICA_V2_DIR / "scene_graph.json").exists():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    capture = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    representation = MeshRepresentation(
        bundle=OracleReplicaAdapter().reconstruct(capture, _adapter_config())
    )
    artifacts = OracleReplicaExtractor(enriched_v2_path=REPLICA_V2_DIR).extract(
        representation, _extractor_config(),
    )
    if len(artifacts.entities) != 73:
        raise AssertionError(f"entity count wrong: {len(artifacts.entities)}, expected 73")
    if len(artifacts.structural_surfaces) != 7:
        raise AssertionError(
            f"surface count wrong: {len(artifacts.structural_surfaces)}, expected 7"
        )
    counts = Counter(s.surface_type for s in artifacts.structural_surfaces)
    if counts != {"floor": 1, "wall": 5, "ceiling": 1}:
        raise AssertionError(f"Replica surface counts wrong: {counts!r}")
    for s in artifacts.structural_surfaces:
        if s.source != "habitat_label":
            raise AssertionError(
                f"Replica surface {s.surface_uid} source={s.source!r}, "
                f"expected 'habitat_label'"
            )
    for e in artifacts.entities:
        if e.bbox_obb is None:
            raise AssertionError(f"Replica entity {e.identity.object_uid} missing bbox_obb")


TESTS = [
    test_default_mode_bundle_hash_payload_has_exact_phase1_shape,
    test_default_mode_emits_empty_structural_surfaces,
    test_default_mode_entities_have_no_obb_and_approximate_diagnostic,
    test_default_mode_capabilities_phase1,
    test_enriched_v2_mode_populates_obb_on_every_entity,
    test_enriched_v2_mode_populates_structural_surfaces_with_source,
    test_enriched_v2_mode_capabilities_flip,
    test_enriched_v2_diagnostics_record_surface_count,
    test_default_and_v2_bundle_hashes_differ,
    test_enriched_v2_mode_is_deterministic,
    test_enriched_v2_bundle_hash_is_path_independent,
    test_enriched_v2_bundle_hash_changes_when_content_changes,
    test_enriched_v2_rejects_schema_v1_input,
    test_enriched_v2_rejects_scene_id_mismatch,
    test_real_replica_v2_pipeline_matches_expected_counts,
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

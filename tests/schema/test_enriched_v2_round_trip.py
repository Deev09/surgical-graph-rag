"""P2.05 tests: serde round-trip for the enriched-v2 oracle bundle.

Exercises the dump → load path on EntityArtifacts produced via
OracleReplicaExtractor(enriched_v2_path=...). Covers:

  - field-for-field equality after dump/load (array_aware_equal),
  - `source` round-trips on every StructuralSurface,
  - schema_version on the manifest matches the bumped value (2),
  - loading a v1 manifest fails with SchemaVersionError,
  - loading a v2 manifest with a missing/unknown `source` fails fast.

Run: python tests/schema/test_enriched_v2_round_trip.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.base import ReconstructionConfig
from adapters.oracle_replica import OracleReplicaAdapter, build_replica_capture_bundle
from common.equality import array_aware_equal
from common.serde import SchemaVersionError
from extractors.base import InstanceExtractorConfig
from extractors.oracle_replica import OracleReplicaExtractor
from extractors.serde import (
    CURRENT_SCHEMA_VERSION,
    dump_entity_artifacts,
    load_entity_artifacts,
)
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


def _build_v2_artifacts(td: Path):
    raw = _write_raw_replica_scene(td)
    out = td / "out"
    import_replica(raw, "synthetic", out, keep_structural=False)
    import_replica(raw, "synthetic", out, keep_structural=False, enriched_v2=True)
    legacy_dir = out / "synthetic"
    v2_dir = legacy_dir / "enriched" / "v2"
    capture = build_replica_capture_bundle(legacy_dir)
    repr_bundle = OracleReplicaAdapter().reconstruct(capture, _adapter_config())
    representation = MeshRepresentation(bundle=repr_bundle)
    artifacts = OracleReplicaExtractor(enriched_v2_path=v2_dir).extract(
        representation, _extractor_config(),
    )
    return artifacts


def _dump_then_load(artifacts, td: Path):
    dump_dir = td / "dump"
    dump_entity_artifacts(artifacts, dump_dir)
    return load_entity_artifacts(dump_dir), dump_dir


# --- synthetic round-trip --------------------------------------------------


def test_round_trip_preserves_schema_version() -> None:
    with tempfile.TemporaryDirectory() as td:
        artifacts = _build_v2_artifacts(Path(td))
        loaded, _ = _dump_then_load(artifacts, Path(td))
    if artifacts.schema_version != CURRENT_SCHEMA_VERSION:
        raise AssertionError(
            f"produced artifacts schema_version={artifacts.schema_version}, "
            f"expected {CURRENT_SCHEMA_VERSION}"
        )
    if loaded.schema_version != CURRENT_SCHEMA_VERSION:
        raise AssertionError(
            f"loaded schema_version={loaded.schema_version}, "
            f"expected {CURRENT_SCHEMA_VERSION}"
        )


def test_round_trip_preserves_bundle_hash() -> None:
    with tempfile.TemporaryDirectory() as td:
        artifacts = _build_v2_artifacts(Path(td))
        loaded, _ = _dump_then_load(artifacts, Path(td))
    if loaded.bundle_hash != artifacts.bundle_hash:
        raise AssertionError(
            f"bundle_hash drift: dumped={artifacts.bundle_hash} loaded={loaded.bundle_hash}"
        )


def test_round_trip_preserves_source_on_every_structural_surface() -> None:
    with tempfile.TemporaryDirectory() as td:
        artifacts = _build_v2_artifacts(Path(td))
        loaded, _ = _dump_then_load(artifacts, Path(td))
    if len(loaded.structural_surfaces) != len(artifacts.structural_surfaces):
        raise AssertionError(
            f"surface count drift: {len(artifacts.structural_surfaces)} vs "
            f"{len(loaded.structural_surfaces)}"
        )
    for a, b in zip(artifacts.structural_surfaces, loaded.structural_surfaces):
        if a.source != b.source:
            raise AssertionError(
                f"surface {a.surface_uid} source drift: {a.source} vs {b.source}"
            )
        if a.source != "habitat_label":
            raise AssertionError(
                f"surface {a.surface_uid} unexpected source {a.source!r}; "
                f"synthetic fixture should be habitat_label"
            )


def test_round_trip_field_equality_via_array_aware_equal() -> None:
    with tempfile.TemporaryDirectory() as td:
        artifacts = _build_v2_artifacts(Path(td))
        loaded, _ = _dump_then_load(artifacts, Path(td))
    for a, b in zip(artifacts.entities, loaded.entities):
        if not array_aware_equal(a, b):
            raise AssertionError(f"entity {a.identity.object_uid} round-trip mismatch")
    for a, b in zip(artifacts.structural_surfaces, loaded.structural_surfaces):
        if not array_aware_equal(a, b):
            raise AssertionError(f"surface {a.surface_uid} round-trip mismatch")
    if not array_aware_equal(artifacts.diagnostics, loaded.diagnostics):
        raise AssertionError("diagnostics round-trip mismatch")


def test_round_trip_manifest_records_source_field() -> None:
    """Direct inspection: the dumped JSON must contain a `source` key on
    every structural surface entry."""
    with tempfile.TemporaryDirectory() as td:
        artifacts = _build_v2_artifacts(Path(td))
        _loaded, dump_dir = _dump_then_load(artifacts, Path(td))
        manifest = json.loads((dump_dir / "manifest.json").read_text(encoding="utf-8"))
    for s in manifest["structural_surfaces"]:
        if "source" not in s:
            raise AssertionError(f"dumped surface missing 'source' key: {s!r}")
        if s["source"] != "habitat_label":
            raise AssertionError(
                f"dumped surface unexpected source: {s['source']!r}"
            )
    if manifest["schema_version"] != CURRENT_SCHEMA_VERSION:
        raise AssertionError(
            f"dumped schema_version wrong: {manifest['schema_version']!r}"
        )


# --- failure modes ---------------------------------------------------------


def test_load_rejects_v1_manifest_with_schema_version_error() -> None:
    """A manifest with schema_version=1 must fail the strict version
    check now that we're at v2."""
    with tempfile.TemporaryDirectory() as td:
        artifacts = _build_v2_artifacts(Path(td))
        dump_dir = Path(td) / "dump_v1"
        dump_entity_artifacts(artifacts, dump_dir)
        manifest_path = dump_dir / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["schema_version"] = 1
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        raised = False
        try:
            load_entity_artifacts(dump_dir)
        except SchemaVersionError:
            raised = True
    if not raised:
        raise AssertionError("expected SchemaVersionError on v1 manifest")


def test_load_rejects_v2_manifest_missing_source() -> None:
    with tempfile.TemporaryDirectory() as td:
        artifacts = _build_v2_artifacts(Path(td))
        dump_dir = Path(td) / "dump_nosrc"
        dump_entity_artifacts(artifacts, dump_dir)
        manifest_path = dump_dir / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for s in payload["structural_surfaces"]:
            s.pop("source", None)
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        raised = False
        try:
            load_entity_artifacts(dump_dir)
        except ValueError:
            raised = True
    if not raised:
        raise AssertionError("expected ValueError on missing 'source' field")


def test_load_rejects_v2_manifest_with_unknown_source() -> None:
    with tempfile.TemporaryDirectory() as td:
        artifacts = _build_v2_artifacts(Path(td))
        dump_dir = Path(td) / "dump_badsrc"
        dump_entity_artifacts(artifacts, dump_dir)
        manifest_path = dump_dir / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["structural_surfaces"][0]["source"] = "manual_authored"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        raised = False
        try:
            load_entity_artifacts(dump_dir)
        except ValueError:
            raised = True
    if not raised:
        raise AssertionError("expected ValueError on unrecognized 'source' value")


# --- real Replica round-trip (skip if v2 artifact not present) ------------


def test_real_replica_v2_artifacts_round_trip() -> None:
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
    with tempfile.TemporaryDirectory() as td:
        loaded, _ = _dump_then_load(artifacts, Path(td))
    if loaded.bundle_hash != artifacts.bundle_hash:
        raise AssertionError("Replica v2 round-trip changed bundle_hash")
    if len(loaded.entities) != 73:
        raise AssertionError(f"Replica entity count drift: {len(loaded.entities)}")
    if len(loaded.structural_surfaces) != 7:
        raise AssertionError(
            f"Replica surface count drift: {len(loaded.structural_surfaces)}"
        )
    for a, b in zip(artifacts.entities, loaded.entities):
        if not array_aware_equal(a, b):
            raise AssertionError(f"Replica entity {a.identity.object_uid} round-trip mismatch")
    for a, b in zip(artifacts.structural_surfaces, loaded.structural_surfaces):
        if not array_aware_equal(a, b):
            raise AssertionError(f"Replica surface {a.surface_uid} round-trip mismatch")
        if a.source != "habitat_label":
            raise AssertionError(
                f"Replica surface {a.surface_uid} source wrong: {a.source!r}"
            )


TESTS = [
    test_round_trip_preserves_schema_version,
    test_round_trip_preserves_bundle_hash,
    test_round_trip_preserves_source_on_every_structural_surface,
    test_round_trip_field_equality_via_array_aware_equal,
    test_round_trip_manifest_records_source_field,
    test_load_rejects_v1_manifest_with_schema_version_error,
    test_load_rejects_v2_manifest_missing_source,
    test_load_rejects_v2_manifest_with_unknown_source,
    test_real_replica_v2_artifacts_round_trip,
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

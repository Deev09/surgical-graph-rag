"""P2.02 tests: Replica importer schema-v2 OBB output.

Run: python tests/importers/test_replica_enriched_v2.py
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

from importers.replica import ENRICHED_SCHEMA_VERSION, import_replica


def _write_raw_scene(root: Path) -> Path:
    scene_dir = root / "raw"
    habitat = scene_dir / "habitat"
    habitat.mkdir(parents=True)
    s = math.sqrt(0.5)
    identity = [0.0, 0.0, 0.0, 1.0]
    info = {
        "gravity_dir": [0.0, 0.0, -1.0],
        "objects": [
            {
                "id": 1,
                "class_name": "table",
                "oriented_bbox": {
                    "abb": {"center": [1.0, 0.0, -1.0], "sizes": [2.0, 4.0, 6.0]},
                    "orientation": {"rotation": [0.0, 0.0, s, s]},
                },
            },
            {
                "id": 2,
                "class_name": "undefined",
                "oriented_bbox": {
                    "abb": {"center": [0.0, 0.0, 0.0], "sizes": [1.0, 1.0, 1.0]},
                    "orientation": {"rotation": [0.0, 0.0, 0.0, 1.0]},
                },
            },
            # Structural set required by --enriched-v2 (P2.03). All instances
            # placed so the table's world-frame AABB still bounds z_min, so
            # the P2.02 translated-AABB expectations remain unchanged.
            {
                "id": 100,
                "class_name": "floor",
                "oriented_bbox": {
                    "abb": {"center": [0.0, 0.0, -3.95], "sizes": [10.0, 8.0, 0.1]},
                    "orientation": {"rotation": identity},
                },
            },
            {
                "id": 101,
                "class_name": "ceiling",
                "oriented_bbox": {
                    "abb": {"center": [0.0, 0.0, 2.0], "sizes": [10.0, 8.0, 0.1]},
                    "orientation": {"rotation": identity},
                },
            },
            {
                "id": 102,
                "class_name": "wall",
                "oriented_bbox": {
                    "abb": {"center": [4.95, 0.0, -1.0], "sizes": [0.1, 8.0, 6.0]},
                    "orientation": {"rotation": identity},
                },
            },
            {
                "id": 103,
                "class_name": "wall",
                "oriented_bbox": {
                    "abb": {"center": [-4.95, 0.0, -1.0], "sizes": [0.1, 8.0, 6.0]},
                    "orientation": {"rotation": identity},
                },
            },
        ],
    }
    (habitat / "info_semantic.json").write_text(json.dumps(info), encoding="utf-8")
    return scene_dir


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_legacy_path_stays_unversioned_and_separate() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        scene_dir = _write_raw_scene(root)
        out = root / "out"
        import_replica(scene_dir, "synthetic", out, keep_structural=False)
        scene = _load(out / "synthetic" / "scene_graph.json")
    if "schema_version" in scene:
        raise AssertionError("legacy output must remain unversioned")
    if "bbox_obb" in scene["objects"][0] or "bbox_aabb" in scene["objects"][0]:
        raise AssertionError("legacy output must not gain v2 geometry fields")


def test_enriched_v2_writes_versioned_path_without_touching_legacy() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        scene_dir = _write_raw_scene(root)
        out = root / "out"
        legacy_dir = out / "synthetic"
        legacy_dir.mkdir(parents=True)
        legacy_scene = legacy_dir / "scene_graph.json"
        legacy_meta = legacy_dir / "capture_meta.json"
        legacy_scene.write_text("legacy-scene", encoding="utf-8")
        legacy_meta.write_text("legacy-meta", encoding="utf-8")
        import_replica(scene_dir, "synthetic", out, keep_structural=False, enriched_v2=True)
        scene = _load(out / "synthetic" / "enriched" / "v2" / "scene_graph.json")
        meta = _load(out / "synthetic" / "enriched" / "v2" / "capture_meta.json")
        legacy_after = (legacy_scene.read_text(), legacy_meta.read_text())
    if legacy_after != ("legacy-scene", "legacy-meta"):
        raise AssertionError(f"enriched path modified legacy fixture: {legacy_after!r}")
    if scene["schema_version"] != ENRICHED_SCHEMA_VERSION:
        raise AssertionError(f"scene schema version wrong: {scene.get('schema_version')!r}")
    if meta["schema_version"] != ENRICHED_SCHEMA_VERSION:
        raise AssertionError(f"meta schema version wrong: {meta.get('schema_version')!r}")


def test_enriched_v2_derives_tight_world_aabb_from_rotated_obb() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        scene_dir = _write_raw_scene(root)
        out = root / "out"
        import_replica(scene_dir, "synthetic", out, keep_structural=False, enriched_v2=True)
        scene = _load(out / "synthetic" / "enriched" / "v2" / "scene_graph.json")
    obj = scene["objects"][0]
    if obj["bbox_obb"]["center"] != [0.0, 1.0, 3.0]:
        raise AssertionError(f"unexpected translated OBB center: {obj['bbox_obb']['center']!r}")
    if obj["bbox_obb"]["extents"] != [1.0, 2.0, 3.0]:
        raise AssertionError(f"unexpected extents: {obj['bbox_obb']['extents']!r}")
    if obj["bbox_aabb"] != [[-2.0, 0.0, 0.0], [2.0, 2.0, 6.0]]:
        raise AssertionError(f"unexpected tight AABB: {obj['bbox_aabb']!r}")


def test_enriched_v2_is_deterministic() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        scene_dir = _write_raw_scene(root)
        out_a = root / "a"
        out_b = root / "b"
        import_replica(scene_dir, "synthetic", out_a, keep_structural=False, enriched_v2=True)
        import_replica(scene_dir, "synthetic", out_b, keep_structural=False, enriched_v2=True)
        scene_a = (out_a / "synthetic" / "enriched" / "v2" / "scene_graph.json").read_bytes()
        scene_b = (out_b / "synthetic" / "enriched" / "v2" / "scene_graph.json").read_bytes()
        meta_a = (out_a / "synthetic" / "enriched" / "v2" / "capture_meta.json").read_bytes()
        meta_b = (out_b / "synthetic" / "enriched" / "v2" / "capture_meta.json").read_bytes()
    if scene_a != scene_b or meta_a != meta_b:
        raise AssertionError("enriched v2 importer output is not deterministic")


TESTS = [
    test_legacy_path_stays_unversioned_and_separate,
    test_enriched_v2_writes_versioned_path_without_touching_legacy,
    test_enriched_v2_derives_tight_world_aabb_from_rotated_obb,
    test_enriched_v2_is_deterministic,
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

"""P2.03 tests: structural-surface extraction from Habitat labels.

Run: python tests/importers/test_replica_enriched_v2_structural.py

Covers the synthetic fixture (deterministic geometry) plus a real Replica
room_0 smoke test that activates only if data/replica/room_0/habitat/
info_semantic.json is present (per Q1, raw inputs are user-supplied and
the verifier gates them — when absent the smoke is skipped, not failed).
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

from importers.replica import (
    ENRICHED_SCHEMA_VERSION,
    STRUCTURAL_REQUIRED,
    import_replica,
)

REPLICA_RAW = REPO_ROOT / "data" / "replica" / "room_0"
REPLICA_INFO = REPLICA_RAW / "habitat" / "info_semantic.json"


def _write_raw_scene_full(root: Path) -> Path:
    """Synthetic raw scene with the full structural set (1 floor + 1 ceiling
    + 2 walls). Matches the fixture in test_replica_enriched_v2.py — kept
    in lockstep here so geometry expectations are easy to hand-check."""
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


def _write_raw_scene_missing(root: Path, missing_class: str) -> Path:
    """Synthetic raw scene with one required structural class omitted."""
    scene_dir = root / "raw"
    habitat = scene_dir / "habitat"
    habitat.mkdir(parents=True)
    identity = [0.0, 0.0, 0.0, 1.0]
    base = {
        "floor": {
            "id": 100, "class_name": "floor",
            "oriented_bbox": {
                "abb": {"center": [0.0, 0.0, -3.95], "sizes": [10.0, 8.0, 0.1]},
                "orientation": {"rotation": identity},
            },
        },
        "ceiling": {
            "id": 101, "class_name": "ceiling",
            "oriented_bbox": {
                "abb": {"center": [0.0, 0.0, 2.0], "sizes": [10.0, 8.0, 0.1]},
                "orientation": {"rotation": identity},
            },
        },
        "wall": {
            "id": 102, "class_name": "wall",
            "oriented_bbox": {
                "abb": {"center": [4.95, 0.0, -1.0], "sizes": [0.1, 8.0, 6.0]},
                "orientation": {"rotation": identity},
            },
        },
    }
    objects = [
        {
            "id": 1, "class_name": "table",
            "oriented_bbox": {
                "abb": {"center": [0.0, 0.0, -1.0], "sizes": [2.0, 2.0, 6.0]},
                "orientation": {"rotation": identity},
            },
        },
    ]
    for cname, inst in base.items():
        if cname != missing_class:
            objects.append(inst)
    info = {"gravity_dir": [0.0, 0.0, -1.0], "objects": objects}
    (habitat / "info_semantic.json").write_text(json.dumps(info), encoding="utf-8")
    return scene_dir


def _run_v2(fixture_writer, root: Path) -> dict:
    scene_dir = fixture_writer(root)
    out = root / "out"
    import_replica(scene_dir, "synthetic", out, keep_structural=False, enriched_v2=True)
    return json.loads(
        (out / "synthetic" / "enriched" / "v2" / "scene_graph.json").read_text(encoding="utf-8")
    )


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _winding_dot(poly, normal):
    a = _sub(poly[1], poly[0])
    b = _sub(poly[2], poly[0])
    return _dot(_cross(a, b), normal)


# --- synthetic-fixture tests ----------------------------------------------


def test_structural_surfaces_count_per_type_synthetic() -> None:
    with tempfile.TemporaryDirectory() as td:
        scene = _run_v2(_write_raw_scene_full, Path(td))
    counts: dict[str, int] = {}
    for s in scene["structural_surfaces"]:
        counts[s["surface_type"]] = counts.get(s["surface_type"], 0) + 1
    if counts != {"floor": 1, "ceiling": 1, "wall": 2}:
        raise AssertionError(f"structural counts wrong: {counts!r}")


def test_structural_surfaces_all_sourced_from_habitat_label() -> None:
    with tempfile.TemporaryDirectory() as td:
        scene = _run_v2(_write_raw_scene_full, Path(td))
    sources = {s["source"] for s in scene["structural_surfaces"]}
    if sources != {"habitat_label"}:
        raise AssertionError(f"expected only habitat_label sources, got {sources!r}")


def test_structural_surface_uid_schema_uses_habitat_ids() -> None:
    with tempfile.TemporaryDirectory() as td:
        scene = _run_v2(_write_raw_scene_full, Path(td))
    uids = {s["surface_type"]: [] for s in scene["structural_surfaces"]}
    for s in scene["structural_surfaces"]:
        uids[s["surface_type"]].append(s["surface_uid"])
    if uids["floor"] != ["floor_100"]:
        raise AssertionError(f"floor UID wrong: {uids['floor']!r}")
    if uids["ceiling"] != ["ceiling_101"]:
        raise AssertionError(f"ceiling UID wrong: {uids['ceiling']!r}")
    wall_set = set(uids["wall"])
    # Wall 102 sits at x=+4.95, interior is at x=0, so its interior-facing
    # normal points toward -x => xminus. Wall 103 at x=-4.95 => xplus.
    if wall_set != {"wall_102_xminus", "wall_103_xplus"}:
        raise AssertionError(f"wall UIDs wrong: {sorted(wall_set)!r}")


def test_structural_surface_uids_unique() -> None:
    with tempfile.TemporaryDirectory() as td:
        scene = _run_v2(_write_raw_scene_full, Path(td))
    uids = [s["surface_uid"] for s in scene["structural_surfaces"]]
    if len(uids) != len(set(uids)):
        raise AssertionError(f"duplicate surface UIDs: {uids!r}")


def test_floor_and_ceiling_normals_aligned_with_gravity_axis() -> None:
    with tempfile.TemporaryDirectory() as td:
        scene = _run_v2(_write_raw_scene_full, Path(td))
    for s in scene["structural_surfaces"]:
        n = s["plane"]["normal"]
        if s["surface_type"] == "floor" and n[2] <= 0.99:
            raise AssertionError(f"floor normal not +z: {n!r}")
        if s["surface_type"] == "ceiling" and n[2] >= -0.99:
            raise AssertionError(f"ceiling normal not -z: {n!r}")
        if s["surface_type"] == "wall" and abs(n[2]) >= 0.05:
            raise AssertionError(f"wall normal not gravity-perpendicular: {n!r}")


def test_wall_normals_face_floor_face_centroid() -> None:
    """Adjustment (a): walls oriented using a structural interior reference
    point (floor-face centroid), NOT the object-centroid room_bbox."""
    with tempfile.TemporaryDirectory() as td:
        scene = _run_v2(_write_raw_scene_full, Path(td))
    floors = [s for s in scene["structural_surfaces"] if s["surface_type"] == "floor"]
    if not floors:
        raise AssertionError("no floor in synthetic fixture")
    floor_poly = floors[0]["polygon"]
    floor_centroid = [sum(c[k] for c in floor_poly) / 4.0 for k in range(3)]
    for s in scene["structural_surfaces"]:
        if s["surface_type"] != "wall":
            continue
        n = s["plane"]["normal"]
        wall_obb_center = s["source_obb"]["center"]
        delta = _sub(floor_centroid, wall_obb_center)
        if _dot(n, delta) <= 0:
            raise AssertionError(
                f"wall {s['surface_uid']} normal does not face floor-face centroid: "
                f"normal={n!r}, delta={delta!r}, dot={_dot(n, delta)}"
            )


def test_polygon_winding_matches_normal() -> None:
    """Adjustment (c): mathematical winding test
    dot(cross(p1-p0, p2-p0), normal) > 0."""
    with tempfile.TemporaryDirectory() as td:
        scene = _run_v2(_write_raw_scene_full, Path(td))
    for s in scene["structural_surfaces"]:
        n = s["plane"]["normal"]
        poly = s["polygon"]
        if len(poly) != 4:
            raise AssertionError(f"{s['surface_uid']} polygon has {len(poly)} corners")
        wd = _winding_dot(poly, n)
        if wd <= 0:
            raise AssertionError(
                f"{s['surface_uid']} winding test failed: dot(cross, normal)={wd}"
            )


def test_polygon_corners_on_plane() -> None:
    """Plane invariant: every polygon corner satisfies |normal · p + d| < tol.
    Tolerance accounts for the 6-decimal rounding in the importer output."""
    with tempfile.TemporaryDirectory() as td:
        scene = _run_v2(_write_raw_scene_full, Path(td))
    tol = 5e-6
    for s in scene["structural_surfaces"]:
        n = s["plane"]["normal"]
        d = s["plane"]["d"]
        for p in s["polygon"]:
            residual = abs(_dot(n, p) + d)
            if residual >= tol:
                raise AssertionError(
                    f"{s['surface_uid']} corner {p!r} off plane by {residual:.2e}"
                )


def test_source_obb_fields_present_and_consistent() -> None:
    """Adjustment (4): retain source_instance_ref + full source_obb evidence."""
    with tempfile.TemporaryDirectory() as td:
        scene = _run_v2(_write_raw_scene_full, Path(td))
    for s in scene["structural_surfaces"]:
        if "source_instance_ref" not in s:
            raise AssertionError(f"{s['surface_uid']} missing source_instance_ref")
        obb = s.get("source_obb")
        if not obb:
            raise AssertionError(f"{s['surface_uid']} missing source_obb")
        for key in ("center", "extents", "rotation_quat"):
            if key not in obb:
                raise AssertionError(f"{s['surface_uid']} source_obb missing {key}")
        if len(obb["center"]) != 3 or len(obb["extents"]) != 3:
            raise AssertionError(f"{s['surface_uid']} source_obb dims wrong")
        if len(obb["rotation_quat"]) != 4:
            raise AssertionError(f"{s['surface_uid']} source_obb quat dim wrong")


def test_missing_structural_class_raises_systemexit() -> None:
    """P2.03 fails explicitly when any required Habitat class is absent."""
    for missing in STRUCTURAL_REQUIRED:
        with tempfile.TemporaryDirectory() as td:
            scene_dir = _write_raw_scene_missing(Path(td), missing)
            out = Path(td) / "out"
            raised = False
            try:
                import_replica(scene_dir, "synthetic", out, keep_structural=False, enriched_v2=True)
            except SystemExit as e:
                raised = True
                if missing not in str(e):
                    raise AssertionError(
                        f"SystemExit message did not name missing class {missing!r}: {e}"
                    )
            if not raised:
                raise AssertionError(
                    f"importer did not fail on missing structural class {missing!r}"
                )


def test_structural_surfaces_deterministic() -> None:
    with tempfile.TemporaryDirectory() as td_a, tempfile.TemporaryDirectory() as td_b:
        a = _run_v2(_write_raw_scene_full, Path(td_a))
        b = _run_v2(_write_raw_scene_full, Path(td_b))
    if a["structural_surfaces"] != b["structural_surfaces"]:
        raise AssertionError("structural_surfaces section is not deterministic")


def test_capture_meta_records_structural_counts() -> None:
    with tempfile.TemporaryDirectory() as td:
        scene_dir = _write_raw_scene_full(Path(td))
        out = Path(td) / "out"
        import_replica(scene_dir, "synthetic", out, keep_structural=False, enriched_v2=True)
        meta = json.loads(
            (out / "synthetic" / "enriched" / "v2" / "capture_meta.json").read_text(encoding="utf-8")
        )
    if meta.get("schema_version") != ENRICHED_SCHEMA_VERSION:
        raise AssertionError("capture_meta schema_version wrong")
    notes = meta["import_notes"]
    if notes.get("structural_surface_count") != 4:
        raise AssertionError(f"structural_surface_count wrong: {notes!r}")
    if notes.get("structural_surface_count_per_type") != {"floor": 1, "ceiling": 1, "wall": 2}:
        raise AssertionError(f"per_type wrong: {notes!r}")
    if notes.get("structural_surface_sources") != ["habitat_label"]:
        raise AssertionError(f"sources wrong: {notes!r}")


# --- real Replica room_0 smoke (skip if data absent) -----------------------


def test_real_replica_room0_has_one_floor_five_walls_one_ceiling() -> None:
    if not REPLICA_INFO.exists():
        print("  SKIP (raw Replica data not on disk)")
        return
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out"
        import_replica(REPLICA_RAW, "replica_room_0", out, keep_structural=False, enriched_v2=True)
        scene = json.loads(
            (out / "replica_room_0" / "enriched" / "v2" / "scene_graph.json").read_text(encoding="utf-8")
        )
    counts: dict[str, int] = {}
    for s in scene["structural_surfaces"]:
        counts[s["surface_type"]] = counts.get(s["surface_type"], 0) + 1
    if counts != {"floor": 1, "wall": 5, "ceiling": 1}:
        raise AssertionError(f"Replica room_0 structural counts wrong: {counts!r}")
    for s in scene["structural_surfaces"]:
        if s["source"] != "habitat_label":
            raise AssertionError(f"unexpected source on Replica: {s['surface_uid']} -> {s['source']}")
        n = s["plane"]["normal"]
        if s["surface_type"] == "floor" and n[2] <= 0.99:
            raise AssertionError(f"Replica floor normal not +z: {n!r}")
        if s["surface_type"] == "ceiling" and n[2] >= -0.99:
            raise AssertionError(f"Replica ceiling normal not -z: {n!r}")
        if s["surface_type"] == "wall" and abs(n[2]) >= 0.05:
            raise AssertionError(f"Replica wall normal not gravity-perpendicular: {n!r}")
    # Winding + plane invariants on the real geometry
    for s in scene["structural_surfaces"]:
        if _winding_dot(s["polygon"], s["plane"]["normal"]) <= 0:
            raise AssertionError(f"Replica {s['surface_uid']} winding wrong")
        for p in s["polygon"]:
            residual = abs(_dot(s["plane"]["normal"], p) + s["plane"]["d"])
            if residual >= 5e-6:
                raise AssertionError(
                    f"Replica {s['surface_uid']} corner {p!r} off plane by {residual:.2e}"
                )


TESTS = [
    test_structural_surfaces_count_per_type_synthetic,
    test_structural_surfaces_all_sourced_from_habitat_label,
    test_structural_surface_uid_schema_uses_habitat_ids,
    test_structural_surface_uids_unique,
    test_floor_and_ceiling_normals_aligned_with_gravity_axis,
    test_wall_normals_face_floor_face_centroid,
    test_polygon_winding_matches_normal,
    test_polygon_corners_on_plane,
    test_source_obb_fields_present_and_consistent,
    test_missing_structural_class_raises_systemexit,
    test_structural_surfaces_deterministic,
    test_capture_meta_records_structural_counts,
    test_real_replica_room0_has_one_floor_five_walls_one_ceiling,
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

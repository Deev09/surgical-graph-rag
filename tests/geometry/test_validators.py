"""P2.04 tests: geometry validators.

Unit tests cover each validator's happy path + at least one failure
mode. Integration tests run the validators against the actual
--enriched-v2 importer output (synthetic fixture; real Replica room_0
when the raw data is on disk).

Run: python tests/geometry/test_validators.py
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

from geometry.validators import (
    ALLOWED_SURFACE_SOURCES,
    GeometryValidationError,
    sha256_of_bytes,
    validate_canonical_surface_set,
    validate_deterministic_hash,
    validate_gravity_alignment,
    validate_obb_sanity,
    validate_plane_normalized,
    validate_surface_extents,
    validate_surface_source,
)
from importers.replica import import_replica

REPLICA_RAW = REPO_ROOT / "data" / "replica" / "room_0"
REPLICA_INFO = REPLICA_RAW / "habitat" / "info_semantic.json"


def _expect_raises(fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except GeometryValidationError:
        return
    raise AssertionError(
        f"expected GeometryValidationError from {fn.__name__}({args!r}, {kwargs!r})"
    )


def _write_synthetic_scene(root: Path) -> Path:
    """Synthetic Replica scene with the full structural set. Kept aligned
    with the P2.03 fixture."""
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


def _run_v2_synthetic(root: Path) -> tuple[dict, dict]:
    scene_dir = _write_synthetic_scene(root)
    out = root / "out"
    import_replica(scene_dir, "synthetic", out, keep_structural=False, enriched_v2=True)
    return (
        json.loads((out / "synthetic" / "enriched" / "v2" / "scene_graph.json").read_text(encoding="utf-8")),
        json.loads((out / "synthetic" / "enriched" / "v2" / "capture_meta.json").read_text(encoding="utf-8")),
    )


# --- validate_gravity_alignment -------------------------------------------


def test_gravity_alignment_accepts_canonical_negative_z() -> None:
    validate_gravity_alignment([0.0, 0.0, -1.0])


def test_gravity_alignment_accepts_near_canonical_replica_value() -> None:
    validate_gravity_alignment([0.0036, 0.0030, -0.99999])


def test_gravity_alignment_rejects_non_unit() -> None:
    _expect_raises(validate_gravity_alignment, [0.0, 0.0, -2.0])


def test_gravity_alignment_rejects_diagonal() -> None:
    _expect_raises(validate_gravity_alignment, [math.sqrt(0.5), 0.0, -math.sqrt(0.5)])


def test_gravity_alignment_rejects_non_finite() -> None:
    _expect_raises(validate_gravity_alignment, [float("nan"), 0.0, -1.0])


def test_gravity_alignment_rejects_wrong_length() -> None:
    _expect_raises(validate_gravity_alignment, [0.0, -1.0])


# --- validate_obb_sanity --------------------------------------------------


def test_obb_sanity_accepts_identity_box() -> None:
    validate_obb_sanity({
        "center": [0.0, 0.0, 0.0], "extents": [1.0, 1.0, 1.0],
        "rotation_quat": [0.0, 0.0, 0.0, 1.0],
    })


def test_obb_sanity_rejects_zero_extent() -> None:
    _expect_raises(validate_obb_sanity, {
        "center": [0.0]*3, "extents": [1.0, 0.0, 1.0],
        "rotation_quat": [0.0, 0.0, 0.0, 1.0],
    })


def test_obb_sanity_rejects_negative_extent() -> None:
    _expect_raises(validate_obb_sanity, {
        "center": [0.0]*3, "extents": [1.0, -2.0, 1.0],
        "rotation_quat": [0.0, 0.0, 0.0, 1.0],
    })


def test_obb_sanity_rejects_non_unit_quat() -> None:
    _expect_raises(validate_obb_sanity, {
        "center": [0.0]*3, "extents": [1.0]*3,
        "rotation_quat": [1.0, 0.0, 0.0, 1.0],
    })


def test_obb_sanity_rejects_non_finite_center() -> None:
    _expect_raises(validate_obb_sanity, {
        "center": [float("inf"), 0.0, 0.0], "extents": [1.0]*3,
        "rotation_quat": [0.0, 0.0, 0.0, 1.0],
    })


def test_obb_sanity_rejects_missing_key() -> None:
    _expect_raises(validate_obb_sanity, {"center": [0.0]*3, "extents": [1.0]*3})


# --- validate_plane_normalized --------------------------------------------


def test_plane_normalized_accepts_canonical_z() -> None:
    validate_plane_normalized({"normal": [0.0, 0.0, 1.0], "d": 1.5})


def test_plane_normalized_rejects_non_unit() -> None:
    _expect_raises(validate_plane_normalized, {"normal": [2.0, 0.0, 0.0], "d": 0.0})


def test_plane_normalized_rejects_non_finite_d() -> None:
    _expect_raises(validate_plane_normalized, {"normal": [0.0, 0.0, 1.0], "d": float("nan")})


def test_plane_normalized_rejects_missing_key() -> None:
    _expect_raises(validate_plane_normalized, {"normal": [0.0, 0.0, 1.0]})


# --- validate_surface_extents ---------------------------------------------


def test_surface_extents_accepts_polygon_inside_bbox() -> None:
    surface = {"surface_uid": "test", "polygon": [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    ]}
    validate_surface_extents(surface, [[-1.0, -1.0, -1.0], [2.0, 2.0, 2.0]])


def test_surface_extents_skipped_when_polygon_none() -> None:
    validate_surface_extents({"surface_uid": "test", "polygon": None},
                             [[0.0]*3, [1.0]*3])


def test_surface_extents_skipped_when_polygon_absent() -> None:
    validate_surface_extents({"surface_uid": "test"}, [[0.0]*3, [1.0]*3])


def test_surface_extents_rejects_corner_outside() -> None:
    surface = {"surface_uid": "test", "polygon": [
        [0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    ]}
    _expect_raises(validate_surface_extents, surface,
                   [[-1.0, -1.0, -1.0], [2.0, 2.0, 2.0]])


def test_surface_extents_rejects_bad_bbox_shape() -> None:
    surface = {"surface_uid": "test", "polygon": [[0.0]*3]*4}
    _expect_raises(validate_surface_extents, surface, [[0.0, 0.0], [1.0, 1.0]])


def test_surface_extents_rejects_non_finite_bbox() -> None:
    surface = {"surface_uid": "test", "polygon": [[0.0]*3]*4}
    _expect_raises(
        validate_surface_extents, surface,
        [[float("nan"), 0.0, 0.0], [1.0, 1.0, 1.0]],
    )


def test_surface_extents_rejects_inverted_bbox() -> None:
    surface = {"surface_uid": "test", "polygon": [[0.0]*3]*4}
    _expect_raises(
        validate_surface_extents, surface,
        [[2.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    )


def test_surface_extents_rejects_degenerate_polygon() -> None:
    surface = {"surface_uid": "test", "polygon": [[0.0, 0.0, 0.0]]}
    _expect_raises(
        validate_surface_extents, surface,
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
    )


# --- validate_surface_source (A7) -----------------------------------------


def test_surface_source_accepts_each_allowed_value() -> None:
    for s in ALLOWED_SURFACE_SOURCES:
        validate_surface_source({"surface_uid": "u", "source": s})


def test_surface_source_rejects_missing() -> None:
    _expect_raises(validate_surface_source, {"surface_uid": "u"})


def test_surface_source_rejects_unknown() -> None:
    _expect_raises(validate_surface_source,
                   {"surface_uid": "u", "source": "manual_authored"})


# --- validate_canonical_surface_set (A7) ----------------------------------


def test_canonical_surface_set_accepts_minimal_set() -> None:
    validate_canonical_surface_set([
        {"surface_type": "floor",   "source": "habitat_label", "surface_uid": "f"},
        {"surface_type": "wall",    "source": "habitat_label", "surface_uid": "w1"},
        {"surface_type": "wall",    "source": "mesh_ransac",   "surface_uid": "w2"},
        {"surface_type": "ceiling", "source": "habitat_label", "surface_uid": "c"},
    ])


def test_canonical_surface_set_rejects_missing_wall() -> None:
    _expect_raises(validate_canonical_surface_set, [
        {"surface_type": "floor",   "source": "habitat_label"},
        {"surface_type": "wall",    "source": "habitat_label"},
        {"surface_type": "ceiling", "source": "habitat_label"},
    ])


def test_canonical_surface_set_rejects_missing_floor() -> None:
    _expect_raises(validate_canonical_surface_set, [
        {"surface_type": "wall",    "source": "habitat_label"},
        {"surface_type": "wall",    "source": "habitat_label"},
        {"surface_type": "ceiling", "source": "habitat_label"},
    ])


def test_canonical_surface_set_rejects_synth_fallback() -> None:
    _expect_raises(validate_canonical_surface_set, [
        {"surface_type": "floor",   "source": "synth_bbox_fallback", "surface_uid": "f"},
        {"surface_type": "wall",    "source": "habitat_label",       "surface_uid": "w1"},
        {"surface_type": "wall",    "source": "habitat_label",       "surface_uid": "w2"},
        {"surface_type": "ceiling", "source": "habitat_label",       "surface_uid": "c"},
    ])


def test_canonical_surface_set_rejects_missing_surface_type() -> None:
    _expect_raises(validate_canonical_surface_set, [
        {"surface_uid": "f", "source": "habitat_label"},
    ])


def test_canonical_surface_set_rejects_unknown_surface_type() -> None:
    _expect_raises(validate_canonical_surface_set, [
        {"surface_type": "floor",   "source": "habitat_label", "surface_uid": "f"},
        {"surface_type": "wall",    "source": "habitat_label", "surface_uid": "w1"},
        {"surface_type": "wall",    "source": "habitat_label", "surface_uid": "w2"},
        {"surface_type": "ceiling", "source": "habitat_label", "surface_uid": "c"},
        {"surface_type": "door",    "source": "habitat_label", "surface_uid": "d"},
    ])


def test_canonical_surface_set_rejects_unknown_source() -> None:
    _expect_raises(validate_canonical_surface_set, [
        {"surface_type": "floor",   "source": "habitat_label",  "surface_uid": "f"},
        {"surface_type": "wall",    "source": "habitat_label",  "surface_uid": "w1"},
        {"surface_type": "wall",    "source": "manual_authored", "surface_uid": "w2"},
        {"surface_type": "ceiling", "source": "habitat_label",  "surface_uid": "c"},
    ])


# --- validate_deterministic_hash ------------------------------------------


def test_deterministic_hash_accepts_equal() -> None:
    h = sha256_of_bytes(b"hello world")
    validate_deterministic_hash(h, h)


def test_deterministic_hash_rejects_mismatch() -> None:
    _expect_raises(
        validate_deterministic_hash,
        sha256_of_bytes(b"hello world"),
        sha256_of_bytes(b"goodbye world"),
    )


def test_deterministic_hash_message_includes_context() -> None:
    try:
        validate_deterministic_hash("a"*64, "b"*64, context="importer-v2")
    except GeometryValidationError as e:
        if "importer-v2" not in str(e):
            raise AssertionError(f"context missing from error: {e}")
        return
    raise AssertionError("expected GeometryValidationError")


# --- integration: synthetic enriched-v2 output ----------------------------


def test_validators_accept_synthetic_enriched_v2_output() -> None:
    with tempfile.TemporaryDirectory() as td:
        scene, meta = _run_v2_synthetic(Path(td))
    validate_gravity_alignment(meta["axis_convention"]["gravity_dir_raw"])
    for obj in scene["objects"]:
        validate_obb_sanity(obj["bbox_obb"])
    for s in scene["structural_surfaces"]:
        validate_plane_normalized(s["plane"])
        validate_surface_source(s)
        validate_obb_sanity(s["source_obb"])
    validate_canonical_surface_set(scene["structural_surfaces"])
    # Build a consistent bbox from the actual polygon corners — capture_meta's
    # room_bbox is derived from object centroids (A3) and is NOT room
    # geometry, so we don't use it here.
    all_corners = [c for s in scene["structural_surfaces"] for c in s["polygon"]]
    lo = [min(c[i] for c in all_corners) for i in range(3)]
    hi = [max(c[i] for c in all_corners) for i in range(3)]
    for s in scene["structural_surfaces"]:
        validate_surface_extents(s, [lo, hi])


def test_validators_catch_tampered_synthetic_source() -> None:
    with tempfile.TemporaryDirectory() as td:
        scene, _ = _run_v2_synthetic(Path(td))
    # An unknown source (e.g. 'manual_authored') is caught by the per-surface
    # validator but not by the canonical-set validator, which bans only
    # synth_bbox_fallback. Separate tampering exercises both gates.
    surface = scene["structural_surfaces"][0]
    surface["source"] = "manual_authored"
    _expect_raises(validate_surface_source, surface)

    with tempfile.TemporaryDirectory() as td:
        scene2, _ = _run_v2_synthetic(Path(td))
    scene2["structural_surfaces"][0]["source"] = "synth_bbox_fallback"
    _expect_raises(validate_canonical_surface_set, scene2["structural_surfaces"])


def test_validators_catch_tampered_synthetic_plane() -> None:
    with tempfile.TemporaryDirectory() as td:
        scene, _ = _run_v2_synthetic(Path(td))
    scene["structural_surfaces"][0]["plane"]["normal"] = [2.0, 0.0, 0.0]
    _expect_raises(validate_plane_normalized, scene["structural_surfaces"][0]["plane"])


def test_validators_synthetic_runs_are_byte_equal() -> None:
    with tempfile.TemporaryDirectory() as td_a, tempfile.TemporaryDirectory() as td_b:
        scene_a, meta_a = _run_v2_synthetic(Path(td_a))
        scene_b, meta_b = _run_v2_synthetic(Path(td_b))
    pa = json.dumps(scene_a, sort_keys=True).encode()
    pb = json.dumps(scene_b, sort_keys=True).encode()
    validate_deterministic_hash(
        sha256_of_bytes(pa), sha256_of_bytes(pb),
        context="synthetic enriched-v2 scene_graph.json",
    )
    ma = json.dumps(meta_a, sort_keys=True).encode()
    mb = json.dumps(meta_b, sort_keys=True).encode()
    validate_deterministic_hash(
        sha256_of_bytes(ma), sha256_of_bytes(mb),
        context="synthetic enriched-v2 capture_meta.json",
    )


# --- integration: real Replica room_0 (skip if raw data absent) -----------


def test_validators_accept_real_replica_room0() -> None:
    if not REPLICA_INFO.exists():
        print("  SKIP (raw Replica data not on disk)")
        return
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out"
        import_replica(REPLICA_RAW, "replica_room_0", out, keep_structural=False, enriched_v2=True)
        scene = json.loads(
            (out / "replica_room_0" / "enriched" / "v2" / "scene_graph.json").read_text(encoding="utf-8")
        )
        meta = json.loads(
            (out / "replica_room_0" / "enriched" / "v2" / "capture_meta.json").read_text(encoding="utf-8")
        )
    validate_gravity_alignment(meta["axis_convention"]["gravity_dir_raw"])
    for obj in scene["objects"]:
        validate_obb_sanity(obj["bbox_obb"])
    for s in scene["structural_surfaces"]:
        validate_plane_normalized(s["plane"])
        validate_surface_source(s)
        validate_obb_sanity(s["source_obb"])
    validate_canonical_surface_set(scene["structural_surfaces"])


TESTS = [
    test_gravity_alignment_accepts_canonical_negative_z,
    test_gravity_alignment_accepts_near_canonical_replica_value,
    test_gravity_alignment_rejects_non_unit,
    test_gravity_alignment_rejects_diagonal,
    test_gravity_alignment_rejects_non_finite,
    test_gravity_alignment_rejects_wrong_length,
    test_obb_sanity_accepts_identity_box,
    test_obb_sanity_rejects_zero_extent,
    test_obb_sanity_rejects_negative_extent,
    test_obb_sanity_rejects_non_unit_quat,
    test_obb_sanity_rejects_non_finite_center,
    test_obb_sanity_rejects_missing_key,
    test_plane_normalized_accepts_canonical_z,
    test_plane_normalized_rejects_non_unit,
    test_plane_normalized_rejects_non_finite_d,
    test_plane_normalized_rejects_missing_key,
    test_surface_extents_accepts_polygon_inside_bbox,
    test_surface_extents_skipped_when_polygon_none,
    test_surface_extents_skipped_when_polygon_absent,
    test_surface_extents_rejects_corner_outside,
    test_surface_extents_rejects_bad_bbox_shape,
    test_surface_extents_rejects_non_finite_bbox,
    test_surface_extents_rejects_inverted_bbox,
    test_surface_extents_rejects_degenerate_polygon,
    test_surface_source_accepts_each_allowed_value,
    test_surface_source_rejects_missing,
    test_surface_source_rejects_unknown,
    test_canonical_surface_set_accepts_minimal_set,
    test_canonical_surface_set_rejects_missing_wall,
    test_canonical_surface_set_rejects_missing_floor,
    test_canonical_surface_set_rejects_synth_fallback,
    test_canonical_surface_set_rejects_missing_surface_type,
    test_canonical_surface_set_rejects_unknown_surface_type,
    test_canonical_surface_set_rejects_unknown_source,
    test_deterministic_hash_accepts_equal,
    test_deterministic_hash_rejects_mismatch,
    test_deterministic_hash_message_includes_context,
    test_validators_accept_synthetic_enriched_v2_output,
    test_validators_catch_tampered_synthetic_source,
    test_validators_catch_tampered_synthetic_plane,
    test_validators_synthetic_runs_are_byte_equal,
    test_validators_accept_real_replica_room0,
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

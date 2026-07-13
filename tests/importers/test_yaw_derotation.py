"""Phase 8 F1/F3 tests: guarded yaw de-rotation in the habitat importer.

Run: python tests/importers/test_yaw_derotation.py

Covers the estimator (90-deg-symmetric stats, length weighting) plus a
synthetic end-to-end import: a room rotated 20 deg about z must import with
yaw_derotation_deg ~= 20, axis-aligned walls, and TIGHT object AABBs, while a
near-axis room (2 deg) must import unrotated (the guard).
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

from common.types import Plane
from extractors.base import StructuralSurface
from demo.replica_habitat_import import (
    YAW_DEROTATE_GUARD_DEG,
    _dominant_yaw_deg,
    import_habitat_room,
)


def _wall_at(theta_deg: float, length: float) -> StructuralSurface:
    t = math.radians(theta_deg)
    a, b = math.cos(t), math.sin(t)
    # a horizontal segment of the given length, perpendicular to the normal
    dx, dy = -b * length / 2.0, a * length / 2.0
    poly = [(-dx, -dy, 0.0), (dx, dy, 0.0), (dx, dy, 2.0), (-dx, -dy, 2.0)]
    return StructuralSurface(
        surface_uid=f"wall_{theta_deg}_{length}", surface_type="wall",
        plane=Plane(a=a, b=b, c=0.0, d=0.0), polygon=poly,
        confidence=1.0, source="habitat_label")


def _quat_z(deg: float):
    r = math.radians(deg) / 2.0
    return [0.0, 0.0, math.sin(r), math.cos(r)]


def _synthetic_room(yaw_deg: float) -> dict:
    """info_semantic.json dict: floor + 4 walls + 1 box, all rotated yaw_deg."""
    def inst(iid, cls, center, sizes, translation=(0.0, 0.0, 0.0)):
        return {
            "id": iid, "class_name": cls,
            "oriented_bbox": {
                "abb": {"center": list(center), "sizes": list(sizes)},
                "orientation": {"rotation": _quat_z(yaw_deg),
                                "translation": list(translation)},
            },
        }
    objects = [
        inst(1, "floor", (0.0, 0.0, 0.0), (6.0, 6.0, 0.05)),
        inst(2, "wall", (3.0, 0.0, 1.0), (0.05, 6.0, 2.0)),
        inst(3, "wall", (-3.0, 0.0, 1.0), (0.05, 6.0, 2.0)),
        inst(4, "wall", (0.0, 3.0, 1.0), (6.0, 0.05, 2.0)),
        inst(5, "wall", (0.0, -3.0, 1.0), (6.0, 0.05, 2.0)),
        # a 1.0 x 0.4 box standing just above the floor, room-aligned
        inst(6, "table", (0.0, 0.0, 0.30), (1.0, 0.4, 0.5)),
    ]
    return {"gravity_dir": [0.0, 0.0, -1.0], "objects": objects}


def _import_synthetic(yaw_deg: float):
    with tempfile.TemporaryDirectory() as td:
        room = Path(td) / "room"
        (room / "habitat").mkdir(parents=True)
        (room / "habitat" / "info_semantic.json").write_text(
            json.dumps(_synthetic_room(yaw_deg)), encoding="utf-8")
        return import_habitat_room(room, "synthetic", z_translation=0.0)


def test_estimator_rectangle():
    walls = [_wall_at(20.0, 6.0), _wall_at(110.0, 4.0),
             _wall_at(200.0, 6.0), _wall_at(290.0, 4.0)]
    yaw = _dominant_yaw_deg(walls)
    if abs(yaw - 20.0) > 0.1:
        raise AssertionError(f"rectangle at 20 deg must estimate 20: {yaw}")


def test_estimator_length_weighting():
    # a long facade at 10 deg outvotes a short angled nook at 40 deg
    walls = [_wall_at(10.0, 8.0), _wall_at(100.0, 8.0), _wall_at(40.0, 1.0)]
    yaw = _dominant_yaw_deg(walls)
    if abs(yaw - 10.0) > 3.0:
        raise AssertionError(f"long walls must dominate: {yaw}")


def test_rotated_room_derotates_and_tightens_aabbs():
    arts = _import_synthetic(20.0)
    yaw = arts.notes["yaw_derotation_deg"]
    if abs(abs(yaw) - 20.0) > 0.1:
        raise AssertionError(f"expected ~20 deg de-rotation: {yaw}")
    for w in (s for s in arts.structural_surfaces if s.surface_type == "wall"):
        off = math.degrees(math.atan2(w.plane.b, w.plane.a)) % 90.0
        off = min(off, 90.0 - off)
        if off > 0.1:
            raise AssertionError(f"wall not axis-aligned after de-rotation: {off}")
    (lo, hi), = [e.bbox_aabb for e in arts.entities]
    dx, dy = hi[0] - lo[0], hi[1] - lo[1]
    # without de-rotation the 1.0 x 0.4 box inflates to ~1.08 x 0.72
    if abs(dx - 1.0) > 0.01 or abs(dy - 0.4) > 0.01:
        raise AssertionError(f"AABB must be tight after de-rotation: {dx} x {dy}")


def test_near_axis_room_untouched():
    arts = _import_synthetic(2.0)
    if arts.notes["yaw_derotation_deg"] != 0.0:
        raise AssertionError(
            f"2 deg is under the {YAW_DEROTATE_GUARD_DEG} deg guard: {arts.notes}")
    (lo, hi), = [e.bbox_aabb for e in arts.entities]
    dx = hi[0] - lo[0]
    if not dx > 1.0:  # mild inflation preserved = pipeline unchanged
        raise AssertionError(f"near-axis import must be byte-stable, got dx={dx}")


TESTS = [
    test_estimator_rectangle,
    test_estimator_length_weighting,
    test_rotated_room_derotates_and_tightens_aabbs,
    test_near_axis_room_untouched,
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

"""Phase 8 F2 tests: guarded floor-plane calibration in the habitat importer.

Run: python tests/importers/test_floor_calibration.py

Synthetic-only (no dataset): builds floors + entities directly and calls
_calibrate_floor_planes. The properties under test:
  1. gross penetration (>0.10 m) snaps the plane AND polygon down;
  2. mild disagreement (the room_0 / apartment_0 regime) is untouched;
  3. objects on another story (outside the 0.5 m window) never vote;
  4. fewer than 3 voters -> no action;
  5. walls are never touched.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.types import Plane
from extractors.base import EntityArtifact, EntityIdentity, StructuralSurface
from demo.replica_habitat_import import (
    FLOOR_CAL_GUARD_M,
    _calibrate_floor_planes,
    _point_in_convex_poly_xy,
)


def _floor(z: float, uid: str = "floor_1") -> StructuralSurface:
    return StructuralSurface(
        surface_uid=uid, surface_type="floor",
        plane=Plane(a=0.0, b=0.0, c=1.0, d=-z),
        polygon=[(0.0, 0.0, z), (4.0, 0.0, z), (4.0, 4.0, z), (0.0, 4.0, z)],
        confidence=1.0, source="habitat_label")


def _wall() -> StructuralSurface:
    return StructuralSurface(
        surface_uid="wall_1", surface_type="wall",
        plane=Plane(a=0.0, b=-1.0, c=0.0, d=4.0),
        polygon=[(0.0, 4.0, 0.0), (4.0, 4.0, 0.0), (4.0, 4.0, 2.0), (0.0, 4.0, 2.0)],
        confidence=1.0, source="habitat_label")


def _entity(uid: str, bottom_z: float, x: float = 1.0, y: float = 1.0) -> EntityArtifact:
    lo = (x, y, bottom_z)
    hi = (x + 0.4, y + 0.4, bottom_z + 0.5)
    return EntityArtifact(
        identity=EntityIdentity(object_uid=uid, display_label=uid,
                                aliases=[], source_instance_ref=uid),
        bbox_aabb=(lo, hi), bbox_obb=None,
        centroid=tuple((lo[i] + hi[i]) / 2.0 for i in range(3)),
        geometry_handle=None, semantic_hypotheses=[],
        extraction_diagnostics={})


def test_gross_penetration_shifts_plane_and_polygon():
    # floor claims z=0.30 while three objects stand at z=0.19 (0.11 m through)
    floor = _floor(0.30)
    ents = [_entity(f"obj_{i}", 0.19, x=0.5 + i) for i in range(3)]
    out, applied = _calibrate_floor_planes([floor], ents)
    if applied != {"floor_1": -0.11}:
        raise AssertionError(f"expected -0.11 shift: {applied}")
    cal = out[0]
    new_z = -cal.plane.d / cal.plane.c
    if abs(new_z - 0.19) > 1e-9:
        raise AssertionError(f"plane not snapped to 0.19: z={new_z}")
    if any(abs(q[2] - 0.19) > 1e-9 for q in cal.polygon):
        raise AssertionError(f"polygon must move with the plane: {cal.polygon}")


def test_mild_offset_untouched():
    # the room_0 / apartment_0 regime: bottoms 0.04 m through -> keep
    floor = _floor(0.30)
    ents = [_entity(f"obj_{i}", 0.26, x=0.5 + i) for i in range(4)]
    out, applied = _calibrate_floor_planes([floor], ents)
    if applied or out[0] is not floor:
        raise AssertionError(f"mild offset must be left alone: {applied}")
    if abs(0.04) > FLOOR_CAL_GUARD_M:
        raise AssertionError("test premise broken: 0.04 must be under the guard")


def test_other_story_does_not_vote():
    # upper-story floor at z=3.0; three lower-story objects sit at z=0.2
    # directly under it in XY. Without the window they'd drag the floor down
    # 2.8 m; with it they are ignored and nothing shifts.
    floor = _floor(3.0)
    ents = [_entity(f"obj_{i}", 0.2, x=0.5 + i) for i in range(3)]
    out, applied = _calibrate_floor_planes([floor], ents)
    if applied:
        raise AssertionError(f"cross-story voters must be excluded: {applied}")


def test_too_few_voters_no_action():
    floor = _floor(0.30)
    ents = [_entity("obj_0", 0.15), _entity("obj_outside", 0.15, x=9.0)]
    out, applied = _calibrate_floor_planes([floor], ents)
    if applied:
        raise AssertionError(f"2 voters must not trigger a shift: {applied}")


def test_walls_never_touched():
    wall = _wall()
    ents = [_entity(f"obj_{i}", 0.19) for i in range(3)]
    out, applied = _calibrate_floor_planes([wall], ents)
    if out[0] is not wall or applied:
        raise AssertionError("non-floor surfaces must pass through untouched")


def test_point_in_poly():
    poly = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 4.0, 0.0), (0.0, 4.0, 0.0)]
    if not _point_in_convex_poly_xy(2.0, 2.0, poly):
        raise AssertionError("interior point must be inside")
    if _point_in_convex_poly_xy(5.0, 2.0, poly):
        raise AssertionError("exterior point must be outside")


TESTS = [
    test_gross_penetration_shifts_plane_and_polygon,
    test_mild_offset_untouched,
    test_other_story_does_not_vote,
    test_too_few_voters_no_action,
    test_walls_never_touched,
    test_point_in_poly,
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

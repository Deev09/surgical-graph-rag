"""P2.07 tests: surface-distance helpers.

Each function has happy-path + property-style + named-regression tests.
The A4-rejected min-on-dominant-axis behavior is explicitly tested
against in `test_aabb_to_aabb_surface_diagonal_separation_regression`.

Run: python tests/geometry/test_surface_distance.py
"""
from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.types import Plane
from geometry.surface_distance import (
    aabb_to_aabb_surface,
    bbox_to_plane,
    point_to_aabb,
    point_to_plane,
)


def _close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) < tol


def _assert_value_error(fn, *args) -> None:
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# --- point_to_plane (signed) ----------------------------------------------


def test_point_to_plane_on_plane_returns_zero() -> None:
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    if not _close(point_to_plane((1.0, 2.0, 0.0), plane), 0.0):
        raise AssertionError("point on plane should give 0")


def test_point_to_plane_above_is_positive() -> None:
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    if not _close(point_to_plane((0.0, 0.0, 3.0), plane), 3.0):
        raise AssertionError("point above plane wrong sign / magnitude")


def test_point_to_plane_below_is_negative() -> None:
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    if not _close(point_to_plane((0.0, 0.0, -2.5), plane), -2.5):
        raise AssertionError("point below plane wrong sign / magnitude")


def test_point_to_plane_offset_plane() -> None:
    # Plane z = 2 → normal (0,0,1), d = -2
    plane = Plane(a=0.0, b=0.0, c=1.0, d=-2.0)
    if not _close(point_to_plane((0.0, 0.0, 2.0), plane), 0.0):
        raise AssertionError("point on offset plane should give 0")
    if not _close(point_to_plane((0.0, 0.0, 5.0), plane), 3.0):
        raise AssertionError("point above offset plane wrong distance")


def test_point_to_plane_tilted_plane() -> None:
    # Plane x + y = 1 in normalized form: normal=(1,1,0)/sqrt(2), d=-1/sqrt(2)
    s = 1.0 / math.sqrt(2.0)
    plane = Plane(a=s, b=s, c=0.0, d=-s)
    if not _close(point_to_plane((1.0, 0.0, 0.0), plane), 0.0):
        raise AssertionError("point on tilted plane should give 0")
    expected = s  # signed distance from (1,1,0) to plane x+y=1: (1+1-1)/sqrt(2)
    if not _close(point_to_plane((1.0, 1.0, 0.0), plane), expected):
        raise AssertionError("point off tilted plane wrong distance")


def test_point_to_plane_rejects_non_finite_point() -> None:
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    _assert_value_error(point_to_plane, (0.0, 0.0, math.nan), plane)


# --- point_to_aabb (non-negative) -----------------------------------------


def test_point_to_aabb_inside_is_zero() -> None:
    aabb = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    if not _close(point_to_aabb((0.0, 0.0, 0.0), aabb), 0.0):
        raise AssertionError("point inside box should give 0")


def test_point_to_aabb_on_face_boundary_is_zero() -> None:
    aabb = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    if not _close(point_to_aabb((1.0, 0.0, 0.0), aabb), 0.0):
        raise AssertionError("point on face should give 0")


def test_point_to_aabb_on_corner_is_zero() -> None:
    aabb = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    if not _close(point_to_aabb((-1.0, -1.0, 1.0), aabb), 0.0):
        raise AssertionError("point on corner should give 0")


def test_point_to_aabb_single_axis_outside() -> None:
    aabb = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    if not _close(point_to_aabb((2.0, 0.5, 0.5), aabb), 1.0):
        raise AssertionError("axial-only excess wrong distance")


def test_point_to_aabb_diagonal_corner_outside() -> None:
    """Point (2,2,2) outside unit cube at origin: excess (1,1,1) → sqrt(3)."""
    aabb = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    expected = math.sqrt(3.0)
    if not _close(point_to_aabb((2.0, 2.0, 2.0), aabb), expected):
        raise AssertionError(f"diagonal point wrong distance, expected {expected}")


def test_point_to_aabb_negative_excess_axis() -> None:
    """Point below the box on x-axis (lo - p > 0)."""
    aabb = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    if not _close(point_to_aabb((-3.0, 0.5, 0.5), aabb), 3.0):
        raise AssertionError("negative-side excess wrong distance")


def test_point_to_aabb_rejects_inverted_bounds() -> None:
    _assert_value_error(
        point_to_aabb,
        (0.0, 0.0, 0.0),
        ((0.0, 0.0, 1.0), (1.0, 1.0, 0.0)),
    )


# --- aabb_to_aabb_surface (non-negative, Euclidean) -----------------------


def test_aabb_to_aabb_surface_overlapping_is_zero() -> None:
    a = ((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    b = ((1.0, 1.0, 1.0), (3.0, 3.0, 3.0))
    if not _close(aabb_to_aabb_surface(a, b), 0.0):
        raise AssertionError("overlapping boxes should give 0")


def test_aabb_to_aabb_surface_touching_face_is_zero() -> None:
    a = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    b = ((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))  # share x=1 face
    if not _close(aabb_to_aabb_surface(a, b), 0.0):
        raise AssertionError("face-touching boxes should give 0")


def test_aabb_to_aabb_surface_separated_one_axis() -> None:
    a = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    b = ((3.0, 0.0, 0.0), (4.0, 1.0, 1.0))  # gap of 2 on x
    if not _close(aabb_to_aabb_surface(a, b), 2.0):
        raise AssertionError("axial separation wrong distance")


def test_aabb_to_aabb_surface_diagonal_separation_regression() -> None:
    """A4 regression. Two unit cubes at (0,0,0) and (2,2,2) — gap of 1 on
    each axis. Correct: sqrt(3) ≈ 1.732. Rejected min-on-dominant-axis
    would return 1."""
    a = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    b = ((2.0, 2.0, 2.0), (3.0, 3.0, 3.0))
    expected = math.sqrt(3.0)
    actual = aabb_to_aabb_surface(a, b)
    if not _close(actual, expected, tol=1e-12):
        raise AssertionError(
            f"diagonal separation: expected {expected}, got {actual}"
        )


def test_aabb_to_aabb_surface_is_symmetric() -> None:
    a = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    b = ((2.0, 2.0, 2.0), (3.0, 3.0, 3.0))
    if not _close(aabb_to_aabb_surface(a, b), aabb_to_aabb_surface(b, a)):
        raise AssertionError("aabb_to_aabb_surface should be symmetric")


def test_aabb_to_aabb_surface_two_axis_separation() -> None:
    """Box separation on x=1, y=1, but z overlaps: distance = sqrt(2)."""
    a = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    b = ((2.0, 2.0, 0.5), (3.0, 3.0, 1.5))
    expected = math.sqrt(2.0)
    if not _close(aabb_to_aabb_surface(a, b), expected):
        raise AssertionError(f"two-axis separation wrong: expected {expected}")


def test_aabb_to_aabb_surface_rejects_non_finite_bounds() -> None:
    a = ((0.0, 0.0, 0.0), (1.0, 1.0, math.inf))
    b = ((2.0, 2.0, 2.0), (3.0, 3.0, 3.0))
    _assert_value_error(aabb_to_aabb_surface, a, b)


# --- bbox_to_plane (non-negative) -----------------------------------------


def test_bbox_to_plane_intersecting_is_zero() -> None:
    aabb = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)  # z=0
    if not _close(bbox_to_plane(aabb, plane), 0.0):
        raise AssertionError("plane-bisecting-box should give 0")


def test_bbox_to_plane_touching_face_is_zero() -> None:
    aabb = ((-1.0, -1.0, 2.0), (1.0, 1.0, 3.0))  # bottom face at z=2
    plane = Plane(a=0.0, b=0.0, c=1.0, d=-2.0)  # z=2
    if not _close(bbox_to_plane(aabb, plane), 0.0):
        raise AssertionError("face-touching plane should give 0")


def test_bbox_to_plane_touching_corner_is_zero() -> None:
    """Plane through a single corner: still touches → 0."""
    aabb = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    # Diagonal plane through origin: normal (1,1,1)/sqrt(3), d=0.
    s = 1.0 / math.sqrt(3.0)
    plane = Plane(a=s, b=s, c=s, d=0.0)
    if not _close(bbox_to_plane(aabb, plane), 0.0):
        raise AssertionError("corner-touching plane should give 0")


def test_bbox_to_plane_box_entirely_above() -> None:
    aabb = ((-1.0, -1.0, 2.0), (1.0, 1.0, 3.0))
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)  # z=0
    if not _close(bbox_to_plane(aabb, plane), 2.0):
        raise AssertionError("above-plane distance wrong (expected 2.0)")


def test_bbox_to_plane_box_entirely_below() -> None:
    aabb = ((-1.0, -1.0, -3.0), (1.0, 1.0, -2.0))
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)  # z=0
    if not _close(bbox_to_plane(aabb, plane), 2.0):
        raise AssertionError("below-plane distance wrong (expected 2.0)")


def test_bbox_to_plane_axis_aligned_x() -> None:
    aabb = ((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    plane = Plane(a=1.0, b=0.0, c=0.0, d=0.0)  # x=0
    if not _close(bbox_to_plane(aabb, plane), 1.0):
        raise AssertionError("x-axis plane distance wrong (expected 1.0)")


def test_bbox_to_plane_negative_normal_direction() -> None:
    """Box at z=2..3, plane normal -z: signed distances at corners are
    negative, but distance is positive."""
    aabb = ((-1.0, -1.0, 2.0), (1.0, 1.0, 3.0))
    plane = Plane(a=0.0, b=0.0, c=-1.0, d=0.0)  # normal pointing -z, plane z=0
    # Corner signed distances: -z values = -2 and -3; sd_max = -2, sd_min = -3.
    # Both < 0 → return -sd_max = 2.
    if not _close(bbox_to_plane(aabb, plane), 2.0):
        raise AssertionError("negative-normal direction wrong distance")


def test_bbox_to_plane_rejects_inverted_bounds() -> None:
    aabb = ((0.0, 1.0, 0.0), (1.0, 0.0, 1.0))
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    _assert_value_error(bbox_to_plane, aabb, plane)


TESTS = [
    test_point_to_plane_on_plane_returns_zero,
    test_point_to_plane_above_is_positive,
    test_point_to_plane_below_is_negative,
    test_point_to_plane_offset_plane,
    test_point_to_plane_tilted_plane,
    test_point_to_plane_rejects_non_finite_point,
    test_point_to_aabb_inside_is_zero,
    test_point_to_aabb_on_face_boundary_is_zero,
    test_point_to_aabb_on_corner_is_zero,
    test_point_to_aabb_single_axis_outside,
    test_point_to_aabb_diagonal_corner_outside,
    test_point_to_aabb_negative_excess_axis,
    test_point_to_aabb_rejects_inverted_bounds,
    test_aabb_to_aabb_surface_overlapping_is_zero,
    test_aabb_to_aabb_surface_touching_face_is_zero,
    test_aabb_to_aabb_surface_separated_one_axis,
    test_aabb_to_aabb_surface_diagonal_separation_regression,
    test_aabb_to_aabb_surface_is_symmetric,
    test_aabb_to_aabb_surface_two_axis_separation,
    test_aabb_to_aabb_surface_rejects_non_finite_bounds,
    test_bbox_to_plane_intersecting_is_zero,
    test_bbox_to_plane_touching_face_is_zero,
    test_bbox_to_plane_touching_corner_is_zero,
    test_bbox_to_plane_box_entirely_above,
    test_bbox_to_plane_box_entirely_below,
    test_bbox_to_plane_axis_aligned_x,
    test_bbox_to_plane_negative_normal_direction,
    test_bbox_to_plane_rejects_inverted_bounds,
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

"""Phase 2 P2.07 — pure surface-distance helpers.

Four functions, all pure:

  point_to_plane(p, plane)       — SIGNED scalar; sign matches plane.normal.
  point_to_aabb(p, aabb)         — non-negative; 0 inside or on the box.
  aabb_to_aabb_surface(a, b)     — non-negative; 0 on overlap or touch;
                                   Euclidean norm of positive axis gaps.
  bbox_to_plane(aabb, plane)     — non-negative; 0 when the plane intersects
                                   or touches the box.

Math conventions per A4 / P2.07 sign-off:

  - `point_to_aabb` uses `sqrt(sum(max(0, lo[i] - p[i], p[i] - hi[i])**2))`
    so an inside point contributes 0 on every axis.
  - `aabb_to_aabb_surface` uses the Euclidean norm of positive axis gaps,
    NOT min-on-dominant-axis (rejected: two unit cubes at (0,0,0) and
    (2,2,2) must return sqrt(3) ≈ 1.732, not 1).
  - `bbox_to_plane` is non-negative; returns 0 when min and max corner
    signed distances straddle zero. The naive "min over the 8 corners"
    silently goes negative on intersection; this version doesn't.

Out of scope (deferred per the P2.07 sign-off):
  - point_to_obb — not added until Phase 3 needs it.
  - obb_to_obb_surface — deferred; the closest-corner approximation was
    rejected as a quality metric (A5). Phase 3 may add SAT/GJK if needed.
"""
from __future__ import annotations

import math

from common.types import Plane, Vec3


def _validate_vec3(name: str, value: Vec3) -> None:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly 3 coordinates")
    if not all(math.isfinite(component) for component in value):
        raise ValueError(f"{name} coordinates must be finite")


def _validate_aabb(name: str, aabb: tuple[Vec3, Vec3]) -> None:
    if len(aabb) != 2:
        raise ValueError(f"{name} must contain (lo, hi) bounds")
    lo, hi = aabb
    _validate_vec3(f"{name}.lo", lo)
    _validate_vec3(f"{name}.hi", hi)
    for axis in range(3):
        if lo[axis] > hi[axis]:
            raise ValueError(f"{name} has inverted bounds on axis {axis}")


def point_to_plane(point: Vec3, plane: Plane) -> float:
    """Signed scalar distance from `point` to `plane`. Positive when point
    is on the side that the plane's normal points to; negative on the
    back side; zero when on the plane.

    Assumes plane.normal is unit-norm (validated upstream by
    geometry.validators.validate_plane_normalized). Returns
    `a*x + b*y + c*z + d` directly."""
    _validate_vec3("point", point)
    return (
        plane.a * point[0]
        + plane.b * point[1]
        + plane.c * point[2]
        + plane.d
    )


def point_to_aabb(point: Vec3, aabb: tuple[Vec3, Vec3]) -> float:
    """Non-negative shortest Euclidean distance from `point` to the
    closed AABB `(lo, hi)`. Returns 0 when the point is inside or on
    the box boundary."""
    _validate_vec3("point", point)
    _validate_aabb("aabb", aabb)
    lo, hi = aabb
    sq_sum = 0.0
    for i in range(3):
        excess = max(0.0, lo[i] - point[i], point[i] - hi[i])
        sq_sum += excess * excess
    return math.sqrt(sq_sum)


def aabb_to_aabb_surface(
    a: tuple[Vec3, Vec3], b: tuple[Vec3, Vec3],
) -> float:
    """Non-negative shortest Euclidean distance between two closed AABBs.
    Returns 0 when the boxes overlap or share a face/edge/corner.

    Implemented as the Euclidean norm of per-axis positive gaps:
    `sqrt(sum(max(0, a.lo[i] - b.hi[i], b.lo[i] - a.hi[i])**2))`. This is
    exact (rejected alternative: min-on-dominant-axis, which is wrong
    for diagonal separations — see the diagonal regression test)."""
    _validate_aabb("a", a)
    _validate_aabb("b", b)
    a_lo, a_hi = a
    b_lo, b_hi = b
    sq_sum = 0.0
    for i in range(3):
        gap = max(0.0, a_lo[i] - b_hi[i], b_lo[i] - a_hi[i])
        sq_sum += gap * gap
    return math.sqrt(sq_sum)


def bbox_to_plane(
    aabb: tuple[Vec3, Vec3], plane: Plane,
) -> float:
    """Non-negative shortest distance from a closed AABB to the plane.
    Returns 0 when the plane intersects or touches the box (any signed
    corner distance crosses zero).

    Computed by checking the 8 corner signed distances:
      - if `min <= 0 <= max`, the plane intersects → return 0,
      - if all corners are strictly above (min > 0), the closest corner
        is `min`,
      - if all corners are strictly below (max < 0), the closest corner
        distance is `|max|`.
    """
    _validate_aabb("aabb", aabb)
    lo, hi = aabb
    sd_min = math.inf
    sd_max = -math.inf
    for sx in (lo[0], hi[0]):
        for sy in (lo[1], hi[1]):
            for sz in (lo[2], hi[2]):
                signed = plane.a * sx + plane.b * sy + plane.c * sz + plane.d
                if signed < sd_min:
                    sd_min = signed
                if signed > sd_max:
                    sd_max = signed
    if sd_min <= 0.0 <= sd_max:
        return 0.0
    if sd_min > 0.0:
        return sd_min
    return -sd_max

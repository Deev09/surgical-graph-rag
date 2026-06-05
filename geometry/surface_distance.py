"""Phase 2 P2.07 + Phase 3 P3.01 — pure surface-distance helpers.

Phase 2 helpers (frozen, byte-equal):

  point_to_plane(p, plane)       — SIGNED scalar; sign matches plane.normal.
  point_to_aabb(p, aabb)         — non-negative; 0 inside or on the box.
  aabb_to_aabb_surface(a, b)     — non-negative; 0 on overlap or touch;
                                   Euclidean norm of positive axis gaps.
  bbox_to_plane(aabb, plane)     — non-negative; 0 when the plane intersects
                                   or touches the box.

Phase 3 helper (P3.01, added 2026-06-04):

  aabb_to_polygon_planar(aabb, plane, polygon) -> float
      Generic 2D polygon-vs-polygon distance after projecting an AABB and
      a planar polygon into a common plane-local 2D basis. Returns the
      IN-PLANE 2D gap only — NOT the combined 3D distance. The dispatcher
      `bbox_to_surface` (P3.02) combines this with `bbox_to_plane` via
      `hypot` to produce the final 3D distance.

Math conventions per A4 / P2.07 / P3.01 / A2 sign-off:

  - `point_to_aabb` uses `sqrt(sum(max(0, lo[i] - p[i], p[i] - hi[i])**2))`
    so an inside point contributes 0 on every axis.
  - `aabb_to_aabb_surface` uses the Euclidean norm of positive axis gaps,
    NOT min-on-dominant-axis (rejected: two unit cubes at (0,0,0) and
    (2,2,2) must return sqrt(3) ≈ 1.732, not 1).
  - `bbox_to_plane` is non-negative; returns 0 when min and max corner
    signed distances straddle zero. The naive "min over the 8 corners"
    silently goes negative on intersection; this version doesn't.
  - `aabb_to_polygon_planar` is non-negative; returns the in-plane gap
    only. It is NOT dimensionally comparable to `bbox_to_plane`. The
    algorithm is GENERIC polygon-vs-polygon (not rectangle-specialized
    per A2): project 8 AABB corners → convex hull A in 2D; polygon-as-
    region distance to B = 0 on overlap, else min vertex-to-edge in
    both directions. Winding-agnostic: ray-casting containment + segment
    intersection do not depend on polygon orientation.

Out of scope (deferred per the P2.07 / Phase 3 plan):
  - point_to_obb — not added until a phase needs it.
  - obb_to_obb_surface — deferred; the closest-corner approximation was
    rejected as a quality metric (A5). A later phase may add SAT/GJK if
    needed.
"""
from __future__ import annotations

import math

from common.types import Plane, Vec3


POLYGON_ON_PLANE_TOL_M = 1e-6


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


def _validate_plane_finite(name: str, plane: Plane) -> None:
    for label, value in (("a", plane.a), ("b", plane.b), ("c", plane.c), ("d", plane.d)):
        if not math.isfinite(value):
            raise ValueError(f"{name}.{label} must be finite")
    if plane.a == 0.0 and plane.b == 0.0 and plane.c == 0.0:
        raise ValueError(f"{name} has zero normal")


def _validate_polygon_on_plane(
    name: str, polygon: list[Vec3], plane: Plane,
) -> None:
    if len(polygon) < 3:
        raise ValueError(f"{name} must have at least 3 vertices")
    for i, vertex in enumerate(polygon):
        _validate_vec3(f"{name}[{i}]", vertex)
        signed = (
            plane.a * vertex[0]
            + plane.b * vertex[1]
            + plane.c * vertex[2]
            + plane.d
        )
        if abs(signed) > POLYGON_ON_PLANE_TOL_M:
            raise ValueError(
                f"{name}[{i}] is not on plane (signed distance {signed!r} "
                f"exceeds tolerance {POLYGON_ON_PLANE_TOL_M})"
            )


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


# --- Phase 3 P3.01 — polygon-clipped planar distance ----------------------

Vec2 = tuple[float, float]


def _aabb_corners(aabb: tuple[Vec3, Vec3]) -> list[Vec3]:
    lo, hi = aabb
    return [
        (lo[0], lo[1], lo[2]),
        (hi[0], lo[1], lo[2]),
        (lo[0], hi[1], lo[2]),
        (hi[0], hi[1], lo[2]),
        (lo[0], lo[1], hi[2]),
        (hi[0], lo[1], hi[2]),
        (lo[0], hi[1], hi[2]),
        (hi[0], hi[1], hi[2]),
    ]


def _normalize3(v: Vec3) -> Vec3:
    mag = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if mag == 0.0:
        raise ValueError("cannot normalize zero vector")
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def _cross3(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot3(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _plane_basis(plane: Plane) -> tuple[Vec3, Vec3]:
    """Return an orthonormal (u, v) basis spanning the plane.

    Deterministic tie-break for `t`: pick the world axis with smallest
    |n[i]|; on ties prefer x, then y, then z. Then u = normalize(n x t),
    v = normalize(n x u). Both u and v are unit and perpendicular to
    each other and to the plane normal.
    """
    n_unit = _normalize3((plane.a, plane.b, plane.c))
    abs_n = (abs(n_unit[0]), abs(n_unit[1]), abs(n_unit[2]))
    if abs_n[0] <= abs_n[1] and abs_n[0] <= abs_n[2]:
        t = (1.0, 0.0, 0.0)
    elif abs_n[1] <= abs_n[2]:
        t = (0.0, 1.0, 0.0)
    else:
        t = (0.0, 0.0, 1.0)
    u = _normalize3(_cross3(n_unit, t))
    v = _normalize3(_cross3(n_unit, u))
    return u, v


def _project_to_uv(p: Vec3, u: Vec3, v: Vec3) -> Vec2:
    return (_dot3(p, u), _dot3(p, v))


def _convex_hull_2d(points: list[Vec2]) -> list[Vec2]:
    """Andrew's monotone chain. Returns CCW hull without duplicating the
    first point. Collinear points on the hull are kept dropped (strict
    `<= 0` cross-product rejection)."""
    pts = sorted(set(points))
    if len(pts) <= 1:
        return list(pts)

    def cross(o: Vec2, a: Vec2, b: Vec2) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[Vec2] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: list[Vec2] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _point_in_polygon_2d(p: Vec2, poly: list[Vec2]) -> bool:
    """Ray-casting point-in-polygon. Treats points on the boundary as
    inside (returns True). Works for non-convex polygons. Winding-
    agnostic."""
    n = len(poly)
    if n < 3:
        return False
    x, y = p
    inside = False
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        # Boundary check: collinear and within segment bounds
        cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
        if cross == 0.0:
            if (
                min(ax, bx) <= x <= max(ax, bx)
                and min(ay, by) <= y <= max(ay, by)
            ):
                return True
        # Ray-cast: horizontal +x ray; toggle when edge crosses ray
        if (ay > y) != (by > y):
            x_intersect = ax + (y - ay) * (bx - ax) / (by - ay)
            if x < x_intersect:
                inside = not inside
    return inside


def _segments_intersect_2d(
    p1: Vec2, p2: Vec2, q1: Vec2, q2: Vec2,
) -> bool:
    """Proper-or-touching segment intersection test. Returns True if the
    closed segments share any point. Handles collinear overlap."""

    def orient(a: Vec2, b: Vec2, c: Vec2) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a: Vec2, b: Vec2, c: Vec2) -> bool:
        return (
            min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
        )

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)

    if (
        ((o1 > 0.0) != (o2 > 0.0) or (o1 < 0.0) != (o2 < 0.0))
        and ((o3 > 0.0) != (o4 > 0.0) or (o3 < 0.0) != (o4 < 0.0))
    ):
        return True

    if o1 == 0.0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0.0 and on_segment(p1, p2, q2):
        return True
    if o3 == 0.0 and on_segment(q1, q2, p1):
        return True
    if o4 == 0.0 and on_segment(q1, q2, p2):
        return True
    return False


def _point_to_segment_2d(p: Vec2, a: Vec2, b: Vec2) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0.0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / seg_len_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    closest_x = a[0] + t * dx
    closest_y = a[1] + t * dy
    return math.hypot(p[0] - closest_x, p[1] - closest_y)


def _min_vertex_to_edge_distance_2d(
    vertices: list[Vec2], poly: list[Vec2],
) -> float:
    n = len(poly)
    best = math.inf
    for vertex in vertices:
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            d = _point_to_segment_2d(vertex, a, b)
            if d < best:
                best = d
    return best


def _polygons_overlap_2d(a: list[Vec2], b: list[Vec2]) -> bool:
    """Treat polygons as regions. Returns True when they share any point:
    vertex containment OR edge intersection. Handles a fully inside b,
    b fully inside a, and partial overlap. Winding-agnostic."""
    for vertex in a:
        if _point_in_polygon_2d(vertex, b):
            return True
    for vertex in b:
        if _point_in_polygon_2d(vertex, a):
            return True
    na, nb = len(a), len(b)
    for i in range(na):
        for j in range(nb):
            if _segments_intersect_2d(
                a[i], a[(i + 1) % na], b[j], b[(j + 1) % nb],
            ):
                return True
    return False


def aabb_to_polygon_planar(
    aabb: tuple[Vec3, Vec3],
    plane: Plane,
    polygon: list[Vec3],
) -> float:
    """Non-negative 2D distance between an AABB's projection onto a plane
    and a polygon lying on that plane.

    Algorithm (per A2):
      1. Build orthonormal plane-local (u, v) basis from plane.normal.
      2. Project the 8 AABB corners into (u, v); take the convex hull A.
      3. Project polygon vertices into (u, v) → polygon B (since polygon
         is on plane, the normal-axis component is zero modulo tolerance).
      4. If A and B overlap (vertex containment or edge intersection),
         return 0; otherwise return the minimum vertex-to-edge distance
         in both directions.

    Returns the IN-PLANE 2D gap ONLY. To get the full 3D distance, the
    dispatcher `bbox_to_surface` (P3.02) combines this with
    `bbox_to_plane` via `hypot`. Comparing this helper's output directly
    to `bbox_to_plane` is dimensionally incorrect — see plan G8 / A6.

    Winding-agnostic: containment uses ray casting and intersection uses
    signed-area orientation tests; both are independent of polygon
    orientation. Polygons need not be convex (the AABB projection's hull
    is convex by construction, but B may be arbitrary).
    """
    _validate_aabb("aabb", aabb)
    _validate_plane_finite("plane", plane)
    _validate_polygon_on_plane("polygon", polygon, plane)

    u_axis, v_axis = _plane_basis(plane)

    aabb_corners_2d = [
        _project_to_uv(corner, u_axis, v_axis)
        for corner in _aabb_corners(aabb)
    ]
    hull_a = _convex_hull_2d(aabb_corners_2d)
    poly_b_2d = [_project_to_uv(v, u_axis, v_axis) for v in polygon]

    if len(hull_a) == 0 or len(poly_b_2d) < 3:
        return math.inf

    if _polygons_overlap_2d(hull_a, poly_b_2d):
        return 0.0

    return min(
        _min_vertex_to_edge_distance_2d(hull_a, poly_b_2d),
        _min_vertex_to_edge_distance_2d(poly_b_2d, hull_a),
    )

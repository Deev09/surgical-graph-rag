"""P3.01 tests: aabb_to_polygon_planar (in-plane 2D distance).

Helper-only tests. The dispatcher monotonicity claim
(`bbox_to_surface(...).distance >= bbox_to_plane(...)`) is tested in
P3.02; that comparison is NOT dimensionally valid for the bare helper
(see plan G8 / A6).

Run: python tests/geometry/test_polygon_distance.py

Fixture-grounded checks load the frozen P3.00 smoke fixture and assert
the helper's in-plane gap matches each case's `expected_in_plane_gap_m`
within the fixture's declared `numeric_tolerance_m`.
"""
from __future__ import annotations

import json
import math
import random
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.types import Plane
from geometry.surface_distance import (
    aabb_to_polygon_planar,
)


P3_FIXTURE_PATH = (
    REPO_ROOT / "eval" / "questions" / "phase3_near_surface_polygon_smoke.json"
)


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _assert_value_error(fn, *args) -> None:
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _fixture_aabb(case: dict) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    mn = case["entity_aabb"]["min"]
    mx = case["entity_aabb"]["max"]
    return ((mn[0], mn[1], mn[2]), (mx[0], mx[1], mx[2]))


def _fixture_plane(case: dict) -> Plane:
    p = case["surface_record"]["plane"]
    return Plane(a=p["a"], b=p["b"], c=p["c"], d=p["d"])


def _fixture_polygon(case: dict) -> list[tuple[float, float, float]]:
    poly = case["surface_record"]["polygon"]
    return [(v[0], v[1], v[2]) for v in poly]


# --- happy-path: containment, far, boundary -------------------------------


def test_aabb_inside_polygon_footprint_returns_zero() -> None:
    """AABB projection fully inside polygon → in_plane_gap = 0."""
    aabb = ((0.5, 0.5, 0.0), (0.6, 0.6, 0.1))
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    polygon = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
    if not _close(aabb_to_polygon_planar(aabb, plane, polygon), 0.0):
        raise AssertionError("AABB projection inside polygon should give 0")


def test_aabb_far_from_polygon_returns_diagonal_distance() -> None:
    """AABB projection at (5,5,...), polygon corner at (2,2) → 3*sqrt(2)."""
    aabb = ((5.0, 5.0, 0.02), (5.1, 5.1, 0.07))
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    polygon = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
    expected = math.sqrt(2.0) * 3.0
    actual = aabb_to_polygon_planar(aabb, plane, polygon)
    if not _close(actual, expected, tol=1e-6):
        raise AssertionError(
            f"diagonal-far case wrong: expected {expected}, got {actual}"
        )


def test_aabb_straddles_polygon_edge_returns_zero() -> None:
    """AABB footprint partially overlaps polygon at the edge → 0."""
    aabb = ((1.9, 0.5, 0.02), (2.1, 0.7, 0.04))
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    polygon = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
    if not _close(aabb_to_polygon_planar(aabb, plane, polygon), 0.0):
        raise AssertionError("boundary-overlap should give 0")


def test_aabb_just_outside_polygon_edge_returns_axial_gap() -> None:
    """AABB at x=2.04..2.09, polygon ends at x=2 → in_plane_gap = 0.04."""
    aabb = ((2.04, 0.95, 0.03), (2.09, 1.05, 0.08))
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    polygon = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
    expected = 0.04
    actual = aabb_to_polygon_planar(aabb, plane, polygon)
    if not _close(actual, expected, tol=1e-9):
        raise AssertionError(
            f"axial-outside case wrong: expected {expected}, got {actual}"
        )


# --- normal-axis independence (in-plane only) -----------------------------


def test_helper_ignores_normal_axis_offset() -> None:
    """Helper returns only the IN-PLANE gap. AABB lifted off the plane
    along the normal MUST NOT change the planar distance — that's the
    whole point of separating helper from dispatcher (G8 / A6)."""
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    polygon = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
    near_plane = ((5.0, 5.0, 0.02), (5.1, 5.1, 0.07))
    far_plane = ((5.0, 5.0, 100.0), (5.1, 5.1, 100.5))
    d_near = aabb_to_polygon_planar(near_plane, plane, polygon)
    d_far = aabb_to_polygon_planar(far_plane, plane, polygon)
    if not _close(d_near, d_far, tol=1e-9):
        raise AssertionError(
            "lifting AABB along normal MUST NOT change planar gap "
            f"(near {d_near} vs far {d_far}) — dimensional bug"
        )


# --- tilted plane (not axis-aligned) --------------------------------------


def test_tilted_plane_basis_projection() -> None:
    """Plane x+y=1 normalized: normal (1,1,0)/sqrt(2). Polygon = a square
    on this plane centered at (0.5, 0.5, 0); AABB far from it."""
    s = 1.0 / math.sqrt(2.0)
    plane = Plane(a=s, b=s, c=0.0, d=-s)
    polygon = [
        (1.0, 0.0, -1.0),
        (0.0, 1.0, -1.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
    ]
    aabb = ((1.0, 0.0, 5.0), (1.05, 0.05, 5.05))
    actual = aabb_to_polygon_planar(aabb, plane, polygon)
    if actual < 3.9 or actual > 4.1:
        raise AssertionError(
            f"tilted-plane far case unexpected magnitude (got {actual}, "
            "expected close to 4)"
        )


# --- winding-agnosticism --------------------------------------------------


def test_polygon_winding_does_not_affect_distance() -> None:
    """Reverse polygon vertex order → identical helper result."""
    aabb = ((5.0, 5.0, 0.02), (5.1, 5.1, 0.07))
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    poly_ccw = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
    poly_cw = list(reversed(poly_ccw))
    d_ccw = aabb_to_polygon_planar(aabb, plane, poly_ccw)
    d_cw = aabb_to_polygon_planar(aabb, plane, poly_cw)
    if not _close(d_ccw, d_cw, tol=1e-12):
        raise AssertionError(
            f"winding-dependent: CCW {d_ccw} != CW {d_cw}"
        )


def test_winding_invariance_randomized_property() -> None:
    """Hardens the winding-agnostic claim beyond a single hand case.
    100 deterministic random AABBs against each of three polygon shapes
    (convex quad, triangle, non-convex L); reversed-vertex form MUST
    produce identical distance to within machine epsilon."""
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    polygons = [
        [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)],
        [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 2.0, 0.0)],
        [
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 2.0, 0.0),
            (0.0, 2.0, 0.0),
        ],
    ]
    rng = random.Random(0xDEADBEEF)
    mismatches: list[str] = []
    total = 0
    for poly_idx, poly_ccw in enumerate(polygons):
        poly_cw = list(reversed(poly_ccw))
        for trial in range(100):
            lo = (
                rng.uniform(-3.0, 5.0),
                rng.uniform(-3.0, 5.0),
                rng.uniform(-0.5, 0.5),
            )
            size = (
                rng.uniform(0.01, 0.5),
                rng.uniform(0.01, 0.5),
                rng.uniform(0.01, 0.5),
            )
            hi = (lo[0] + size[0], lo[1] + size[1], lo[2] + size[2])
            aabb = (lo, hi)
            d_ccw = aabb_to_polygon_planar(aabb, plane, poly_ccw)
            d_cw = aabb_to_polygon_planar(aabb, plane, poly_cw)
            total += 1
            if not _close(d_ccw, d_cw, tol=1e-12):
                mismatches.append(
                    f"poly#{poly_idx} trial#{trial}: CCW {d_ccw} != CW {d_cw} "
                    f"(delta {abs(d_ccw - d_cw):.3e})"
                )
    if total != 300:
        raise AssertionError(f"expected 300 trials, ran {total}")
    if mismatches:
        raise AssertionError(
            f"{len(mismatches)}/300 winding mismatches:\n  "
            + "\n  ".join(mismatches[:5])
        )


# --- determinism ----------------------------------------------------------


def test_repeated_calls_are_deterministic() -> None:
    aabb = ((5.0, 5.0, 0.02), (5.1, 5.1, 0.07))
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    polygon = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
    results = [aabb_to_polygon_planar(aabb, plane, polygon) for _ in range(5)]
    if len(set(results)) != 1:
        raise AssertionError(f"non-deterministic: {results}")


# --- non-convex polygon ---------------------------------------------------


def test_non_convex_polygon_distance() -> None:
    """L-shaped polygon. AABB sits in the concavity → must compute
    distance to the polygon's interior boundary, not the convex hull."""
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    # L-shape: occupies (0..2, 0..1) and (0..1, 1..2). Concavity in (1..2, 1..2).
    polygon = [
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 2.0, 0.0),
        (0.0, 2.0, 0.0),
    ]
    # AABB footprint (1.4..1.6, 1.4..1.6) lies in the L's concavity, which
    # is OUTSIDE the polygon. The closest polygon edges are y=1 for x in
    # [1, 2] and x=1 for y in [1, 2]. Perpendicular drop from AABB corner
    # (1.4, 1.4) to either edge = 0.4 (projection lands inside the edge's
    # span). The corner (1, 1) is farther (sqrt(0.32) ≈ 0.566) but the
    # perpendicular wins.
    aabb = ((1.4, 1.4, 0.1), (1.6, 1.6, 0.2))
    actual = aabb_to_polygon_planar(aabb, plane, polygon)
    expected = 0.4
    if not _close(actual, expected, tol=1e-9):
        raise AssertionError(
            f"non-convex concavity case wrong: expected {expected}, got {actual}"
        )


# --- monotonicity / non-negativity (property) -----------------------------


def test_helper_is_non_negative_property() -> None:
    """Helper distance is always >= 0 across random AABBs + planes +
    polygons. Fixed seed for determinism."""
    rng = random.Random(0xC0FFEE)
    for _ in range(200):
        lo = (
            rng.uniform(-5.0, 5.0),
            rng.uniform(-5.0, 5.0),
            rng.uniform(-5.0, 5.0),
        )
        size = (
            rng.uniform(0.01, 1.0),
            rng.uniform(0.01, 1.0),
            rng.uniform(0.01, 1.0),
        )
        hi = (lo[0] + size[0], lo[1] + size[1], lo[2] + size[2])
        # Axis-aligned floor-style polygon
        plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
        polygon = [
            (-2.0, -2.0, 0.0),
            (2.0, -2.0, 0.0),
            (2.0, 2.0, 0.0),
            (-2.0, 2.0, 0.0),
        ]
        d = aabb_to_polygon_planar((lo, hi), plane, polygon)
        if d < 0.0 or not math.isfinite(d):
            raise AssertionError(f"helper returned {d} for lo={lo} hi={hi}")


def test_helper_zero_when_aabb_strictly_inside_polygon_footprint() -> None:
    """For axis-aligned floor polygon (-2, 2)^2, any AABB whose xy footprint
    fits inside that square should give 0."""
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    polygon = [
        (-2.0, -2.0, 0.0),
        (2.0, -2.0, 0.0),
        (2.0, 2.0, 0.0),
        (-2.0, 2.0, 0.0),
    ]
    rng = random.Random(0x5EED)
    for _ in range(50):
        cx = rng.uniform(-1.5, 1.5)
        cy = rng.uniform(-1.5, 1.5)
        cz = rng.uniform(-3.0, 3.0)
        hx = rng.uniform(0.01, 0.4)
        hy = rng.uniform(0.01, 0.4)
        hz = rng.uniform(0.01, 0.4)
        aabb = ((cx - hx, cy - hy, cz - hz), (cx + hx, cy + hy, cz + hz))
        d = aabb_to_polygon_planar(aabb, plane, polygon)
        if not _close(d, 0.0, tol=1e-9):
            raise AssertionError(
                f"AABB inside polygon footprint should give 0, got {d}"
            )


# --- input validation -----------------------------------------------------


def test_rejects_polygon_with_less_than_three_vertices() -> None:
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    aabb = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    _assert_value_error(
        aabb_to_polygon_planar, aabb, plane, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    )


def test_rejects_polygon_vertex_off_plane() -> None:
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    aabb = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    polygon = [
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.0, 2.0, 0.5),  # off-plane
    ]
    _assert_value_error(aabb_to_polygon_planar, aabb, plane, polygon)


def test_rejects_non_finite_aabb_bound() -> None:
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    polygon = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0)]
    aabb = ((0.0, 0.0, 0.0), (1.0, 1.0, math.inf))
    _assert_value_error(aabb_to_polygon_planar, aabb, plane, polygon)


def test_rejects_zero_normal_plane() -> None:
    plane = Plane(a=0.0, b=0.0, c=0.0, d=0.0)
    aabb = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    polygon = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0)]
    _assert_value_error(aabb_to_polygon_planar, aabb, plane, polygon)


def test_rejects_inverted_aabb_bounds() -> None:
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    polygon = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0)]
    aabb = ((1.0, 0.0, 0.0), (0.0, 1.0, 1.0))
    _assert_value_error(aabb_to_polygon_planar, aabb, plane, polygon)


# --- P3.00 fixture-grounded checks ----------------------------------------


def _load_fixture() -> dict:
    with P3_FIXTURE_PATH.open() as fh:
        return json.load(fh)


def test_fixture_loads_and_has_expected_shape() -> None:
    fixture = _load_fixture()
    if fixture["schema"] != "phase3_near_surface_polygon_smoke":
        raise AssertionError("fixture schema mismatch")
    if not fixture.get("initial_fixture_frozen_before_extractor_code"):
        raise AssertionError("fixture not marked as pre-code frozen")
    if len(fixture["cases"]) < 8:
        raise AssertionError("fixture has fewer cases than A1 minimum")


def test_fixture_synthetic_cases_match_expected_in_plane_gap() -> None:
    """Every synthetic case that declares expected_in_plane_gap_m must
    match this helper within the fixture's tolerance."""
    fixture = _load_fixture()
    tol = fixture["numeric_tolerance_m"]
    failures: list[str] = []
    checked = 0
    for case in fixture["cases"]:
        if not case["synthetic"]:
            continue
        if case["surface_record"]["polygon"] is None:
            # Fallback case — dispatcher handles, helper not exercised
            continue
        if "expected_in_plane_gap_m" not in case:
            continue
        aabb = _fixture_aabb(case)
        plane = _fixture_plane(case)
        polygon = _fixture_polygon(case)
        actual = aabb_to_polygon_planar(aabb, plane, polygon)
        expected = case["expected_in_plane_gap_m"]
        if abs(actual - expected) > tol:
            failures.append(
                f"{case['id']}: expected_in_plane_gap_m={expected} "
                f"actual={actual} delta={abs(actual - expected)} > tol={tol}"
            )
        checked += 1
    if checked == 0:
        raise AssertionError("no fixture cases checked — fixture mis-shaped?")
    if failures:
        raise AssertionError(
            f"{len(failures)}/{checked} fixture mismatches:\n  "
            + "\n  ".join(failures)
        )


TESTS = [
    test_aabb_inside_polygon_footprint_returns_zero,
    test_aabb_far_from_polygon_returns_diagonal_distance,
    test_aabb_straddles_polygon_edge_returns_zero,
    test_aabb_just_outside_polygon_edge_returns_axial_gap,
    test_helper_ignores_normal_axis_offset,
    test_tilted_plane_basis_projection,
    test_polygon_winding_does_not_affect_distance,
    test_winding_invariance_randomized_property,
    test_repeated_calls_are_deterministic,
    test_non_convex_polygon_distance,
    test_helper_is_non_negative_property,
    test_helper_zero_when_aabb_strictly_inside_polygon_footprint,
    test_rejects_polygon_with_less_than_three_vertices,
    test_rejects_polygon_vertex_off_plane,
    test_rejects_non_finite_aabb_bound,
    test_rejects_zero_normal_plane,
    test_rejects_inverted_aabb_bounds,
    test_fixture_loads_and_has_expected_shape,
    test_fixture_synthetic_cases_match_expected_in_plane_gap,
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

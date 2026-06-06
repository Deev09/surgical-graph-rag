"""P3.02 tests: bbox_to_surface dispatcher.

Asserts the dispatcher contract:

  - polygon present → distance = hypot(normal, in_plane); evidence
    {distance_metric="polygon_clipped", polygon_clipping_applied=True,
     normal_gap_m, in_plane_gap_m, distance_m}.
  - polygon None    → distance = bbox_to_plane; evidence
    {distance_metric="bbox_to_plane", polygon_clipping_applied=False,
     normal_gap_m, distance_m, fallback_reason="polygon_none"}.
  - monotonicity   → distance >= bbox_to_plane always (A6 / G8 dispatcher form).
  - fixture S2/S3/S4 plane-vs-polygon disagreement cases match
    expected_distance_m within fixture tolerance.
  - fixture S8 fallback case emits bbox_to_plane evidence.

Run: python tests/geometry/test_bbox_to_surface.py
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
    bbox_to_plane,
    bbox_to_surface,
)


P3_FIXTURE_PATH = (
    REPO_ROOT / "eval" / "questions" / "phase3_near_surface_polygon_smoke.json"
)


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _assert_value_error(fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
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


def _fixture_polygon(case: dict) -> list[tuple[float, float, float]] | None:
    poly = case["surface_record"]["polygon"]
    if poly is None:
        return None
    return [(v[0], v[1], v[2]) for v in poly]


def _load_fixture() -> dict:
    with P3_FIXTURE_PATH.open() as fh:
        return json.load(fh)


# --- contract: polygon-present branch ------------------------------------


def test_polygon_present_returns_hypot_and_polygon_clipped_evidence() -> None:
    """S7-style boundary case: normal=0.03, in_plane=0.04 → hypot=0.05."""
    aabb = ((2.04, 0.95, 0.03), (2.09, 1.05, 0.08))
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    polygon = [
        (0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
        (2.0, 2.0, 0.0), (0.0, 2.0, 0.0),
    ]
    distance, evidence = bbox_to_surface(aabb, plane, polygon)
    if not _close(distance, 0.05, tol=1e-9):
        raise AssertionError(f"expected hypot distance 0.05, got {distance}")
    if evidence["distance_metric"] != "polygon_clipped":
        raise AssertionError("polygon-present must emit distance_metric=polygon_clipped")
    if evidence["polygon_clipping_applied"] is not True:
        raise AssertionError("polygon-present must emit polygon_clipping_applied=True")
    if not _close(evidence["normal_gap_m"], 0.03):
        raise AssertionError(f"normal_gap_m wrong: {evidence['normal_gap_m']}")
    if not _close(evidence["in_plane_gap_m"], 0.04):
        raise AssertionError(f"in_plane_gap_m wrong: {evidence['in_plane_gap_m']}")
    if not _close(evidence["distance_m"], 0.05):
        raise AssertionError(f"distance_m wrong: {evidence['distance_m']}")
    if "fallback_reason" in evidence:
        raise AssertionError("polygon-present must NOT emit fallback_reason")


def test_polygon_present_evidence_distance_m_equals_returned_distance() -> None:
    """The tuple's distance and evidence['distance_m'] must agree exactly —
    they are not two separately computed values."""
    aabb = ((5.0, 5.0, 0.02), (5.1, 5.1, 0.07))
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    polygon = [
        (0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
        (2.0, 2.0, 0.0), (0.0, 2.0, 0.0),
    ]
    distance, evidence = bbox_to_surface(aabb, plane, polygon)
    if distance != evidence["distance_m"]:
        raise AssertionError(
            f"returned distance {distance} != evidence distance_m "
            f"{evidence['distance_m']}"
        )


def test_polygon_present_inside_footprint_returns_normal_gap_only() -> None:
    """AABB projection fully inside polygon → in_plane=0 → distance=normal_gap."""
    aabb = ((0.95, 0.95, 0.0), (1.05, 1.05, 0.04))
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    polygon = [
        (0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
        (2.0, 2.0, 0.0), (0.0, 2.0, 0.0),
    ]
    distance, evidence = bbox_to_surface(aabb, plane, polygon)
    # AABB straddles plane (z=0..0.04 contains 0) → normal_gap=0 → distance=0.
    if not _close(distance, 0.0):
        raise AssertionError(
            f"straddle+inside-footprint should be 0, got {distance}"
        )
    if not _close(evidence["in_plane_gap_m"], 0.0):
        raise AssertionError("inside-polygon must give in_plane_gap_m=0")
    if not _close(evidence["normal_gap_m"], 0.0):
        raise AssertionError("plane-straddle must give normal_gap_m=0")


# --- contract: polygon-None fallback branch ------------------------------


def test_polygon_none_returns_bbox_to_plane_and_fallback_evidence() -> None:
    """polygon=None → distance=bbox_to_plane; evidence carries
    bbox_to_plane metric + fallback_reason=polygon_none."""
    aabb = ((0.95, 0.95, 0.0), (1.05, 1.05, 0.04))
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    distance, evidence = bbox_to_surface(aabb, plane, None)
    plane_only = bbox_to_plane(aabb, plane)
    if distance != plane_only:
        raise AssertionError(
            f"fallback distance {distance} != bbox_to_plane {plane_only}"
        )
    if evidence["distance_metric"] != "bbox_to_plane":
        raise AssertionError("fallback must emit distance_metric=bbox_to_plane")
    if evidence["polygon_clipping_applied"] is not False:
        raise AssertionError("fallback must emit polygon_clipping_applied=False")
    if evidence["fallback_reason"] != "polygon_none":
        raise AssertionError(
            f"fallback_reason wrong: {evidence.get('fallback_reason')!r}"
        )
    if evidence["normal_gap_m"] != plane_only:
        raise AssertionError("fallback normal_gap_m must equal bbox_to_plane")
    if evidence["distance_m"] != plane_only:
        raise AssertionError("fallback distance_m must equal bbox_to_plane")
    if "in_plane_gap_m" in evidence:
        raise AssertionError(
            "fallback evidence must NOT include in_plane_gap_m (no in-plane component)"
        )


def test_polygon_none_evidence_keys_match_phase2_bbox_to_plane_metric_name() -> None:
    """Phase 2 NEAR_SURFACE evidence uses distance_metric='bbox_to_plane'.
    Fallback evidence must reuse the EXACT same metric name so downstream
    telemetry doesn't fork."""
    aabb = ((0.0, 0.0, 0.5), (1.0, 1.0, 1.5))
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    _, evidence = bbox_to_surface(aabb, plane, None)
    if evidence["distance_metric"] != "bbox_to_plane":
        raise AssertionError(
            "fallback metric name must be 'bbox_to_plane' (not 'plane', "
            "not 'plane_fallback') to match Phase 2 evidence schema"
        )


# --- contract: dispatcher monotonicity (A6 / G8) --------------------------


def test_dispatcher_distance_ge_bbox_to_plane_property_random() -> None:
    """A6 / G8: bbox_to_surface(B, plane, polygon).distance >= bbox_to_plane(B, plane)
    for every (AABB, plane, polygon) where polygon lies on plane.

    Random seeded property test. Uses an axis-aligned plane (z=0) so a
    random in-plane polygon is trivially on-plane without precision drift.
    Both polygon-present and polygon-None branches asserted: present is
    the substantive A6 claim; None branch trivially holds with equality.
    """
    rng = random.Random(0xC0FFEE)
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    for trial in range(400):
        cx = rng.uniform(-5.0, 5.0)
        cy = rng.uniform(-5.0, 5.0)
        cz = rng.uniform(-5.0, 5.0)
        hx = rng.uniform(0.05, 2.0)
        hy = rng.uniform(0.05, 2.0)
        hz = rng.uniform(0.05, 2.0)
        aabb = (
            (cx - hx, cy - hy, cz - hz),
            (cx + hx, cy + hy, cz + hz),
        )
        # Random axis-aligned polygon on z=0 (trivially in plane to machine precision).
        px = rng.uniform(-5.0, 5.0)
        py = rng.uniform(-5.0, 5.0)
        ex = rng.uniform(0.05, 3.0)
        ey = rng.uniform(0.05, 3.0)
        polygon = [
            (px - ex, py - ey, 0.0),
            (px + ex, py - ey, 0.0),
            (px + ex, py + ey, 0.0),
            (px - ex, py + ey, 0.0),
        ]
        plane_only = bbox_to_plane(aabb, plane)
        d_poly, _ = bbox_to_surface(aabb, plane, polygon)
        d_none, _ = bbox_to_surface(aabb, plane, None)
        if d_none != plane_only:
            raise AssertionError(
                f"trial {trial}: None branch must equal bbox_to_plane "
                f"({d_none} vs {plane_only})"
            )
        if d_poly + 1e-12 < plane_only:
            raise AssertionError(
                f"trial {trial}: monotonicity violated — "
                f"polygon distance {d_poly} < plane distance {plane_only}"
            )


def test_dispatcher_distance_equals_plane_when_aabb_above_polygon() -> None:
    """When the AABB projects ENTIRELY inside the polygon footprint, the
    in-plane gap is 0, so the dispatcher distance collapses to the plane
    distance — monotonicity holds with equality."""
    aabb = ((0.5, 0.5, 1.0), (1.5, 1.5, 1.2))
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    polygon = [
        (0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
        (2.0, 2.0, 0.0), (0.0, 2.0, 0.0),
    ]
    distance, evidence = bbox_to_surface(aabb, plane, polygon)
    plane_only = bbox_to_plane(aabb, plane)
    if not _close(distance, plane_only):
        raise AssertionError(
            f"inside-footprint dispatcher distance should equal plane "
            f"distance, got {distance} vs {plane_only}"
        )
    if not _close(evidence["in_plane_gap_m"], 0.0):
        raise AssertionError("inside-footprint must give in_plane_gap_m=0")


def test_dispatcher_distance_strictly_greater_when_outside_polygon() -> None:
    """When the AABB projection is OUTSIDE the polygon, the in-plane gap
    is positive, so the dispatcher distance is STRICTLY greater than the
    plane distance."""
    aabb = ((5.0, 5.0, 0.02), (5.1, 5.1, 0.07))
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    polygon = [
        (0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
        (2.0, 2.0, 0.0), (0.0, 2.0, 0.0),
    ]
    distance, _ = bbox_to_surface(aabb, plane, polygon)
    plane_only = bbox_to_plane(aabb, plane)
    if not (distance > plane_only):
        raise AssertionError(
            f"outside-polygon dispatcher distance ({distance}) must be "
            f"strictly greater than plane distance ({plane_only})"
        )


# --- contract: validation passthrough ------------------------------------


def test_dispatcher_rejects_zero_normal_plane() -> None:
    aabb = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    plane = Plane(a=0.0, b=0.0, c=0.0, d=0.0)
    _assert_value_error(bbox_to_surface, aabb, plane, None)


def test_dispatcher_rejects_inverted_aabb_polygon_none() -> None:
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    aabb = ((1.0, 0.0, 0.0), (0.0, 1.0, 1.0))
    _assert_value_error(bbox_to_surface, aabb, plane, None)


def test_dispatcher_rejects_polygon_off_plane() -> None:
    """When polygon is provided, on-plane validation must still fire
    (deferred to aabb_to_polygon_planar)."""
    aabb = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    polygon = [(0.0, 0.0, 0.5), (1.0, 0.0, 0.5), (1.0, 1.0, 0.5)]
    _assert_value_error(bbox_to_surface, aabb, plane, polygon)


# --- P3.00 fixture: required disagreement + fallback cases ---------------


def _resolve_fixture_distance(case: dict) -> tuple[float, dict] | None:
    if not case["synthetic"]:
        return None
    aabb = _fixture_aabb(case)
    plane = _fixture_plane(case)
    polygon = _fixture_polygon(case)
    return bbox_to_surface(aabb, plane, polygon)


def test_fixture_s2_floor_outside_footprint_flips_distance() -> None:
    """S2: AABB above floor at (5,5,..) — Phase 2 would say near
    (normal_gap=0.02 <= 0.05); Phase 3 dispatcher returns hypot ~= 4.2427."""
    fixture = _load_fixture()
    tol = fixture["numeric_tolerance_m"]
    case = next(c for c in fixture["cases"] if c["id"] == "S2")
    distance, evidence = _resolve_fixture_distance(case)
    if abs(distance - case["expected_distance_m"]) > tol:
        raise AssertionError(
            f"S2 distance_m: expected {case['expected_distance_m']} "
            f"got {distance}"
        )
    if evidence["distance_metric"] != case["expected_distance_metric"]:
        raise AssertionError(
            f"S2 metric: expected {case['expected_distance_metric']} "
            f"got {evidence['distance_metric']}"
        )
    if abs(evidence["normal_gap_m"] - case["expected_normal_gap_m"]) > tol:
        raise AssertionError(
            f"S2 normal_gap_m: expected {case['expected_normal_gap_m']} "
            f"got {evidence['normal_gap_m']}"
        )
    if abs(evidence["in_plane_gap_m"] - case["expected_in_plane_gap_m"]) > tol:
        raise AssertionError(
            f"S2 in_plane_gap_m: expected {case['expected_in_plane_gap_m']} "
            f"got {evidence['in_plane_gap_m']}"
        )


def test_fixture_s3_wall_straddle_outside_polygon_flips_distance() -> None:
    """S3: AABB straddles wall plane but is 3 m outside polygon."""
    fixture = _load_fixture()
    tol = fixture["numeric_tolerance_m"]
    case = next(c for c in fixture["cases"] if c["id"] == "S3")
    distance, evidence = _resolve_fixture_distance(case)
    if abs(distance - case["expected_distance_m"]) > tol:
        raise AssertionError(
            f"S3 distance_m: expected {case['expected_distance_m']} "
            f"got {distance}"
        )
    if abs(evidence["normal_gap_m"] - case["expected_normal_gap_m"]) > tol:
        raise AssertionError(
            f"S3 normal_gap_m: expected {case['expected_normal_gap_m']} "
            f"got {evidence['normal_gap_m']}"
        )
    if abs(evidence["in_plane_gap_m"] - case["expected_in_plane_gap_m"]) > tol:
        raise AssertionError(
            f"S3 in_plane_gap_m: expected {case['expected_in_plane_gap_m']} "
            f"got {evidence['in_plane_gap_m']}"
        )


def test_fixture_s4_ceiling_outside_footprint_flips_distance() -> None:
    """S4: AABB near ceiling plane but laterally far from ceiling polygon."""
    fixture = _load_fixture()
    tol = fixture["numeric_tolerance_m"]
    case = next(c for c in fixture["cases"] if c["id"] == "S4")
    distance, evidence = _resolve_fixture_distance(case)
    if abs(distance - case["expected_distance_m"]) > tol:
        raise AssertionError(
            f"S4 distance_m: expected {case['expected_distance_m']} "
            f"got {distance}"
        )
    if abs(evidence["normal_gap_m"] - case["expected_normal_gap_m"]) > tol:
        raise AssertionError(
            f"S4 normal_gap_m: expected {case['expected_normal_gap_m']} "
            f"got {evidence['normal_gap_m']}"
        )
    if abs(evidence["in_plane_gap_m"] - case["expected_in_plane_gap_m"]) > tol:
        raise AssertionError(
            f"S4 in_plane_gap_m: expected {case['expected_in_plane_gap_m']} "
            f"got {evidence['in_plane_gap_m']}"
        )


def test_fixture_s8_fallback_emits_bbox_to_plane_evidence() -> None:
    """S8: surface with polygon=null, source=mesh_ransac. Dispatcher
    must fall back to bbox_to_plane and emit the fallback evidence shape
    (distance_metric='bbox_to_plane', fallback_reason='polygon_none',
    polygon_clipping_applied=False)."""
    fixture = _load_fixture()
    tol = fixture["numeric_tolerance_m"]
    case = next(c for c in fixture["cases"] if c["id"] == "S8")
    distance, evidence = _resolve_fixture_distance(case)
    if abs(distance - case["expected_distance_m"]) > tol:
        raise AssertionError(
            f"S8 distance_m: expected {case['expected_distance_m']} "
            f"got {distance}"
        )
    if evidence["distance_metric"] != case["expected_distance_metric"]:
        raise AssertionError(
            f"S8 metric: expected {case['expected_distance_metric']} "
            f"got {evidence['distance_metric']}"
        )
    if (
        evidence["polygon_clipping_applied"]
        is not case["expected_polygon_clipping_applied"]
    ):
        raise AssertionError(
            f"S8 polygon_clipping_applied: expected "
            f"{case['expected_polygon_clipping_applied']} "
            f"got {evidence['polygon_clipping_applied']}"
        )
    if evidence.get("fallback_reason") != case["expected_fallback_reason"]:
        raise AssertionError(
            f"S8 fallback_reason: expected {case['expected_fallback_reason']} "
            f"got {evidence.get('fallback_reason')}"
        )
    if "in_plane_gap_m" in evidence:
        raise AssertionError(
            "S8 fallback evidence must NOT include in_plane_gap_m"
        )


def test_fixture_synthetic_polygon_cases_consistent_with_helpers() -> None:
    """All synthetic polygon-present cases: dispatcher distance must equal
    hypot(bbox_to_plane, aabb_to_polygon_planar) computed standalone.
    Catches any future drift between dispatcher and underlying helpers."""
    fixture = _load_fixture()
    tol = 1e-9
    checked = 0
    for case in fixture["cases"]:
        if not case["synthetic"]:
            continue
        polygon = _fixture_polygon(case)
        if polygon is None:
            continue
        aabb = _fixture_aabb(case)
        plane = _fixture_plane(case)
        distance, _ = bbox_to_surface(aabb, plane, polygon)
        expected = math.hypot(
            bbox_to_plane(aabb, plane),
            aabb_to_polygon_planar(aabb, plane, polygon),
        )
        if abs(distance - expected) > tol:
            raise AssertionError(
                f"{case['id']}: dispatcher={distance} != "
                f"hypot(helpers)={expected}"
            )
        checked += 1
    if checked == 0:
        raise AssertionError("no synthetic polygon cases checked")


TESTS = [
    test_polygon_present_returns_hypot_and_polygon_clipped_evidence,
    test_polygon_present_evidence_distance_m_equals_returned_distance,
    test_polygon_present_inside_footprint_returns_normal_gap_only,
    test_polygon_none_returns_bbox_to_plane_and_fallback_evidence,
    test_polygon_none_evidence_keys_match_phase2_bbox_to_plane_metric_name,
    test_dispatcher_distance_ge_bbox_to_plane_property_random,
    test_dispatcher_distance_equals_plane_when_aabb_above_polygon,
    test_dispatcher_distance_strictly_greater_when_outside_polygon,
    test_dispatcher_rejects_zero_normal_plane,
    test_dispatcher_rejects_inverted_aabb_polygon_none,
    test_dispatcher_rejects_polygon_off_plane,
    test_fixture_s2_floor_outside_footprint_flips_distance,
    test_fixture_s3_wall_straddle_outside_polygon_flips_distance,
    test_fixture_s4_ceiling_outside_footprint_flips_distance,
    test_fixture_s8_fallback_emits_bbox_to_plane_evidence,
    test_fixture_synthetic_polygon_cases_consistent_with_helpers,
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

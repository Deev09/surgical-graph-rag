"""P5.01 tests: wall_contact predicate (pure geometry).

Fixture-grounded: every synthetic case (WS1-WS6) in the frozen P5.00
wall-contact smoke fixture is replayed through wall_contact and asserted to
match its declared `expected_contacts_surface`, `expected_failed_clauses`,
and `expected_clauses`. Real W1/WN cases are Replica-grounded and exercised
at the extractor layer (P5.02); P5.01 stays dependency-light.

Plus validation-failure coverage and the WS6 threshold pin.

Run: python tests/geometry/test_wall_contact.py
"""
from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.types import Plane
from geometry.wall_contact import (
    WallContactConfig,
    WallContactResult,
    wall_contact,
)


P5_FIXTURE_PATH = (
    REPO_ROOT / "eval" / "questions" / "phase5_wall_contact_smoke.json"
)
SYNTH_GRAVITY = (0.0, 0.0, -1.0)


def _assert_value_error(fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _load_fixture() -> dict:
    with P5_FIXTURE_PATH.open() as fh:
        return json.load(fh)


def _config_from_fixture(fixture: dict) -> WallContactConfig:
    d = fixture["config_defaults"]
    return WallContactConfig(
        contact_threshold_m=d["contact_threshold_m"],
        penetration_tolerance_m=d["penetration_tolerance_m"],
        max_wall_tilt_deg=d["max_wall_tilt_deg"],
        footprint_tolerance_m=d["footprint_tolerance_m"],
    )


def _aabb_from(case: dict):
    mn = case["entity_aabb"]["min"]
    mx = case["entity_aabb"]["max"]
    return ((mn[0], mn[1], mn[2]), (mx[0], mx[1], mx[2]))


def _surface(fixture: dict, ref: str):
    s = fixture["synthetic_surfaces"][ref]
    p = s["plane"]
    plane = Plane(a=p["a"], b=p["b"], c=p["c"], d=p["d"])
    polygon = [(v[0], v[1], v[2]) for v in s["polygon"]]
    return plane, polygon


# --- config defaults agree with fixture ----------------------------------


def test_default_config_matches_fixture_defaults() -> None:
    fixture = _load_fixture()
    cfg = _config_from_fixture(fixture)
    if cfg != WallContactConfig():
        raise AssertionError(
            f"WallContactConfig() defaults drifted from fixture:\n"
            f"  code:    {WallContactConfig()}\n  fixture: {cfg}"
        )


# --- fixture-driven: synthetic WS1-WS6 -----------------------------------


def test_all_synthetic_cases_match_declared_clauses() -> None:
    fixture = _load_fixture()
    tol = fixture["numeric_tolerance_m"]
    cfg = _config_from_fixture(fixture)
    bool_keys = {"wall_capable", "on_interior_side", "contact"}

    checked = 0
    failures: list[str] = []
    for case in fixture["cases"]:
        if not case["synthetic"]:
            continue
        plane, polygon = _surface(fixture, case["surface_ref"])
        result = wall_contact(
            _aabb_from(case), tuple(case["entity_centroid"]),
            plane, polygon, SYNTH_GRAVITY, cfg,
        )
        cid = case["id"]
        if result.contacts_surface != case["expected_contacts_surface"]:
            failures.append(
                f"{cid}: contacts_surface={result.contacts_surface} "
                f"expected {case['expected_contacts_surface']}"
            )
        if result.failed_clauses != case["expected_failed_clauses"]:
            failures.append(
                f"{cid}: failed_clauses={result.failed_clauses} "
                f"expected {case['expected_failed_clauses']}"
            )
        for key, expected in case.get("expected_clauses", {}).items():
            actual = result.evidence[key]
            if key in bool_keys:
                if actual is not expected:
                    failures.append(f"{cid}: ev[{key}]={actual} exp {expected}")
            else:
                if abs(actual - expected) > tol:
                    failures.append(
                        f"{cid}: ev[{key}]={actual} exp {expected} (>tol {tol})"
                    )
        checked += 1

    if checked < 6:
        raise AssertionError(f"expected >=6 synthetic cases, checked {checked}")
    if failures:
        raise AssertionError(
            f"{len(failures)} fixture mismatches:\n  " + "\n  ".join(failures)
        )


def test_ws6_threshold_pin_passes_at_0_03_rejected_at_0_02() -> None:
    """WS6 (wall_gap -0.025) must be rejected by contact at the frozen
    symmetric 0.02 band and accepted at a looser 0.03 penetration. Proves
    the symmetric choice is sharp, not arbitrary."""
    fixture = _load_fixture()
    case = next(c for c in fixture["cases"] if c["id"] == "WS6")
    plane, polygon = _surface(fixture, case["surface_ref"])
    aabb = _aabb_from(case)
    cen = tuple(case["entity_centroid"])

    at_002 = wall_contact(
        aabb, cen, plane, polygon, SYNTH_GRAVITY,
        WallContactConfig(penetration_tolerance_m=0.02),
    )
    if at_002.contacts_surface or "contact" not in at_002.failed_clauses:
        raise AssertionError(f"WS6 must be contact-rejected at 0.02; got {at_002}")
    at_003 = wall_contact(
        aabb, cen, plane, polygon, SYNTH_GRAVITY,
        WallContactConfig(penetration_tolerance_m=0.03),
    )
    if not at_003.contacts_surface:
        raise AssertionError(f"WS6 must be accepted at 0.03; got {at_003}")


def test_ws5_orientation_gate_rejects_floor() -> None:
    """WS5: a floor (up-facing) surface fed to the wall predicate is
    rejected by wall_capable ALONE (contact/side/footprint pass)."""
    fixture = _load_fixture()
    case = next(c for c in fixture["cases"] if c["id"] == "WS5")
    plane, polygon = _surface(fixture, case["surface_ref"])
    result = wall_contact(
        _aabb_from(case), tuple(case["entity_centroid"]),
        plane, polygon, SYNTH_GRAVITY, _config_from_fixture(fixture),
    )
    if result.failed_clauses != ["wall_capable"]:
        raise AssertionError(
            f"WS5 must fail wall_capable alone; got {result.failed_clauses}"
        )


def test_evidence_contains_required_keys() -> None:
    fixture = _load_fixture()
    case = next(c for c in fixture["cases"] if c["id"] == "WS1")
    plane, polygon = _surface(fixture, case["surface_ref"])
    result = wall_contact(
        _aabb_from(case), tuple(case["entity_centroid"]),
        plane, polygon, SYNTH_GRAVITY, _config_from_fixture(fixture),
    )
    required = {
        "wall_capable", "on_interior_side", "sd_centroid_m", "wall_gap_m",
        "sd_min_m", "sd_max_m", "in_plane_gap_m", "footprint_ok", "contact",
        "wall_normal_dot_up", "up", "contact_threshold_m",
        "penetration_tolerance_m", "max_wall_tilt_deg", "footprint_tolerance_m",
    }
    missing = required - set(result.evidence.keys())
    if missing:
        raise AssertionError(f"evidence missing keys: {sorted(missing)}")


# --- validation failures -------------------------------------------------


def test_rejects_invalid_config() -> None:
    _assert_value_error(WallContactConfig, contact_threshold_m=-0.01)
    _assert_value_error(WallContactConfig, penetration_tolerance_m=-0.01)
    _assert_value_error(WallContactConfig, footprint_tolerance_m=-0.01)
    _assert_value_error(WallContactConfig, max_wall_tilt_deg=0.0)
    _assert_value_error(WallContactConfig, max_wall_tilt_deg=90.0)
    _assert_value_error(WallContactConfig, max_wall_tilt_deg=float("nan"))


def _good_wall_args():
    plane = Plane(a=0.0, b=-1.0, c=0.0, d=2.0)  # north wall y=2
    polygon = [(0.0, 2.0, 0.0), (2.0, 2.0, 0.0), (2.0, 2.0, 2.0), (0.0, 2.0, 2.0)]
    aabb = ((0.8, 1.85, 0.5), (1.2, 1.99, 0.9))
    centroid = (1.0, 1.92, 0.7)
    return aabb, centroid, plane, polygon


def test_rejects_zero_gravity() -> None:
    aabb, cen, plane, poly = _good_wall_args()
    _assert_value_error(
        wall_contact, aabb, cen, plane, poly, (0.0, 0.0, 0.0), WallContactConfig(),
    )


def test_rejects_non_finite_gravity() -> None:
    aabb, cen, plane, poly = _good_wall_args()
    _assert_value_error(
        wall_contact, aabb, cen, plane, poly, (0.0, 0.0, float("inf")),
        WallContactConfig(),
    )


def test_rejects_non_normalized_plane() -> None:
    aabb, cen, _plane, poly = _good_wall_args()
    non_unit = Plane(a=0.0, b=-2.0, c=0.0, d=4.0)  # |n| = 2
    _assert_value_error(
        wall_contact, aabb, cen, non_unit, poly, SYNTH_GRAVITY, WallContactConfig(),
    )


def test_rejects_polygon_off_plane() -> None:
    aabb, cen, plane, _poly = _good_wall_args()
    off = [(0.0, 1.95, 0.0), (2.0, 1.95, 0.0), (2.0, 1.95, 2.0)]  # y=1.95, plane y=2
    _assert_value_error(
        wall_contact, aabb, cen, plane, off, SYNTH_GRAVITY, WallContactConfig(),
    )


def test_rejects_inverted_aabb() -> None:
    _aabb, cen, plane, poly = _good_wall_args()
    inverted = ((1.2, 1.85, 0.5), (0.8, 1.99, 0.9))  # lo.x > hi.x
    _assert_value_error(
        wall_contact, inverted, cen, plane, poly, SYNTH_GRAVITY, WallContactConfig(),
    )


def test_result_type_shape() -> None:
    aabb, cen, plane, poly = _good_wall_args()
    r = wall_contact(aabb, cen, plane, poly, SYNTH_GRAVITY, WallContactConfig())
    if not isinstance(r, WallContactResult):
        raise AssertionError("must return WallContactResult")
    if not isinstance(r.failed_clauses, list):
        raise AssertionError("failed_clauses must be a list")
    if r.contacts_surface is not True:
        raise AssertionError(f"clean wall contact should be True; got {r}")


TESTS = [
    test_default_config_matches_fixture_defaults,
    test_all_synthetic_cases_match_declared_clauses,
    test_ws6_threshold_pin_passes_at_0_03_rejected_at_0_02,
    test_ws5_orientation_gate_rejects_floor,
    test_evidence_contains_required_keys,
    test_rejects_invalid_config,
    test_rejects_zero_gravity,
    test_rejects_non_finite_gravity,
    test_rejects_non_normalized_plane,
    test_rejects_polygon_off_plane,
    test_rejects_inverted_aabb,
    test_result_type_shape,
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

"""P4.01 tests: rest_contact predicate (pure geometry).

Fixture-grounded: every synthetic case (F2-F9) in the frozen P4.00 smoke
fixture is replayed through rest_contact and asserted to match its
declared `expected_on_surface`, `expected_failed_clauses`, and
`expected_clauses` numeric/boolean evidence within the fixture tolerance.
F1 is Replica-grounded and deferred to the P4.02 extractor test (it needs
the live enriched-v2 bundle); P4.01 is pure-geometry and stays
dependency-light.

Plus validation-failure coverage: invalid config (the self-contained
threshold sanity; the cross-relation near guard is P4.02/OnSurfaceConfig
per D4a/G8), zero/invalid gravity, non-normalized plane, polygon off-plane.

Run: python tests/geometry/test_rest_contact.py
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
from geometry.rest_contact import (
    RestContactConfig,
    RestContactResult,
    rest_contact,
)


P4_FIXTURE_PATH = (
    REPO_ROOT / "eval" / "questions" / "phase4_on_surface_smoke.json"
)
SYNTH_GRAVITY = (0.0, 0.0, -1.0)  # synthetic_room_convention


def _assert_value_error(fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _load_fixture() -> dict:
    with P4_FIXTURE_PATH.open() as fh:
        return json.load(fh)


def _config_from_fixture(fixture: dict) -> RestContactConfig:
    d = fixture["config_defaults"]
    return RestContactConfig(
        contact_threshold_m=d["contact_threshold_m"],
        penetration_tolerance_m=d["penetration_tolerance_m"],
        max_tilt_deg=d["max_tilt_deg"],
        footprint_tolerance_m=d["footprint_tolerance_m"],
    )


def _aabb_from(case: dict):
    mn = case["entity_aabb"]["min"]
    mx = case["entity_aabb"]["max"]
    return ((mn[0], mn[1], mn[2]), (mx[0], mx[1], mx[2]))


def _plane_from(surface: dict) -> Plane:
    p = surface["plane"]
    return Plane(a=p["a"], b=p["b"], c=p["c"], d=p["d"])


def _polygon_from(surface: dict):
    return [(v[0], v[1], v[2]) for v in surface["polygon"]]


# --- fixture-driven: all synthetic cases F2-F9 ---------------------------


def test_default_config_matches_fixture_defaults() -> None:
    """The config the tests use must equal the frozen fixture defaults —
    otherwise the fixture's expected clause values are being checked against
    a different predicate than the one declared."""
    fixture = _load_fixture()
    cfg = _config_from_fixture(fixture)
    default = RestContactConfig()
    if cfg != default:
        raise AssertionError(
            f"RestContactConfig() defaults drifted from fixture defaults:\n"
            f"  code:    {default}\n  fixture: {cfg}"
        )


def test_all_synthetic_cases_match_declared_clauses() -> None:
    fixture = _load_fixture()
    tol = fixture["numeric_tolerance_m"]
    cfg = _config_from_fixture(fixture)
    surfaces = fixture["synthetic_surfaces"]

    bool_keys = {
        "support_capable",
        "centroid_on_support_side",
        "footprint_ok",
        "contact",
    }
    checked = 0
    failures: list[str] = []
    for case in fixture["cases"]:
        if not case["synthetic"]:
            continue
        surface = surfaces[case["surface_ref"]]
        aabb = _aabb_from(case)
        centroid = tuple(case["entity_centroid"])
        plane = _plane_from(surface)
        polygon = _polygon_from(surface)
        result = rest_contact(
            aabb, centroid, plane, polygon, SYNTH_GRAVITY, cfg,
        )
        cid = case["id"]

        # 1. boolean verdict
        if result.on_surface != case["expected_on_surface"]:
            failures.append(
                f"{cid}: on_surface={result.on_surface} "
                f"expected {case['expected_on_surface']}"
            )
        # 2. failed clauses (exact ordered list)
        if result.failed_clauses != case["expected_failed_clauses"]:
            failures.append(
                f"{cid}: failed_clauses={result.failed_clauses} "
                f"expected {case['expected_failed_clauses']}"
            )
        # 3. declared evidence values
        for key, expected in case.get("expected_clauses", {}).items():
            actual = result.evidence[key]
            if key in bool_keys:
                if actual is not expected:
                    failures.append(
                        f"{cid}: evidence[{key}]={actual} expected {expected}"
                    )
            else:  # *_m numeric
                if abs(actual - expected) > tol:
                    failures.append(
                        f"{cid}: evidence[{key}]={actual} expected {expected} "
                        f"(|delta|>{tol})"
                    )
        checked += 1

    if checked < 8:
        raise AssertionError(
            f"expected >=8 synthetic cases, checked {checked}"
        )
    if failures:
        raise AssertionError(
            f"{len(failures)} fixture mismatches:\n  " + "\n  ".join(failures)
        )


def test_f9_threshold_pinning_passes_at_0_05_rejected_at_0_03() -> None:
    """F9 is the auditable threshold pin (D6): bottom_gap -0.04 must be
    rejected by contact at penetration_tolerance 0.03 and accepted at 0.05.
    Proves the chosen 0.03 is sharp, not arbitrary."""
    fixture = _load_fixture()
    case = next(c for c in fixture["cases"] if c["id"] == "F9")
    surface = fixture["synthetic_surfaces"][case["surface_ref"]]
    aabb = _aabb_from(case)
    centroid = tuple(case["entity_centroid"])
    plane = _plane_from(surface)
    polygon = _polygon_from(surface)

    at_003 = rest_contact(
        aabb, centroid, plane, polygon, SYNTH_GRAVITY,
        RestContactConfig(penetration_tolerance_m=0.03),
    )
    if at_003.on_surface or "contact" not in at_003.failed_clauses:
        raise AssertionError(
            f"F9 must be contact-rejected at 0.03; got {at_003}"
        )
    at_005 = rest_contact(
        aabb, centroid, plane, polygon, SYNTH_GRAVITY,
        RestContactConfig(penetration_tolerance_m=0.05),
    )
    if not at_005.on_surface:
        raise AssertionError(
            f"F9 must be accepted at 0.05 (loose); got {at_005}"
        )


def test_evidence_contains_required_keys() -> None:
    """Per the P4.01 spec, evidence must carry the full clause + config set."""
    fixture = _load_fixture()
    cfg = _config_from_fixture(fixture)
    surface = fixture["synthetic_surfaces"]["synth_floor"]
    case = next(c for c in fixture["cases"] if c["id"] == "F2")
    result = rest_contact(
        _aabb_from(case), tuple(case["entity_centroid"]),
        _plane_from(surface), _polygon_from(surface), SYNTH_GRAVITY, cfg,
    )
    required = {
        "support_capable", "centroid_on_support_side", "sd_centroid_m",
        "bottom_gap_m", "sd_min_m", "sd_max_m", "in_plane_gap_m",
        "footprint_ok", "contact", "support_normal_dot_up", "up",
        "contact_threshold_m", "penetration_tolerance_m", "max_tilt_deg",
        "footprint_tolerance_m",
    }
    missing = required - set(result.evidence.keys())
    if missing:
        raise AssertionError(f"evidence missing keys: {sorted(missing)}")


# --- validation failures -------------------------------------------------


def test_rejects_invalid_config_thresholds() -> None:
    """Self-contained config sanity only. The cross-relation near guard
    belongs in P4.02/OnSurfaceConfig and is not asserted here."""
    _assert_value_error(RestContactConfig, contact_threshold_m=-0.01)
    _assert_value_error(RestContactConfig, penetration_tolerance_m=-0.01)
    _assert_value_error(RestContactConfig, footprint_tolerance_m=-0.01)
    _assert_value_error(RestContactConfig, contact_threshold_m=float("inf"))
    _assert_value_error(RestContactConfig, max_tilt_deg=0.0)
    _assert_value_error(RestContactConfig, max_tilt_deg=90.0)
    _assert_value_error(RestContactConfig, max_tilt_deg=120.0)
    _assert_value_error(RestContactConfig, max_tilt_deg=float("nan"))


def _good_floor_args():
    plane = Plane(a=0.0, b=0.0, c=1.0, d=0.0)
    polygon = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
    aabb = ((0.8, 0.8, -0.01), (1.2, 1.2, 0.3))
    centroid = (1.0, 1.0, 0.145)
    return aabb, centroid, plane, polygon


def test_rejects_zero_gravity() -> None:
    aabb, centroid, plane, polygon = _good_floor_args()
    _assert_value_error(
        rest_contact, aabb, centroid, plane, polygon,
        (0.0, 0.0, 0.0), RestContactConfig(),
    )


def test_rejects_non_finite_gravity() -> None:
    aabb, centroid, plane, polygon = _good_floor_args()
    _assert_value_error(
        rest_contact, aabb, centroid, plane, polygon,
        (0.0, 0.0, float("inf")), RestContactConfig(),
    )


def test_rejects_non_normalized_plane() -> None:
    aabb, centroid, _plane, polygon = _good_floor_args()
    non_unit = Plane(a=0.0, b=0.0, c=2.0, d=0.0)  # |n| = 2
    _assert_value_error(
        rest_contact, aabb, centroid, non_unit, polygon,
        SYNTH_GRAVITY, RestContactConfig(),
    )


def test_rejects_polygon_off_plane() -> None:
    aabb, centroid, plane, _polygon = _good_floor_args()
    off = [(0.0, 0.0, 0.05), (2.0, 0.0, 0.05), (2.0, 2.0, 0.05)]  # z=0.05, plane z=0
    _assert_value_error(
        rest_contact, aabb, centroid, plane, off,
        SYNTH_GRAVITY, RestContactConfig(),
    )


def test_rejects_inverted_aabb() -> None:
    _aabb, centroid, plane, polygon = _good_floor_args()
    inverted = ((1.2, 0.8, -0.01), (0.8, 1.2, 0.3))  # lo.x > hi.x
    _assert_value_error(
        rest_contact, inverted, centroid, plane, polygon,
        SYNTH_GRAVITY, RestContactConfig(),
    )


def test_result_type_shape() -> None:
    aabb, centroid, plane, polygon = _good_floor_args()
    r = rest_contact(aabb, centroid, plane, polygon, SYNTH_GRAVITY, RestContactConfig())
    if not isinstance(r, RestContactResult):
        raise AssertionError("rest_contact must return RestContactResult")
    if not isinstance(r.failed_clauses, list):
        raise AssertionError("failed_clauses must be a list")
    if r.on_surface is not True:
        raise AssertionError(f"clean floor rest should be ON; got {r}")


TESTS = [
    test_default_config_matches_fixture_defaults,
    test_all_synthetic_cases_match_declared_clauses,
    test_f9_threshold_pinning_passes_at_0_05_rejected_at_0_03,
    test_evidence_contains_required_keys,
    test_rejects_invalid_config_thresholds,
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

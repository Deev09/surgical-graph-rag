"""Phase 4 P4.01 — rest-contact predicate (pure geometry).

`ON_SURFACE` (Design B) = gravity-supported rest-contact on an up-facing
surface. This module implements the predicate as a pure-geometry leaf:
it imports only `common.types` and sibling `geometry` helpers — no
`graph/`, `extractors/`, or `graph.schema` imports (same discipline as
`bbox_to_surface`). The extractor wiring, the `ON_SURFACE` EdgeType, and
the cross-relation subset guard (`hypot(contact, footprint) <= near`)
are P4.02 concerns and deliberately not here.

Predicate (see docs/phase4_plan.md, "Rest-contact predicate"):

  up                       = normalize(-gravity)
  support_capable          := dot(normal_unit, up) >= cos(max_tilt_deg)
  centroid_on_support_side := signed_distance(centroid, plane) >= 0
  bottom_gap               := min over 8 AABB corners of signed_distance(corner)
  contact                  := -penetration_tolerance_m <= bottom_gap <= contact_threshold_m
  footprint_ok             := aabb_to_polygon_planar(aabb, plane, polygon) <= footprint_tolerance_m
  ON_SURFACE               := support_capable and centroid_on_support_side
                              and contact and footprint_ok

`signed_distance` is `point_to_plane` (= a*x + b*y + c*z + d, positive on
the normal side). `bottom_gap` is the most-negative corner signed distance
(`sd_min`); positive when the box hovers entirely above the plane, negative
when the lowest corner dips below it.

Precondition (validated): the plane normal `(a,b,c)` is unit-norm. The
up-facing / interior-facing orientation convention is assumed (Phase 3
fixtures use it); `support_capable` is the operational check for it and a
flipped normal cannot be detected from a single (aabb, plane) pair, so it
is documented, not validated here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from common.types import Plane, Vec3
# Reuse sibling geometry primitives + validators (same package; internal
# cohesion, avoids duplicating validation logic).
from geometry.surface_distance import (
    _aabb_corners,
    _validate_aabb,
    _validate_plane_finite,
    _validate_vec3,
    aabb_to_polygon_planar,
    point_to_plane,
)


NORMAL_UNIT_TOL = 1e-6


@dataclass(frozen=True)
class RestContactConfig:
    """Rest-contact thresholds. Self-contained sanity is validated in
    __post_init__ (non-negative tolerances, max_tilt in (0, 90)). The
    cross-relation subset guard (hypot(contact_threshold, footprint_tol)
    <= near_surface_threshold) is NOT here — per the Phase 4 plan (D4a/G8)
    it lives in OnSurfaceConfig (P4.02), where near_surface_threshold is
    in scope."""
    contact_threshold_m: float = 0.02
    penetration_tolerance_m: float = 0.03
    max_tilt_deg: float = 30.0
    footprint_tolerance_m: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "contact_threshold_m",
            "penetration_tolerance_m",
            "footprint_tolerance_m",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative, got {value!r}"
                )
        if not math.isfinite(self.max_tilt_deg) or not (
            0.0 < self.max_tilt_deg < 90.0
        ):
            raise ValueError(
                f"max_tilt_deg must be in (0, 90), got {self.max_tilt_deg!r}"
            )


@dataclass(frozen=True)
class RestContactResult:
    on_surface: bool
    failed_clauses: list[str]
    evidence: dict


def _validate_plane_unit(name: str, plane: Plane) -> None:
    mag = math.sqrt(plane.a * plane.a + plane.b * plane.b + plane.c * plane.c)
    if abs(mag - 1.0) > NORMAL_UNIT_TOL:
        raise ValueError(
            f"{name} normal must be unit-norm (|n|={mag!r}), within "
            f"{NORMAL_UNIT_TOL}"
        )


def _up_from_gravity(gravity: Vec3) -> Vec3:
    _validate_vec3("gravity", gravity)
    gx, gy, gz = gravity
    mag = math.sqrt(gx * gx + gy * gy + gz * gz)
    if mag == 0.0:
        raise ValueError("gravity must be non-zero")
    return (-gx / mag, -gy / mag, -gz / mag)


def rest_contact(
    aabb: tuple[Vec3, Vec3],
    centroid: Vec3,
    plane: Plane,
    polygon: list[Vec3],
    gravity: Vec3,
    config: RestContactConfig,
) -> RestContactResult:
    """Evaluate the rest-contact predicate for one (entity AABB, surface).

    Returns a RestContactResult with the boolean verdict, the ordered list
    of failed clauses (clause order: support_capable, centroid_on_support_side,
    contact, footprint_ok), and a full evidence dict. Pure function; raises
    ValueError on malformed inputs (bad AABB, non-finite/non-unit plane,
    zero/invalid gravity, polygon off-plane).
    """
    _validate_aabb("aabb", aabb)
    _validate_vec3("centroid", centroid)
    _validate_plane_finite("plane", plane)
    _validate_plane_unit("plane", plane)
    up = _up_from_gravity(gravity)

    # support_capable — normalize the (validated ~unit) normal defensively.
    n = (plane.a, plane.b, plane.c)
    nmag = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
    nu = (n[0] / nmag, n[1] / nmag, n[2] / nmag)
    ndot_up = nu[0] * up[0] + nu[1] * up[1] + nu[2] * up[2]
    cos_max = math.cos(math.radians(config.max_tilt_deg))
    support_capable = ndot_up >= cos_max

    # centroid side
    sd_centroid = point_to_plane(centroid, plane)
    centroid_on_support_side = sd_centroid >= 0.0

    # bottom_gap = most-negative corner signed distance
    sds = [point_to_plane(corner, plane) for corner in _aabb_corners(aabb)]
    sd_min = min(sds)
    sd_max = max(sds)
    bottom_gap = sd_min
    contact = (
        bottom_gap <= config.contact_threshold_m
        and bottom_gap >= -config.penetration_tolerance_m
    )

    # footprint — aabb_to_polygon_planar validates aabb / plane-finite /
    # polygon-on-plane internally and raises on a polygon off the plane.
    in_plane_gap = aabb_to_polygon_planar(aabb, plane, polygon)
    footprint_ok = in_plane_gap <= config.footprint_tolerance_m

    on_surface = (
        support_capable
        and centroid_on_support_side
        and contact
        and footprint_ok
    )

    failed_clauses: list[str] = []
    if not support_capable:
        failed_clauses.append("support_capable")
    if not centroid_on_support_side:
        failed_clauses.append("centroid_on_support_side")
    if not contact:
        failed_clauses.append("contact")
    if not footprint_ok:
        failed_clauses.append("footprint_ok")

    evidence = {
        "support_capable": support_capable,
        "centroid_on_support_side": centroid_on_support_side,
        "sd_centroid_m": sd_centroid,
        "bottom_gap_m": bottom_gap,
        "sd_min_m": sd_min,
        "sd_max_m": sd_max,
        "in_plane_gap_m": in_plane_gap,
        "footprint_ok": footprint_ok,
        "contact": contact,
        "support_normal_dot_up": ndot_up,
        "up": up,
        "contact_threshold_m": config.contact_threshold_m,
        "penetration_tolerance_m": config.penetration_tolerance_m,
        "max_tilt_deg": config.max_tilt_deg,
        "footprint_tolerance_m": config.footprint_tolerance_m,
    }
    return RestContactResult(
        on_surface=on_surface,
        failed_clauses=failed_clauses,
        evidence=evidence,
    )

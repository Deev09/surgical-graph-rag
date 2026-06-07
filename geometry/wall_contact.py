"""Phase 5 P5.01 — wall-contact predicate (pure geometry).

`CONTACTS_SURFACE` = normal-side contact with a wall-capable (roughly
vertical) surface. This is the Phase 4 rest-contact predicate with the
orientation gate flipped from gravity-side (`support_capable`) to
normal-perpendicular-to-gravity (`wall_capable`); everything else —
interior-side check, signed min-corner gap with a symmetric penetration
band, polygon-footprint clip — is identical in structure.

Pure-geometry leaf: imports only `common.types` and sibling `geometry`
helpers; no `graph/`, `extractors/`, or `graph.schema` imports.

Predicate (see docs/phase5_plan.md, "Wall-contact predicate"):

  up               = normalize(-gravity)
  wall_capable     := abs(dot(normal_unit, up)) <= sin(max_wall_tilt_deg)
  on_interior_side := signed_distance(centroid, plane) >= 0
  wall_gap         := min over 8 AABB corners of signed_distance(corner)  (sd_min)
  contact          := -penetration_tolerance_m <= wall_gap <= contact_threshold_m
  footprint_ok     := aabb_to_polygon_planar(aabb, plane, polygon) <= footprint_tolerance_m
  CONTACTS_SURFACE := wall_capable and on_interior_side and contact and footprint_ok

`signed_distance` is `point_to_plane` (positive on the interior-facing
normal side). `wall_gap` is the most-negative corner signed distance
(positive when the entity stands off the wall, negative when it penetrates).

Precondition (validated): the plane normal is unit-norm; the interior-facing
orientation convention is assumed (same as P4.01). `wall_capable` is the
operational orientation check.

Note for the extractor layer (P5.02): edge evidence is JSON-serialized, so
`up` (a tuple here) must be normalized to a list at the edge-evidence
assembly boundary — see the P4.06 round-trip fix in
graph/relations/on_surface.py. This helper returns the natural tuple.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from common.types import Plane, Vec3
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
class WallContactConfig:
    """Wall-contact thresholds. Self-contained sanity is validated in
    __post_init__ (non-negative tolerances, max_wall_tilt in (0, 90)). The
    cross-relation subset guard (hypot(contact_threshold, footprint_tol)
    <= near_surface_threshold) is NOT here — per the Phase 5 plan it lives
    in ContactsSurfaceConfig (P5.02), where near_surface_threshold is in
    scope. Frozen P5 defaults are symmetric (0.02/0.02) — walls show no
    systematic fit bias, unlike the floor."""
    contact_threshold_m: float = 0.02
    penetration_tolerance_m: float = 0.02
    max_wall_tilt_deg: float = 30.0
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
        if not math.isfinite(self.max_wall_tilt_deg) or not (
            0.0 < self.max_wall_tilt_deg < 90.0
        ):
            raise ValueError(
                f"max_wall_tilt_deg must be in (0, 90), got {self.max_wall_tilt_deg!r}"
            )


@dataclass(frozen=True)
class WallContactResult:
    contacts_surface: bool
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


def wall_contact(
    aabb: tuple[Vec3, Vec3],
    centroid: Vec3,
    plane: Plane,
    polygon: list[Vec3],
    gravity: Vec3,
    config: WallContactConfig,
) -> WallContactResult:
    """Evaluate the wall-contact predicate for one (entity AABB, surface).

    Returns a WallContactResult with the boolean verdict, the ordered list of
    failed clauses (clause order: wall_capable, on_interior_side, contact,
    footprint_ok), and a full evidence dict. Pure function; raises ValueError
    on malformed inputs (bad AABB, non-finite/non-unit plane, zero/invalid
    gravity, polygon off-plane).
    """
    _validate_aabb("aabb", aabb)
    _validate_vec3("centroid", centroid)
    _validate_plane_finite("plane", plane)
    _validate_plane_unit("plane", plane)
    up = _up_from_gravity(gravity)

    # wall_capable — normal ~perpendicular to gravity (roughly vertical wall).
    n = (plane.a, plane.b, plane.c)
    nmag = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
    nu = (n[0] / nmag, n[1] / nmag, n[2] / nmag)
    ndot_up = nu[0] * up[0] + nu[1] * up[1] + nu[2] * up[2]
    sin_max = math.sin(math.radians(config.max_wall_tilt_deg))
    wall_capable = abs(ndot_up) <= sin_max

    # interior side
    sd_centroid = point_to_plane(centroid, plane)
    on_interior_side = sd_centroid >= 0.0

    # wall_gap = most-negative corner signed distance
    sds = [point_to_plane(corner, plane) for corner in _aabb_corners(aabb)]
    sd_min = min(sds)
    sd_max = max(sds)
    wall_gap = sd_min
    contact = (
        wall_gap <= config.contact_threshold_m
        and wall_gap >= -config.penetration_tolerance_m
    )

    in_plane_gap = aabb_to_polygon_planar(aabb, plane, polygon)
    footprint_ok = in_plane_gap <= config.footprint_tolerance_m

    contacts_surface = (
        wall_capable
        and on_interior_side
        and contact
        and footprint_ok
    )

    failed_clauses: list[str] = []
    if not wall_capable:
        failed_clauses.append("wall_capable")
    if not on_interior_side:
        failed_clauses.append("on_interior_side")
    if not contact:
        failed_clauses.append("contact")
    if not footprint_ok:
        failed_clauses.append("footprint_ok")

    evidence = {
        "wall_capable": wall_capable,
        "on_interior_side": on_interior_side,
        "sd_centroid_m": sd_centroid,
        "wall_gap_m": wall_gap,
        "sd_min_m": sd_min,
        "sd_max_m": sd_max,
        "in_plane_gap_m": in_plane_gap,
        "footprint_ok": footprint_ok,
        "contact": contact,
        "wall_normal_dot_up": ndot_up,
        "up": up,
        "contact_threshold_m": config.contact_threshold_m,
        "penetration_tolerance_m": config.penetration_tolerance_m,
        "max_wall_tilt_deg": config.max_wall_tilt_deg,
        "footprint_tolerance_m": config.footprint_tolerance_m,
    }
    return WallContactResult(
        contacts_surface=contacts_surface,
        failed_clauses=failed_clauses,
        evidence=evidence,
    )

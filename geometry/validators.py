"""Phase 2 P2.04 — geometry validators.

Each validator operates on the JSON records emitted by the Phase 2
enriched-v2 importer (importers/replica.py with --enriched-v2). They
are pure functions; each raises GeometryValidationError on failure
with enough context to localize the bad record.

Scope (per the P2.04 sign-off): geometry validation against the emitted
records only. These validators do NOT depend on the runtime
StructuralSurface dataclass — that read path lands in P2.06 along with
the dataclass-level `source` field. No extractor or graph-schema work
here.

Record shapes assumed (mirrors the importer's v2 JSON):

  obb           : {"center": [x,y,z], "extents": [a,b,c],
                   "rotation_quat": [qx,qy,qz,qw]}
  plane         : {"normal": [a,b,c], "d": d}
  surface       : {"surface_uid", "surface_type", "source",
                   "source_instance_ref", "plane", "polygon" | None,
                   "source_obb", "confidence"}
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable

ALLOWED_SURFACE_SOURCES = (
    "habitat_label", "mesh_ransac", "mesh_region_fit", "synth_bbox_fallback",
)
ALLOWED_SURFACE_TYPES = ("floor", "wall", "ceiling")
REQUIRED_CANONICAL_COUNTS = {"floor": 1, "wall": 2, "ceiling": 1}


class GeometryValidationError(ValueError):
    """Raised when a geometry record fails a P2.04 validator."""


def _all_finite(vec: Iterable[float]) -> bool:
    return all(math.isfinite(float(x)) for x in vec)


def _norm(vec: Iterable[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in vec))


def validate_gravity_alignment(
    gravity: list[float], tolerance: float = 0.05,
) -> None:
    """The gravity vector is approximately a unit vector aligned with one
    canonical axis (±X, ±Y, ±Z) within `tolerance` on every component.
    Real Replica gravity ≈ [0.0036, 0.0030, -0.99999] passes at the default
    tolerance; perfectly diagonal gravity does not."""
    if not isinstance(gravity, (list, tuple)) or len(gravity) != 3:
        raise GeometryValidationError(
            f"gravity must be a 3-vector, got {type(gravity).__name__} "
            f"len={len(gravity) if hasattr(gravity, '__len__') else 'n/a'}"
        )
    if not _all_finite(gravity):
        raise GeometryValidationError(
            f"gravity contains non-finite components: {gravity!r}"
        )
    n = _norm(gravity)
    if abs(n - 1.0) > tolerance:
        raise GeometryValidationError(
            f"gravity is not unit-norm within tolerance={tolerance}: |g|={n}"
        )
    abs_components = [abs(float(c)) for c in gravity]
    dominant = max(range(3), key=lambda k: abs_components[k])
    if abs_components[dominant] < 1.0 - tolerance:
        raise GeometryValidationError(
            f"gravity is not axis-aligned within tolerance={tolerance}: {gravity!r}"
        )
    off_axis = [abs_components[k] for k in range(3) if k != dominant]
    if max(off_axis) > tolerance:
        raise GeometryValidationError(
            f"gravity off-axis components exceed tolerance={tolerance}: {gravity!r}"
        )


def validate_obb_sanity(obb: dict[str, Any]) -> None:
    """OBB record: center finite, extents strictly positive and finite,
    rotation_quat unit-norm. Validates the importer's emitted JSON shape
    (matches both per-object `bbox_obb` and per-surface `source_obb`)."""
    for key in ("center", "extents", "rotation_quat"):
        if key not in obb:
            raise GeometryValidationError(f"OBB missing required key: {key!r}")
    center = obb["center"]
    extents = obb["extents"]
    quat = obb["rotation_quat"]
    if len(center) != 3 or len(extents) != 3:
        raise GeometryValidationError(
            f"OBB center/extents must be 3-vectors, got len={len(center)},{len(extents)}"
        )
    if len(quat) != 4:
        raise GeometryValidationError(
            f"OBB rotation_quat must be 4-vector, got len={len(quat)}"
        )
    if not _all_finite(center):
        raise GeometryValidationError(f"OBB center non-finite: {center!r}")
    for i, e in enumerate(extents):
        if not math.isfinite(float(e)):
            raise GeometryValidationError(f"OBB extent[{i}] non-finite: {e!r}")
        if float(e) <= 0.0:
            raise GeometryValidationError(f"OBB extent[{i}] must be positive: {e}")
    if not _all_finite(quat):
        raise GeometryValidationError(f"OBB rotation_quat non-finite: {quat!r}")
    qn = _norm(quat)
    if abs(qn - 1.0) > 1e-3:
        raise GeometryValidationError(f"OBB rotation_quat not unit-norm: |q|={qn}")


def validate_plane_normalized(
    plane: dict[str, Any], tolerance: float = 1e-6,
) -> None:
    """Plane record: `normal` is unit-norm within `tolerance`; `d` is
    finite. Tolerance default matches the importer's 6-decimal rounding."""
    if "normal" not in plane or "d" not in plane:
        raise GeometryValidationError(f"plane missing 'normal' or 'd': {plane!r}")
    normal = plane["normal"]
    d = plane["d"]
    if len(normal) != 3:
        raise GeometryValidationError(
            f"plane normal must be 3-vector, got len={len(normal)}"
        )
    if not _all_finite(normal):
        raise GeometryValidationError(f"plane normal non-finite: {normal!r}")
    if not math.isfinite(float(d)):
        raise GeometryValidationError(f"plane d non-finite: {d!r}")
    n = _norm(normal)
    if abs(n - 1.0) > tolerance:
        raise GeometryValidationError(
            f"plane normal not unit-norm within tolerance={tolerance}: "
            f"|n|={n}, normal={normal!r}"
        )


def validate_surface_extents(
    surface: dict[str, Any],
    room_bbox: list[list[float]],
    tolerance: float = 1e-6,
) -> None:
    """If the surface has a polygon, every corner must lie within
    `room_bbox` (within `tolerance`). Skipped silently when polygon is
    None or absent — per the P2.03 contract that polygons are optional.

    Caveat: capture_meta's room_bbox is derived from object centroids
    (per A3) and is NOT room geometry. Callers wanting an honest
    extents check should pass a room bbox derived from actual room
    geometry (mesh AABB, raw Habitat bounds, etc.)."""
    polygon = surface.get("polygon")
    if polygon is None:
        return
    if len(room_bbox) != 2 or len(room_bbox[0]) != 3 or len(room_bbox[1]) != 3:
        raise GeometryValidationError(
            f"room_bbox must be [[lo_x,lo_y,lo_z],[hi_x,hi_y,hi_z]], got {room_bbox!r}"
        )
    lo = [float(x) for x in room_bbox[0]]
    hi = [float(x) for x in room_bbox[1]]
    if not _all_finite(lo) or not _all_finite(hi):
        raise GeometryValidationError(f"room_bbox contains non-finite values: {room_bbox!r}")
    for axis in range(3):
        if lo[axis] > hi[axis]:
            raise GeometryValidationError(
                f"room_bbox axis {axis} has lo > hi: {lo[axis]} > {hi[axis]}"
            )
    uid = surface.get("surface_uid", "?")
    if len(polygon) < 3:
        raise GeometryValidationError(
            f"surface {uid!r} polygon must have at least 3 corners, got {len(polygon)}"
        )
    for i, p in enumerate(polygon):
        if len(p) != 3 or not _all_finite(p):
            raise GeometryValidationError(
                f"surface {uid!r} polygon[{i}] invalid: {p!r}"
            )
        for axis in range(3):
            v = float(p[axis])
            if v < lo[axis] - tolerance or v > hi[axis] + tolerance:
                raise GeometryValidationError(
                    f"surface {uid!r} polygon[{i}] axis {axis} value {v} "
                    f"outside room_bbox [{lo[axis]},{hi[axis]}]"
                )


def validate_surface_source(surface: dict[str, Any]) -> None:
    """Per-surface provenance (A7): `source` is present and one of the
    recognized values. Catches missing-field bugs from upstream
    extractor changes."""
    if "source" not in surface:
        raise GeometryValidationError(
            f"surface {surface.get('surface_uid', '?')!r} missing 'source' field"
        )
    s = surface["source"]
    if s not in ALLOWED_SURFACE_SOURCES:
        raise GeometryValidationError(
            f"surface {surface.get('surface_uid', '?')!r} unrecognized "
            f"source={s!r}; allowed: {ALLOWED_SURFACE_SOURCES}"
        )


def validate_canonical_surface_set(
    surfaces: list[dict[str, Any]],
) -> None:
    """Canonical Phase 2 surface-set rule (G1): ≥1 floor, ≥2 walls,
    ≥1 ceiling, with every surface source != 'synth_bbox_fallback'.
    The synth-fallback experiment path skips this validator deliberately
    (see P2.03 / A3)."""
    counts: dict[str, int] = {}
    for s in surfaces:
        if "surface_type" not in s:
            raise GeometryValidationError(f"surface missing surface_type: {s!r}")
        if s["surface_type"] not in ALLOWED_SURFACE_TYPES:
            raise GeometryValidationError(
                f"surface {s.get('surface_uid', '?')!r} has unrecognized "
                f"surface_type={s['surface_type']!r}; allowed: {ALLOWED_SURFACE_TYPES}"
            )
        validate_surface_source(s)
        counts[s["surface_type"]] = counts.get(s["surface_type"], 0) + 1
    for surface_type, required in REQUIRED_CANONICAL_COUNTS.items():
        got = counts.get(surface_type, 0)
        if got < required:
            raise GeometryValidationError(
                f"canonical surface set requires >={required} '{surface_type}', "
                f"got {got}; counts={counts}"
            )
    bad = [
        s.get("surface_uid", "?") for s in surfaces
        if s.get("source") == "synth_bbox_fallback"
    ]
    if bad:
        raise GeometryValidationError(
            f"canonical surface set contains synth_bbox_fallback surfaces: {bad}"
        )


def validate_deterministic_hash(
    hash_a: str, hash_b: str, *, context: str = "",
) -> None:
    """Compare two pre-computed hashes; raise if they differ. The caller
    chooses the hash function and what to hash. For importer-output
    byte-equality checks, use `sha256_of_bytes` on the JSON payload."""
    if hash_a != hash_b:
        ctx = f" ({context})" if context else ""
        raise GeometryValidationError(
            f"determinism check failed{ctx}: "
            f"hash_a={hash_a[:16]}... != hash_b={hash_b[:16]}..."
        )


def sha256_of_bytes(payload: bytes) -> str:
    """SHA-256 hex digest of a byte payload. Helper for the canonical
    byte-equality determinism check."""
    return hashlib.sha256(payload).hexdigest()

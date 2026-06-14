"""Derived entity-top support surfaces for Phase 6.

These are deliberately NOT StructuralSurface records. They are pure
build-time helpers derived from support-class entity AABBs so furniture-top
support can be evaluated without expanding the structural-surface manifest.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from common.types import Plane, SceneFrame, Vec3
from extractors.base import EntityArtifact


DEFAULT_SUPPORT_CLASS_ALLOWLIST: tuple[str, ...] = (
    "table",
    "desk",
    "chair",
    "stool",
    "bench",
    "shelf",
    "sofa",
    "plant-stand",
    "counter",
)

ENTITY_SURFACE_SOURCE = "derived_entity_top"
ENTITY_SURFACE_VERSION = "0.1"


@dataclass(frozen=True)
class EntitySurface:
    surface_uid: str
    owner_entity_uid: str
    owner_class: str
    plane: Plane
    polygon: list[Vec3]
    source: str = ENTITY_SURFACE_SOURCE
    confidence: float = 1.0


def normalize_entity_class(label: str) -> str:
    """Normalize backend labels into class-family strings.

    Keeps hyphenated class names such as ``plant-stand`` intact while stripping
    common instance suffixes such as ``table_5`` or ``chair-1``.
    """
    value = re.sub(r"\s+", " ", str(label).strip().lower())
    value = value.replace("_", "-")
    value = re.sub(r"[-\s]+[0-9]+$", "", value)
    value = value.replace(" ", "-")
    value = re.sub(r"-+", "-", value).strip("-")
    return value


def class_candidates_for_entity(entity: EntityArtifact) -> list[str]:
    """Candidate class-family labels in deterministic preference order."""
    labels: list[str] = []
    if entity.semantic_hypotheses:
        labels.extend(h.label for h in entity.semantic_hypotheses)
    labels.append(entity.identity.display_label)
    labels.extend(entity.identity.aliases)

    out: list[str] = []
    seen: set[str] = set()
    for label in labels:
        cls = normalize_entity_class(label)
        if cls and cls not in seen:
            out.append(cls)
            seen.add(cls)
    return out


def support_class_for_entity(
    entity: EntityArtifact,
    allowlist: tuple[str, ...] = DEFAULT_SUPPORT_CLASS_ALLOWLIST,
) -> str | None:
    allowed = {normalize_entity_class(c) for c in allowlist}
    for cls in class_candidates_for_entity(entity):
        if cls in allowed:
            return cls
    return None


def _up_from_frame(frame: SceneFrame) -> Vec3:
    gx, gy, gz = frame.gravity
    mag = math.sqrt(gx * gx + gy * gy + gz * gz)
    if mag == 0.0:
        raise ValueError("frame.gravity must be non-zero")
    return (-gx / mag, -gy / mag, -gz / mag)


AXIS_ALIGNED_UP_TOL = 0.01


def _axis_aligned_up(frame: SceneFrame) -> tuple[int, float]:
    """Return the dominant AABB axis and sign for an axis-aligned up vector."""
    up = _up_from_frame(frame)
    axis = max(range(3), key=lambda i: abs(up[i]))
    sign = 1.0 if up[axis] >= 0.0 else -1.0
    if abs(abs(up[axis]) - 1.0) > AXIS_ALIGNED_UP_TOL:
        raise ValueError(
            "EntitySurface AABB top derivation requires axis-aligned gravity; "
            f"got up={up!r}"
        )
    for i, value in enumerate(up):
        if i != axis and abs(value) > AXIS_ALIGNED_UP_TOL:
            raise ValueError(
                "EntitySurface AABB top derivation requires axis-aligned gravity; "
                f"got up={up!r}"
            )
    return axis, sign


def _top_face_from_aabb(
    aabb: tuple[Vec3, Vec3],
    *,
    frame: SceneFrame,
) -> tuple[Plane, list[Vec3]]:
    lo, hi = aabb
    axis, sign = _axis_aligned_up(frame)
    normal = [0.0, 0.0, 0.0]
    normal[axis] = sign
    coord = hi[axis] if sign > 0 else lo[axis]
    d = -sign * coord
    plane = Plane(a=normal[0], b=normal[1], c=normal[2], d=d)

    axes = [0, 1, 2]
    axes.remove(axis)
    a0, a1 = axes
    face = [0.0, 0.0, 0.0]
    face[axis] = coord
    corners: list[Vec3] = []
    for v0, v1 in (
        (lo[a0], lo[a1]),
        (hi[a0], lo[a1]),
        (hi[a0], hi[a1]),
        (lo[a0], hi[a1]),
    ):
        p = list(face)
        p[a0] = v0
        p[a1] = v1
        corners.append((p[0], p[1], p[2]))
    return plane, corners


def derive_entity_top_surfaces(
    entities: list[EntityArtifact],
    *,
    frame: SceneFrame,
    support_class_allowlist: tuple[str, ...] = DEFAULT_SUPPORT_CLASS_ALLOWLIST,
) -> list[EntitySurface]:
    """Derive one top support surface for each support-capable entity."""
    surfaces: list[EntitySurface] = []
    for entity in entities:
        owner_class = support_class_for_entity(entity, support_class_allowlist)
        if owner_class is None:
            continue
        plane, polygon = _top_face_from_aabb(entity.bbox_aabb, frame=frame)
        confidence = (
            float(entity.semantic_hypotheses[0].confidence)
            if entity.semantic_hypotheses else 1.0
        )
        owner_uid = entity.identity.object_uid
        surfaces.append(EntitySurface(
            surface_uid=f"ent_surf_{owner_uid}_top",
            owner_entity_uid=owner_uid,
            owner_class=owner_class,
            plane=plane,
            polygon=polygon,
            confidence=confidence,
        ))
    return surfaces

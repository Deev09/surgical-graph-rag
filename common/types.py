"""Shared primitive types used across stages.

Pure data only. No business logic. Imported by everything; imports nothing
from sibling stage modules. Keeps the dependency graph acyclic between
adapters / representations / extractors / graph / reasoner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

JSON = Any  # values produced by json.loads — dict, list, str, int, float, bool, None
Vec3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]  # (x, y, z, w)


@dataclass(frozen=True)
class Plane:
    a: float
    b: float
    c: float
    d: float


@dataclass(frozen=True)
class OrientedBBox:
    center: Vec3
    extents: Vec3
    rotation_quat: Quaternion


@dataclass(frozen=True)
class CameraPose:
    camera_id: str
    position: Vec3
    rotation_quat: Quaternion
    intrinsics: tuple[float, float, float, float]
    width: int
    height: int


@dataclass(frozen=True)
class SceneFrame:
    gravity: Vec3
    canonical_forward: Vec3 | None
    canonical_right: Vec3 | None
    units: Literal["meters"]
    notes: str

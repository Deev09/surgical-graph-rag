"""First-pass geometric relation extractor.

Internal sanity check only. Reads the same `objects` array v1 already uses
and emits an edges list in the same shape as `scene_graph.json["relations"]`.

Conventions (read off the v1 scene; not generalized):
    x small  -> LEFT, x large  -> RIGHT
    y small  -> BEHIND, y large -> IN_FRONT_OF
    z small  -> BELOW, z large -> ABOVE

Rules (no tuning):
    - Directional edge emitted only on the dominant axis (max |delta|).
    - Axis delta must exceed MIN_DELTA.
    - NEAR is symmetric, undirected by axis: euclidean distance < NEAR_THRESHOLD.
    - ATTACHED_TO is not geometric and is intentionally NOT emitted.
"""

from __future__ import annotations

from math import sqrt
from typing import Iterable

MIN_DELTA = 0.3
NEAR_THRESHOLD = 1.0

DIRECTIONAL_TYPES = (
    "LEFT_OF",
    "RIGHT_OF",
    "BEHIND",
    "IN_FRONT_OF",
    "BELOW",
    "ABOVE",
)
SYMMETRIC_TYPES = ("NEAR",)
SKIPPED_TYPES = ("ATTACHED_TO",)


def _euclid(a: Iterable[float], b: Iterable[float]) -> float:
    return sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _directional_edge(a: dict, b: dict) -> dict | None:
    ax, ay, az = a["xyz"]
    bx, by, bz = b["xyz"]
    dx, dy, dz = ax - bx, ay - by, az - bz
    abs_dx, abs_dy, abs_dz = abs(dx), abs(dy), abs(dz)
    dominant = max(abs_dx, abs_dy, abs_dz)
    if dominant < MIN_DELTA:
        return None
    if dominant == abs_dx:
        rel = "LEFT_OF" if dx < 0 else "RIGHT_OF"
    elif dominant == abs_dy:
        rel = "BEHIND" if dy < 0 else "IN_FRONT_OF"
    else:
        rel = "BELOW" if dz < 0 else "ABOVE"
    return {"source": a["id"], "type": rel, "target": b["id"]}


def compute_relations(objects: list[dict]) -> list[dict]:
    edges: list[dict] = []
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i == j:
                continue
            edge = _directional_edge(a, b)
            if edge is not None:
                edges.append(edge)
            if i < j and _euclid(a["xyz"], b["xyz"]) < NEAR_THRESHOLD:
                edges.append({"source": a["id"], "type": "NEAR", "target": b["id"]})
                edges.append({"source": b["id"], "type": "NEAR", "target": a["id"]})
    return edges

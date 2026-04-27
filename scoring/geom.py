"""Geometry helpers extracted from tiny_graph_demo. Behavior identical."""

from __future__ import annotations

import math
from typing import Any


def _get_xyz(node: dict[str, Any]) -> tuple[float, float, float]:
    p = node.get("xyz")
    if not p or len(p) < 3:
        return (0.0, 0.0, 0.0)
    return (float(p[0]), float(p[1]), float(p[2]))


def _dist_xy(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dist3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _find_anchor_node(scene_graph: list[dict[str, Any]], anchor: str) -> dict[str, Any] | None:
    from tiny_graph_demo import _label_equiv  # lazy: avoids import cycle

    for n in scene_graph:
        if _label_equiv(str(n["label"]), anchor):
            return n
    return None

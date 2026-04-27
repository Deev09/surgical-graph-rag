"""Spatial scoring helpers extracted from tiny_graph_demo.

Behavior identical, including the LEFT_OF / RIGHT_OF sign-bug in
_geometric_spatial_bonus where the inner max(0, ...) clips the distance
penalty to zero on the directional branches. NOT FIXED HERE — fix lives
in scorer-v2.
"""

from __future__ import annotations

from typing import Any

from scoring.geom import _dist3, _dist_xy, _get_xyz

# Horizontal salience for BELOW/ABOVE: penalizes candidates far from the
# anchor in the floor plane (reduces false positives like toilet vs sink for
# "below the mirror"). Bathroom-tuned values.
SPATIAL_XY_SALIENCE_LAMBDA = 0.38
SPATIAL_XY_SALIENCE_FLOOR = 0.05


def _best_spatial_relation_weight(
    relations: list[Any],
    spatial_type: str,
    anchor: str,
) -> float:
    from tiny_graph_demo import _label_equiv  # lazy

    best = 0.0
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        if rel.get("type") != spatial_type:
            continue
        tgt = rel.get("target")
        if tgt is None or not _label_equiv(str(anchor), str(tgt)):
            continue
        w = float(rel.get("weight", 1.0))
        best = max(best, w)
    return best


def _spatial_xy_salience(
    node: dict[str, Any],
    spatial_type: str,
    anchor_node: dict[str, Any],
) -> float:
    """Planar proximity factor for vertical relations (v1 spatial salience)."""
    if spatial_type not in ("BELOW", "ABOVE"):
        return 1.0
    na = _get_xyz(node)
    nb = _get_xyz(anchor_node)
    d_xy = _dist_xy(na, nb)
    raw = 1.0 - SPATIAL_XY_SALIENCE_LAMBDA * d_xy
    return max(SPATIAL_XY_SALIENCE_FLOOR, min(1.0, raw))


def _geometric_spatial_bonus(
    node: dict[str, Any],
    spatial_type: str,
    anchor_node: dict[str, Any],
) -> float:
    """Small tie-breaker from coarse xyz; relation weights stay primary."""
    na = _get_xyz(node)
    nb = _get_xyz(anchor_node)
    d_xy = _dist_xy(na, nb)
    d3 = _dist3(na, nb)

    if spatial_type == "BELOW":
        dz = nb[2] - na[2]
        vertical = max(0.0, min(0.22, 0.05 * dz))
        horiz = max(0.0, 0.16 - 0.04 * d_xy)
        return vertical + horiz
    if spatial_type == "ABOVE":
        dz = na[2] - nb[2]
        return max(0.0, min(0.22, 0.05 * dz))
    if spatial_type == "LEFT_OF":
        return max(0.0, 0.14 - 0.05 * max(0.0, na[0] - nb[0])) if na[0] < nb[0] else 0.0
    if spatial_type == "RIGHT_OF":
        return max(0.0, 0.14 - 0.05 * max(0.0, nb[0] - na[0])) if na[0] > nb[0] else 0.0
    if spatial_type == "IN_FRONT_OF":
        return max(0.0, 0.14 - 0.05 * max(0.0, na[1] - nb[1])) if na[1] > nb[1] else 0.0
    if spatial_type == "BEHIND":
        return max(0.0, 0.14 - 0.05 * max(0.0, nb[1] - na[1])) if na[1] < nb[1] else 0.0
    if spatial_type == "NEAR":
        return max(0.0, 0.2 - 0.055 * d3)
    return 0.0

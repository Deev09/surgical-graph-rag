"""Scorer v2 — A/B candidate against scoring.v1.

First-pass hypothesis: fix the LEFT_OF / RIGHT_OF locality sign bug in
scoring.spatial._geometric_spatial_bonus and strengthen the locality bonus
so dense computed graphs don't degenerate to alphabetical ties.

Scope:
    - Only LEFT_OF and RIGHT_OF formulas are changed.
    - All other relation types (ABOVE, BELOW, IN_FRONT_OF, BEHIND, NEAR)
      delegate unchanged to v1.
    - All non-describe_spatial intents delegate unchanged to v1.
    - No constants in v1 / spatial / geom / filter / topk are modified.
"""

from __future__ import annotations

from typing import Any

from scoring.geom import _find_anchor_node, _get_xyz
from scoring.spatial import (
    _best_spatial_relation_weight,
    _geometric_spatial_bonus,
    _spatial_xy_salience,
)
from scoring.v1 import score_node as _score_node_v1


# Linear directional locality. Bonus = alpha at d=0, decreases linearly,
# clipped to 0 at d >= scale. Replaces v1's sign-bugged constant-0.14 branch
# for LEFT_OF / RIGHT_OF.
SPATIAL_DIRECTIONAL_ALPHA = 0.5
SPATIAL_DIRECTIONAL_SCALE = 5.0


def _directional_locality_v2(d: float) -> float:
    return SPATIAL_DIRECTIONAL_ALPHA * max(0.0, 1.0 - d / SPATIAL_DIRECTIONAL_SCALE)


def _geometric_spatial_bonus_v2(
    node: dict[str, Any],
    spatial_type: str,
    anchor_node: dict[str, Any],
) -> float:
    """v2 geometric bonus. Only LEFT_OF / RIGHT_OF differ from v1."""
    if spatial_type == "LEFT_OF":
        na = _get_xyz(node)
        nb = _get_xyz(anchor_node)
        if na[0] >= nb[0]:
            return 0.0
        return _directional_locality_v2(nb[0] - na[0])
    if spatial_type == "RIGHT_OF":
        na = _get_xyz(node)
        nb = _get_xyz(anchor_node)
        if na[0] <= nb[0]:
            return 0.0
        return _directional_locality_v2(na[0] - nb[0])
    return _geometric_spatial_bonus(node, spatial_type, anchor_node)


def score_node(
    node: dict[str, Any],
    parsed: Any,
    *,
    scene_graph: list[dict[str, Any]] | None = None,
) -> float:
    """v2 score_node. Spatial branch uses _geometric_spatial_bonus_v2;
    every other intent delegates to scoring.v1.score_node verbatim."""
    if (
        parsed.intent == "describe_spatial"
        and parsed.spatial_relation
        and parsed.relation_anchor
    ):
        from tiny_graph_demo import scene as default_scene  # lazy

        rels = list(node.get("relations", []))
        graph = scene_graph if scene_graph is not None else default_scene
        w = _best_spatial_relation_weight(rels, parsed.spatial_relation, parsed.relation_anchor)
        if w <= 0:
            return 0.0
        anchor_n = _find_anchor_node(graph, parsed.relation_anchor)
        geo = (
            _geometric_spatial_bonus_v2(node, parsed.spatial_relation, anchor_n)
            if anchor_n
            else 0.0
        )
        base = w + geo
        if anchor_n:
            base *= _spatial_xy_salience(node, parsed.spatial_relation, anchor_n)
        return base

    return _score_node_v1(node, parsed, scene_graph=scene_graph)

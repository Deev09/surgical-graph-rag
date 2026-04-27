"""V1 scoring function extracted from tiny_graph_demo. Bit-identical behavior."""

from __future__ import annotations

from typing import Any

from scoring.geom import _find_anchor_node
from scoring.spatial import (
    _best_spatial_relation_weight,
    _geometric_spatial_bonus,
    _spatial_xy_salience,
)


def score_node(
    node: dict[str, Any],
    parsed: Any,
    *,
    scene_graph: list[dict[str, Any]] | None = None,
) -> float:
    from tiny_graph_demo import (  # lazy to avoid import cycle
        _label_equiv,
        _structured_rel_matches,
        normalize_text,
        scene as default_scene,
    )

    score = 0.0
    rels: list[Any] = list(node.get("relations", []))

    node_label = normalize_text(str(node["label"]).replace("_", " "))
    node_type = normalize_text(node["attributes"].get("type", ""))
    node_zone = str(node["zone"])
    node_zone_norm = normalize_text(node_zone.replace("_", " "))
    node_color = normalize_text(node["attributes"].get("color", ""))
    graph = scene_graph if scene_graph is not None else default_scene

    if parsed.intent == "describe_spatial" and parsed.spatial_relation and parsed.relation_anchor:
        w = _best_spatial_relation_weight(rels, parsed.spatial_relation, parsed.relation_anchor)
        if w <= 0:
            return 0.0
        anchor_n = _find_anchor_node(graph, parsed.relation_anchor)
        geo = _geometric_spatial_bonus(node, parsed.spatial_relation, anchor_n) if anchor_n else 0.0
        base = w + geo
        if anchor_n:
            base *= _spatial_xy_salience(node, parsed.spatial_relation, anchor_n)
        return base

    if parsed.intent == "describe_zone" and parsed.zone_family:
        zf = parsed.zone_family
        if zf == "right_wall" and node_zone in ("right_wall", "right_brick_wall"):
            score += 1.0
        elif zf == "right_brick_wall" and node_zone == "right_brick_wall":
            score += 1.05
        elif zf == "left_wall" and node_zone == "left_wall":
            score += 1.0
        elif zf == "back_wall" and node_zone == "back_wall":
            score += 1.0
        elif zf == "front_right" and (
            node_zone == "front_right" or node_zone.startswith("front_right_")
        ):
            score += 0.9
        elif zf == "ceiling_center" and node_zone == "ceiling_center":
            score += 1.0
        return score

    if parsed.intent == "describe_near_zone" and parsed.zone_family == "back_wall":
        if node_zone == "back_wall":
            score += 1.0
        elif _structured_rel_matches(rels, want_type="NEAR", anchor="black_door"):
            for rel in rels:
                if isinstance(rel, dict) and rel.get("type") == "NEAR" and _label_equiv(
                    "black_door", str(rel.get("target", ""))
                ):
                    score += 0.95 * float(rel.get("weight", 1.0))
                    break
        elif "back" in node_zone_norm and "wall" in node_zone_norm:
            score += 0.35
        return score

    label_hit = False
    if parsed.target_label:
        tl = normalize_text(parsed.target_label.replace("_", " "))
        if tl == node_label or tl in node_label or node_label in tl:
            score += 0.75
            label_hit = True

    if not label_hit and parsed.target_type and parsed.target_type == node_type and not parsed.target_label:
        score += 0.30

    if parsed.color and parsed.color == node_color:
        score += 0.25

    if parsed.zone and parsed.zone == node_zone:
        score += 0.20

    if parsed.anchor_objects:
        for anchor in parsed.anchor_objects:
            for rel in rels:
                if isinstance(rel, dict):
                    rt = str(rel.get("type", ""))
                    tgt = str(rel.get("target", ""))
                    if rt == "NEAR" and _label_equiv(anchor, tgt):
                        score += 0.45 * float(rel.get("weight", 1.0))
                        break
                    if _label_equiv(anchor, tgt):
                        score += 0.22 * float(rel.get("weight", 1.0))
                        break
                else:
                    rel_norm = normalize_text(str(rel))
                    anchor_norm = normalize_text(anchor)
                    if f"near {anchor_norm}" in rel_norm or anchor_norm in rel_norm:
                        score += 0.28
                        break

    return score

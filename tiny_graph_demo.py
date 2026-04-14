from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any

# Docker Model Runner OpenAI-compatible API (host TCP). Override if needed:
#   export CONTEXT_SURGEON_OPENAI_BASE_URL=http://localhost:12434/engines/vllm/v1
#   export CONTEXT_SURGEON_MODEL=ai/llama3.1
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[misc, assignment]


# ----------------------------
# 1) Scene graph (v1: graffiti bathroom, human-interpreted coarse graph)
# ----------------------------

# Approximate room frame (meters): +x right, +y toward front of room, +z up from floor.
# Coarse placeholders until scene reconstruction feeds real poses; good enough to break ties
# between same relation type (e.g. two objects both BELOW mirror). Later you can add per-object
# "bbox": [[xmin, ymin, zmin], [xmax, ymax, zmax]] and pass it through scene_nodes_from_record.
GRAFFITI_BATHROOM: dict[str, Any] = {
    "scene": "graffiti_bathroom",
    "objects": [
        {
            "id": "obj_1",
            "label": "toilet",
            "zone": "center_left",
            "xyz": [1.1, 1.4, 0.45],
            "attributes": {"type": "fixture", "color": "white", "mounted": "floor"},
        },
        {
            "id": "obj_2",
            "label": "sink",
            "zone": "front_right",
            "xyz": [3.15, 2.55, 0.9],
            "attributes": {"type": "fixture", "color": "white", "mounted": "wall"},
        },
        {
            "id": "obj_3",
            "label": "mirror",
            "zone": "front_right_wall",
            "xyz": [3.2, 2.7, 1.55],
            "attributes": {"type": "reflective_fixture", "mounted": "wall"},
        },
        {
            "id": "obj_4",
            "label": "floor_drain",
            "zone": "floor_center",
            "xyz": [2.0, 1.35, 0.02],
            "attributes": {"type": "drain", "mounted": "floor"},
        },
        {
            "id": "obj_5",
            "label": "vending_machine",
            "zone": "left_front_wall",
            "xyz": [0.35, 2.2, 1.3],
            "attributes": {"type": "machine", "mounted": "wall"},
        },
        {
            "id": "obj_6",
            "label": "trash_can",
            "zone": "back_left_floor",
            "xyz": [0.5, 0.45, 0.25],
            "attributes": {"type": "container", "color": "black", "mounted": "floor"},
        },
        {
            "id": "obj_7",
            "label": "paper_towel_dispenser",
            "zone": "right_brick_wall",
            "xyz": [3.75, 2.0, 1.25],
            "attributes": {"type": "dispenser", "mounted": "wall"},
        },
        {
            "id": "obj_8",
            "label": "black_door",
            "zone": "back_wall",
            "xyz": [2.0, 0.15, 1.0],
            "attributes": {"type": "door", "color": "black"},
        },
        {
            "id": "obj_9",
            "label": "wall_dispenser",
            "zone": "front_right_wall",
            "xyz": [3.25, 2.65, 1.2],
            "attributes": {"type": "dispenser", "mounted": "wall"},
        },
        {
            "id": "obj_10",
            "label": "ceiling_vent",
            "zone": "ceiling_center",
            "xyz": [2.0, 1.5, 2.35],
            "attributes": {"type": "vent", "mounted": "ceiling"},
        },
        {
            "id": "obj_11",
            "label": "left_grab_bar",
            "zone": "left_wall",
            "xyz": [0.2, 1.35, 0.85],
            "attributes": {"type": "rail", "mounted": "wall"},
        },
        {
            "id": "obj_12",
            "label": "right_grab_bar",
            "zone": "right_wall",
            "xyz": [3.55, 2.1, 0.8],
            "attributes": {"type": "rail", "mounted": "wall"},
        },
    ],
    "relations": [
        {"source": "obj_1", "type": "LEFT_OF", "target": "obj_2"},
        {"source": "obj_1", "type": "BELOW", "target": "obj_3", "weight": 0.45},
        {"source": "obj_2", "type": "BELOW", "target": "obj_3", "weight": 1.0},
        {"source": "obj_2", "type": "RIGHT_OF", "target": "obj_1"},
        {"source": "obj_4", "type": "IN_FRONT_OF", "target": "obj_1"},
        {"source": "obj_4", "type": "BELOW", "target": "obj_2", "weight": 0.85},
        {"source": "obj_5", "type": "LEFT_OF", "target": "obj_1"},
        {"source": "obj_6", "type": "LEFT_OF", "target": "obj_1"},
        {"source": "obj_6", "type": "NEAR", "target": "obj_8", "weight": 1.0},
        {"source": "obj_7", "type": "RIGHT_OF", "target": "obj_2"},
        {"source": "obj_7", "type": "ATTACHED_TO", "target": "obj_12"},
        {"source": "obj_8", "type": "BEHIND", "target": "obj_1"},
        {"source": "obj_8", "type": "BEHIND", "target": "obj_2"},
        {"source": "obj_9", "type": "RIGHT_OF", "target": "obj_3"},
        {"source": "obj_10", "type": "ABOVE", "target": "obj_4"},
        {"source": "obj_11", "type": "LEFT_OF", "target": "obj_1"},
        {"source": "obj_12", "type": "RIGHT_OF", "target": "obj_2"},
    ],
}


def scene_nodes_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand global edges into per-object relations: {type, target, weight?}; targets use labels."""
    id_to_label = {o["id"]: o["label"] for o in record["objects"]}
    by_source: dict[str, list[dict[str, Any]]] = {}
    for edge in record["relations"]:
        src, typ, tgt = edge["source"], edge["type"], edge["target"]
        w = float(edge.get("weight", 1.0))
        rel: dict[str, Any] = {"type": typ, "target": id_to_label[tgt], "weight": w}
        by_source.setdefault(src, []).append(rel)
    nodes: list[dict[str, Any]] = []
    for o in record["objects"]:
        oid = o["id"]
        xyz = o.get("xyz")
        node: dict[str, Any] = {
            "id": oid,
            "label": o["label"],
            "zone": o["zone"],
            "attributes": dict(o["attributes"]),
            "relations": list(by_source.get(oid, [])),
        }
        if xyz is not None:
            node["xyz"] = [float(x) for x in xyz]
        if "bbox" in o:
            node["bbox"] = o["bbox"]
        nodes.append(node)
    return nodes


SCENE_ID: str = GRAFFITI_BATHROOM["scene"]
scene: list[dict[str, Any]] = scene_nodes_from_record(GRAFFITI_BATHROOM)


# ----------------------------
# 2) Query parser
# ----------------------------

KNOWN_COLORS = {"white", "black", "silver", "red", "wooden"}

# Natural phrase -> zone id or family (substring match on node zone)
ZONE_PHRASES: list[tuple[str, str]] = [
    ("back wall", "back_wall"),
    ("right brick wall", "right_brick_wall"),
    ("right wall", "right_wall"),  # also matches right_brick_wall via prefix
    ("left wall", "left_wall"),
    ("front right", "front_right"),
    ("ceiling", "ceiling_center"),
]

# (regex with one capture, spatial relation type)
SPATIAL_QUESTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"what(?:'s| is) (?:to the )?left of (?:the )?([a-z0-9 _]+?)(?:\?|$)"), "LEFT_OF"),
    (re.compile(r"what(?:'s| is) (?:to the )?right of (?:the )?([a-z0-9 _]+?)(?:\?|$)"), "RIGHT_OF"),
    (re.compile(r"what(?:'s| is) below (?:the )?([a-z0-9 _]+?)(?:\?|$)"), "BELOW"),
    (re.compile(r"what(?:'s| is) under (?:the )?([a-z0-9 _]+?)(?:\?|$)"), "BELOW"),
    (re.compile(r"what(?:'s| is) above (?:the )?([a-z0-9 _]+?)(?:\?|$)"), "ABOVE"),
    (re.compile(r"what(?:'s| is) in front of (?:the )?([a-z0-9 _]+?)(?:\?|$)"), "IN_FRONT_OF"),
    (re.compile(r"what(?:'s| is) behind (?:the )?([a-z0-9 _]+?)(?:\?|$)"), "BEHIND"),
]


def _object_labels_for_matching(graph: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for n in graph:
        lab = str(n["label"])
        labels.append(lab.replace("_", " "))
        if "_" in lab:
            labels.append(lab)
    return sorted(set(labels), key=len, reverse=True)


SCENE_LABEL_PHRASES: list[str] = _object_labels_for_matching(scene)


@dataclass
class ParsedQuery:
    raw_query: str
    intent: str
    target_label: str | None
    target_type: str | None
    color: str | None
    zone: str | None
    anchor_objects: list[str]
    spatial_relation: str | None
    relation_anchor: str | None
    zone_family: str | None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _match_label_phrase(q: str) -> tuple[str | None, str | None]:
    """Return (matched_phrase, coarse_type from attributes) if query mentions a known object label."""
    for phrase in SCENE_LABEL_PHRASES:
        pnorm = normalize_text(phrase)
        if pnorm and pnorm in q:
            for node in scene:
                if normalize_text(str(node["label"]).replace("_", " ")) == pnorm or normalize_text(
                    str(node["label"])
                ) == pnorm.replace(" ", "_"):
                    t = normalize_text(node["attributes"].get("type", ""))
                    return phrase, t or None
            return phrase, None
    return None, None


def _zone_family_from_query(q: str) -> str | None:
    for phrase, family in ZONE_PHRASES:
        if phrase in q:
            return family
    return None


def parse_query(query: str) -> ParsedQuery:
    q = normalize_text(query)

    intent = "find_object"
    if "where" in q:
        intent = "find_location"
    elif "what" in q:
        intent = "describe"

    color = next((c for c in KNOWN_COLORS if c in q), None)
    zone = None
    spatial_relation: str | None = None
    relation_anchor: str | None = None
    zone_family = _zone_family_from_query(q)

    target_label = None
    target_type = None
    anchor_objects: list[str] = []

    # "What is on the right wall?" / on the <zone phrase>
    if "on the" in q and "wall" in q:
        intent = "describe_zone"
        zf = _zone_family_from_query(q)
        if zf:
            zone_family = zf
        return ParsedQuery(
            raw_query=query,
            intent=intent,
            target_label=None,
            target_type=None,
            color=color,
            zone=zone,
            anchor_objects=[],
            spatial_relation=None,
            relation_anchor=None,
            zone_family=zone_family,
        )

    # Spatial template: "What is left of the sink?"
    for pattern, rel_type in SPATIAL_QUESTION_PATTERNS:
        m = pattern.search(q)
        if m:
            intent = "describe_spatial"
            spatial_relation = rel_type
            relation_anchor = m.group(1).strip()
            return ParsedQuery(
                raw_query=query,
                intent=intent,
                target_label=None,
                target_type=None,
                color=color,
                zone=zone,
                anchor_objects=[],
                spatial_relation=spatial_relation,
                relation_anchor=relation_anchor,
                zone_family=zone_family,
            )

    # "What is near the back wall?" -> zone-centric
    if ("what is near" in q or "what's near" in q) and zone_family:
        intent = "describe_near_zone"
        return ParsedQuery(
            raw_query=query,
            intent=intent,
            target_label=None,
            target_type=None,
            color=color,
            zone=zone,
            anchor_objects=[],
            spatial_relation=None,
            relation_anchor=None,
            zone_family=zone_family,
        )

    # "What is near the <object>" (e.g. door)
    if q.startswith("what is near") or q.startswith("what's near"):
        intent = "describe_near_anchor"
        near_match = re.search(r"near\s+the\s+([a-z0-9 _]+)|near\s+([a-z0-9 _]+)", q)
        if near_match:
            anchor = near_match.group(1) or near_match.group(2)
            if anchor:
                anchor_objects.append(anchor.strip())

        return ParsedQuery(
            raw_query=query,
            intent=intent,
            target_label=None,
            target_type=None,
            color=color,
            zone=zone,
            anchor_objects=anchor_objects,
            spatial_relation=None,
            relation_anchor=None,
            zone_family=zone_family,
        )

    phrase, ptype = _match_label_phrase(q)
    if phrase:
        target_label = phrase
        target_type = ptype

    near_match = re.search(r"near\s+the\s+([a-z0-9 _]+)|near\s+([a-z0-9 _]+)", q)
    if near_match:
        anchor = near_match.group(1) or near_match.group(2)
        if anchor:
            anchor_objects.append(anchor.strip())

    return ParsedQuery(
        raw_query=query,
        intent=intent,
        target_label=target_label,
        target_type=target_type,
        color=color,
        zone=zone,
        anchor_objects=anchor_objects,
        spatial_relation=spatial_relation,
        relation_anchor=relation_anchor,
        zone_family=zone_family,
    )


def _label_equiv(a: str, b: str) -> bool:
    na = normalize_text(a.replace("_", " "))
    nb = normalize_text(b.replace("_", " "))
    if na == nb:
        return True
    if na in nb or nb in na:
        return min(len(na), len(nb)) >= 4
    return False


def _structured_rel_matches(
    relations: list[Any],
    *,
    want_type: str | None = None,
    anchor: str | None = None,
) -> bool:
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        if want_type and rel.get("type") != want_type:
            continue
        tgt = rel.get("target")
        if anchor is None:
            return True
        if tgt is not None and _label_equiv(str(anchor), str(tgt)):
            return True
    return False


def _best_spatial_relation_weight(
    relations: list[Any],
    spatial_type: str,
    anchor: str,
) -> float:
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


def _get_xyz(node: dict[str, Any]) -> tuple[float, float, float]:
    p = node.get("xyz")
    if not p or len(p) < 3:
        return (0.0, 0.0, 0.0)
    return (float(p[0]), float(p[1]), float(p[2]))


def _find_anchor_node(scene_graph: list[dict[str, Any]], anchor: str) -> dict[str, Any] | None:
    for n in scene_graph:
        if _label_equiv(str(n["label"]), anchor):
            return n
    return None


def _dist_xy(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dist3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


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


def score_node(
    node: dict[str, Any],
    parsed: ParsedQuery,
    *,
    scene_graph: list[dict[str, Any]] | None = None,
) -> float:
    score = 0.0
    rels: list[Any] = list(node.get("relations", []))

    node_label = normalize_text(str(node["label"]).replace("_", " "))
    node_type = normalize_text(node["attributes"].get("type", ""))
    node_zone = str(node["zone"])
    node_zone_norm = normalize_text(node_zone.replace("_", " "))
    node_color = normalize_text(node["attributes"].get("color", ""))
    graph = scene_graph if scene_graph is not None else scene

    if parsed.intent == "describe_spatial" and parsed.spatial_relation and parsed.relation_anchor:
        w = _best_spatial_relation_weight(rels, parsed.spatial_relation, parsed.relation_anchor)
        if w <= 0:
            return 0.0
        anchor_n = _find_anchor_node(graph, parsed.relation_anchor)
        geo = _geometric_spatial_bonus(node, parsed.spatial_relation, anchor_n) if anchor_n else 0.0
        return w + geo

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

    # Target label / type (e.g. "where is the trash can")
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


def candidate_filter(
    scene_graph: list[dict[str, Any]],
    parsed: ParsedQuery,
    min_score: float = 0.20
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for node in scene_graph:
        node_label_norm = normalize_text(str(node["label"]).replace("_", " "))

        if parsed.intent == "describe_near_anchor" and parsed.anchor_objects:
            if any(_label_equiv(node_label_norm, anchor) for anchor in parsed.anchor_objects):
                continue

        if parsed.intent == "describe_spatial" and parsed.relation_anchor:
            if _label_equiv(node_label_norm, parsed.relation_anchor):
                continue

        score = score_node(node, parsed, scene_graph=scene_graph)
        if score >= min_score:
            node_copy = dict(node)
            node_copy["score"] = round(score, 3)
            candidates.append(node_copy)

    return candidates


# ----------------------------
# 4) Top-k prune
# ----------------------------

def top_k_prune(candidates: list[dict[str, Any]], k: int = 3) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:k]


# ----------------------------
# 5) Compact JSON builder
# ----------------------------

def build_compact_context(parsed: ParsedQuery, pruned_nodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scene": SCENE_ID,
        "intent": parsed.intent,
        "query": parsed.raw_query,
        "target": {
            "label": parsed.target_label,
            "type": parsed.target_type,
            "color": parsed.color,
            "zone": parsed.zone,
            "anchor_objects": parsed.anchor_objects,
            "spatial_relation": parsed.spatial_relation,
            "relation_anchor": parsed.relation_anchor,
            "zone_family": parsed.zone_family,
        },
        "candidates": [
            {
                "id": node["id"],
                "label": node["label"],
                "zone": node["zone"],
                "xyz": node.get("xyz"),
                "bbox": node.get("bbox"),
                "attributes": node["attributes"],
                "relations": node["relations"][:8],
                "score": node["score"],
            }
            for node in pruned_nodes
        ],
    }


# ----------------------------
# 6) LLM prompt + Docker Model Runner (OpenAI-compatible)
# ----------------------------

def build_user_prompt(compact_context: dict[str, Any], question: str) -> str:
    evidence_json = json.dumps(compact_context, separators=(",", ":"))
    intent = compact_context.get("intent", "")

    if intent in (
        "describe_near_anchor",
        "describe_near_zone",
        "describe_zone",
        "describe_spatial",
    ):
        answer_style = (
            "List all candidate objects mentioned in the evidence.\n"
            "Do not omit any candidate unless it is clearly irrelevant.\n"
            "Do not mention object IDs, internal scores, or ranking numbers.\n"
            "Answer naturally in 1-2 sentences."
        )
    else:
        answer_style = (
            "Answer using only the evidence.\n"
            "If the evidence is insufficient, say so clearly.\n"
            "Do not mention object IDs, internal scores, or ranking numbers unless explicitly asked.\n"
            "Prefer natural language answers.\n"
            "If there are multiple plausible matches, say that clearly.\n"
            "Answer in 2 sentences max."
        )

    return (
        "You are helping with a spatial graph project.\n"
        "Use only the evidence provided.\n"
        f"{answer_style}\n"
        "\n"
        f"Evidence:\n{evidence_json}\n"
        "\n"
        f"Question:\n{question}\n"
    )


def answer_with_local_model(user_prompt: str) -> str:
    if OpenAI is None:
        raise RuntimeError("Install dependencies: pip install -r requirements.txt")
    base_url = os.environ.get(
        "CONTEXT_SURGEON_OPENAI_BASE_URL",
        "http://localhost:12434/engines/v1",
    )
    model = os.environ.get("CONTEXT_SURGEON_MODEL", "ai/llama3.1")
    client = OpenAI(base_url=base_url, api_key="not-needed")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.2,
        max_tokens=256,
    )
    choice = response.choices[0].message
    content = choice.content
    if not content:
        return ""
    return content.strip()


# ----------------------------
# 7) Demo runner
# ----------------------------

def top_k_for_intent(parsed: ParsedQuery) -> int:
    if parsed.intent in ("describe_near_anchor", "describe_near_zone", "describe_zone"):
        return 10
    if parsed.intent == "describe_spatial":
        return 6
    return 4


# ----------------------------
# 8) Expected-answer benchmark (pruning quality)
# ----------------------------

# Each value is a list of acceptable labels or zone ids for that query.
EXPECTED_ANSWERS: dict[str, list[str]] = {
    "What is left of the sink?": ["toilet"],
    "What is below the mirror?": ["sink"],
    "What is near the back wall?": ["trash_can", "black_door"],
    "What is on the right wall?": ["paper_towel_dispenser", "right_grab_bar"],
    "Where is the trash can?": ["back_left_floor"],
    "What is in front of the toilet?": ["floor_drain"],
    "Where is the vending machine?": ["left_front_wall"],
    "What is right of the toilet?": ["sink"],
    "What is behind the toilet?": ["black_door"],
    "Where is the paper towel dispenser?": ["right_brick_wall"],
}


def node_matches_expectation(node: dict[str, Any], exp: str) -> bool:
    en = normalize_text(exp.replace("_", " "))
    z = normalize_text(str(node["zone"]).replace("_", " "))
    if z == en:
        return True
    nl = normalize_text(str(node["label"]).replace("_", " "))
    if nl == en:
        return True
    if len(en) >= 4 and (en in nl or nl in en):
        return True
    return False


def run_benchmark(
    cases: dict[str, list[str]] | None = None,
    *,
    k_override: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    cases = cases or EXPECTED_ANSWERS
    top1_hits = 0
    topk_hits = 0
    total_fp = 0
    n = 0
    rows: list[dict[str, Any]] = []
    for query, expected in cases.items():
        n += 1
        parsed = parse_query(query)
        k_use = k_override if k_override is not None else top_k_for_intent(parsed)
        candidates = candidate_filter(scene, parsed)
        pruned = top_k_prune(candidates, k=k_use)
        labels_pruned = [str(p["label"]) for p in pruned]

        def matches_any(node: dict[str, Any]) -> bool:
            return any(node_matches_expectation(node, e) for e in expected)

        top1_ok = bool(pruned) and matches_any(pruned[0])
        topk_ok = any(matches_any(p) for p in pruned)
        fp = sum(1 for p in pruned if not matches_any(p))

        top1_hits += int(top1_ok)
        topk_hits += int(topk_ok)
        total_fp += fp
        rows.append(
            {
                "query": query,
                "top1_ok": top1_ok,
                "topk_ok": topk_ok,
                "false_positives": fp,
                "top1_label": labels_pruned[0] if labels_pruned else None,
                "expected": expected,
                "pruned_labels": labels_pruned,
            }
        )

    summary = {
        "n_queries": n,
        "top1_accuracy": top1_hits / n if n else 0.0,
        "topk_recall": topk_hits / n if n else 0.0,
        "avg_false_positives_per_query": total_fp / n if n else 0.0,
        "rows": rows,
    }
    if verbose:
        print("Benchmark (pruning vs expected)\n")
        for r in rows:
            t1 = "OK" if r["top1_ok"] else "miss"
            tk = "OK" if r["topk_ok"] else "miss"
            print(f"Q: {r['query']}")
            print(f"   top-1: {t1}  (got {r['top1_label']!r}, want any of {r['expected']})")
            print(f"   top-k: {tk}  fp_in_k={r['false_positives']}  k={len(r['pruned_labels'])}")
            print(f"   pruned: {r['pruned_labels']}\n")
        print(
            f"Summary: top1_acc={summary['top1_accuracy']:.2%}  "
            f"topk_recall={summary['topk_recall']:.2%}  "
            f"avg_fp@k={summary['avg_false_positives_per_query']:.2f}"
        )
    return summary


def run_demo(query: str, *, send_to_llm: bool = True, verbose: bool = True) -> None:
    parsed = parse_query(query)
    candidates = candidate_filter(scene, parsed)
    pruned = top_k_prune(candidates, k=top_k_for_intent(parsed))
    compact_context = build_compact_context(parsed, pruned)

    if verbose:
        print("\nPARSED QUERY")
        print(parsed)

        print("\nCANDIDATES")
        print(json.dumps(candidates, indent=2))

    if not send_to_llm or os.environ.get("SKIP_LLM"):
        print("\nPRUNED CONTEXT (compact JSON)")
        print(json.dumps(compact_context, indent=2))
        return

    user_prompt = build_user_prompt(compact_context, query)
    print("\nLLM ANSWER")
    try:
        print(answer_with_local_model(user_prompt))
    except Exception as exc:
        print(f"Could not reach local model ({exc}).")
        print("Check Docker Model Runner TCP (port 12434) and CONTEXT_SURGEON_OPENAI_BASE_URL.")
        print("Falling back to compact JSON only:\n")
        print(json.dumps(compact_context, indent=2))


if __name__ == "__main__":
    test_queries = [
        "What is left of the sink?",
        "What is below the mirror?",
        "What is near the back wall?",
        "What is on the right wall?",
        "Where is the trash can?",
        "What is in front of the toilet?",
        "Where is the vending machine?",
        "What is right of the toilet?",
        "What is behind the toilet?",
        "Where is the paper towel dispenser?",
    ]

    ap = argparse.ArgumentParser(description="Spatial graph prune -> compact JSON -> optional LLM.")
    ap.add_argument(
        "--benchmark",
        action="store_true",
        help="Print EXPECTED_ANSWERS metrics (top-1, top-k, false positives).",
    )
    ap.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run benchmark and exit (no per-query demo).",
    )
    args = ap.parse_args()

    if args.benchmark or args.benchmark_only:
        run_benchmark(verbose=True)

    if args.benchmark_only:
        raise SystemExit(0)

    for q in test_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {q}")
        run_demo(q)
"""Vocabulary snapshot pulled from tiny_graph_demo at import time.

Single source of truth for validation in parsers.llm_parser. Scene is
frozen at v1; if the scene changes, reimport this module.
"""

from __future__ import annotations

from tiny_graph_demo import (
    KNOWN_COLORS,
    SCENE_LABEL_PHRASES,
    ZONE_PHRASES,
    scene,
)

# Intent set emitted by tiny_graph_demo.parse_query (7 values, not 6).
ALLOWED_INTENTS: frozenset[str] = frozenset({
    "find_object",
    "find_location",
    "describe",
    "describe_spatial",
    "describe_zone",
    "describe_near_zone",
    "describe_near_anchor",
})

# Spatial relations emitted by parse_query (6). NEAR is NOT here;
# it is modeled via describe_near_zone / describe_near_anchor intents.
ALLOWED_RELATIONS: frozenset[str] = frozenset({
    "LEFT_OF",
    "RIGHT_OF",
    "BELOW",
    "ABOVE",
    "IN_FRONT_OF",
    "BEHIND",
})

# Query-side zone families (6) — NOT node-side zones (11).
ALLOWED_ZONE_FAMILIES: frozenset[str] = frozenset(family for _, family in ZONE_PHRASES)

ALLOWED_LABEL_PHRASES: frozenset[str] = frozenset(SCENE_LABEL_PHRASES)

ALLOWED_COLORS: frozenset[str] = frozenset(KNOWN_COLORS)

ALLOWED_TARGET_TYPES: frozenset[str] = frozenset(
    str(n["attributes"]["type"]).strip()
    for n in scene
    if isinstance(n.get("attributes"), dict) and n["attributes"].get("type")
)

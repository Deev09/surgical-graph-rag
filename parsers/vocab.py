"""Vocabulary snapshot pulled from tiny_graph_demo at import time.

Single source of truth for validation in parsers.llm_parser. Default at
import is the v1 bathroom snapshot; cross-scene runners can replace the
label vocab via rebuild_label_vocab_from_scene(scene_record). Mutation
is in-place so callers that already imported ALLOWED_LABEL_PHRASES see
the update without re-importing.
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

ALLOWED_LABEL_PHRASES: set[str] = set(SCENE_LABEL_PHRASES)

ALLOWED_COLORS: frozenset[str] = frozenset(KNOWN_COLORS)

ALLOWED_TARGET_TYPES: frozenset[str] = frozenset(
    str(n["attributes"]["type"]).strip()
    for n in scene
    if isinstance(n.get("attributes"), dict) and n["attributes"].get("type")
)


def rebuild_label_vocab_from_scene(scene_record: dict) -> None:
    """Replace ALLOWED_LABEL_PHRASES with labels from scene_record['objects'].

    Mirrors tiny_graph_demo._object_labels_for_matching: each label is
    added in both space-form and underscore-form so the LLM and the
    validator can match either. Mutates the existing set in place.
    """
    new_labels: list[str] = []
    for obj in scene_record.get("objects", []):
        lab = str(obj["label"])
        new_labels.append(lab.replace("_", " "))
        if "_" in lab:
            new_labels.append(lab)
    ALLOWED_LABEL_PHRASES.clear()
    ALLOWED_LABEL_PHRASES.update(new_labels)

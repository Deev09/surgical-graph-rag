"""Candidate filter extracted from tiny_graph_demo.

New signature accepts a score_fn so v1 and future scorer-v2 can both plug in.
The body is otherwise byte-for-byte identical to tiny_graph_demo's prior
candidate_filter.
"""

from __future__ import annotations

from typing import Any, Callable


def candidate_filter(
    scene_graph: list[dict[str, Any]],
    parsed: Any,
    score_fn: Callable[..., float],
    *,
    min_score: float = 0.20,
) -> list[dict[str, Any]]:
    from tiny_graph_demo import _label_equiv, normalize_text  # lazy

    candidates: list[dict[str, Any]] = []

    for node in scene_graph:
        node_label_norm = normalize_text(str(node["label"]).replace("_", " "))

        if parsed.intent == "describe_near_anchor" and parsed.anchor_objects:
            if any(_label_equiv(node_label_norm, anchor) for anchor in parsed.anchor_objects):
                continue

        if parsed.intent == "describe_spatial" and parsed.relation_anchor:
            if _label_equiv(node_label_norm, parsed.relation_anchor):
                continue

        score = score_fn(node, parsed, scene_graph=scene_graph)
        if score >= min_score:
            node_copy = dict(node)
            node_copy["score"] = round(score, 3)
            candidates.append(node_copy)

    return candidates

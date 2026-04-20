"""Strict answer-type eval for the paraphrase A/B.

Additive to `node_matches_expectation` (which stays as-is — it backs v1's
published baseline numbers). This module provides:

    classify_expected(e)         -> "label" | "zone" | "both" | "unknown"
    INTENT_ANSWER_TYPE[intent]   -> "label" | "zone"
    match_strict(node, e, atyp)  -> bool, no label-zone OR, no substring cushion
    score_one_strict(parsed, expected, pruned) -> dict of strict fields

A row is a "real hit" iff strict_top1_ok AND intent_congruent. Lenient
metrics are still computed in the caller (eval_paraphrase.py) so the old
numbers remain side-by-side and auditable.
"""

from __future__ import annotations

from typing import Any

from tiny_graph_demo import (
    ParsedQuery,
    _label_equiv,
    normalize_text,
    scene,
)


INTENT_ANSWER_TYPE: dict[str, str] = {
    "find_location": "zone",
    "find_object": "label",
    "describe": "label",
    "describe_spatial": "label",
    "describe_zone": "label",
    "describe_near_zone": "label",
    "describe_near_anchor": "label",
}


_SCENE_ZONES: frozenset[str] = frozenset(
    normalize_text(str(n["zone"]).replace("_", " ")) for n in scene
)
_SCENE_LABELS_NORM: frozenset[str] = frozenset(
    normalize_text(str(n["label"]).replace("_", " ")) for n in scene
)


def classify_expected(e: str) -> str:
    en = normalize_text(e.replace("_", " "))
    is_zone = en in _SCENE_ZONES
    is_label = en in _SCENE_LABELS_NORM or any(
        _label_equiv(e, lab) for lab in _SCENE_LABELS_NORM
    )
    if is_zone and is_label:
        return "both"
    if is_zone:
        return "zone"
    if is_label:
        return "label"
    return "unknown"


def match_strict(node: dict[str, Any], e: str, answer_type: str) -> bool:
    """Answer-type-only match. No label<->zone OR, no 4-char substring cushion."""
    en = normalize_text(e.replace("_", " "))
    if answer_type == "zone":
        zn = normalize_text(str(node["zone"]).replace("_", " "))
        return zn == en
    if answer_type == "label":
        ln = normalize_text(str(node["label"]).replace("_", " "))
        return ln == en
    if answer_type == "both":
        zn = normalize_text(str(node["zone"]).replace("_", " "))
        ln = normalize_text(str(node["label"]).replace("_", " "))
        return zn == en or ln == en
    return False  # unknown


def resolve_answer_type(expected: list[str]) -> str:
    """Resolve a single answer_type for an expected bucket.

    If all entries agree, return that type. If mixed (e.g. a label + a zone in
    the same bucket), return "both" so strict matching still accepts either.
    """
    types = {classify_expected(e) for e in expected}
    if types == {"label"}:
        return "label"
    if types == {"zone"}:
        return "zone"
    if "unknown" in types and len(types) == 1:
        return "unknown"
    # Mixed or contains a "both" — fall back to "both" (still stricter than
    # lenient because no substring cushion).
    return "both"


def score_one_strict(
    parsed: ParsedQuery | None,
    expected: list[str],
    pruned: list[dict[str, Any]],
) -> dict[str, Any]:
    answer_type = resolve_answer_type(expected)

    if parsed is None:
        return {
            "answer_type": answer_type,
            "intent_congruent": False,
            "strict_top1_ok": False,
            "strict_topk_ok": False,
            "hit_category": "miss",
        }

    intent_type = INTENT_ANSWER_TYPE.get(parsed.intent, "label")
    intent_congruent = (intent_type == answer_type) or (answer_type == "both")

    def strict_match_any(node: dict[str, Any]) -> bool:
        return any(match_strict(node, e, answer_type) for e in expected)

    strict_top1_ok = bool(pruned) and strict_match_any(pruned[0])
    strict_topk_ok = any(strict_match_any(p) for p in pruned)

    if strict_top1_ok and intent_congruent:
        hit_category = "strict_hit"
    elif strict_top1_ok and not intent_congruent:
        hit_category = "intent_cushion"
    else:
        # Caller still has the lenient top1_ok; they decide metric_cushion vs miss.
        hit_category = "pending_lenient"

    return {
        "answer_type": answer_type,
        "intent_congruent": intent_congruent,
        "strict_top1_ok": strict_top1_ok,
        "strict_topk_ok": strict_topk_ok,
        "hit_category": hit_category,
    }

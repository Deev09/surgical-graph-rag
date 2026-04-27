"""Top-k pruning + per-intent k selection, extracted from tiny_graph_demo.

Unchanged behavior.
"""

from __future__ import annotations

from typing import Any


def top_k_prune(candidates: list[dict[str, Any]], k: int = 3) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:k]


def top_k_for_intent(parsed: Any) -> int:
    if parsed.intent in ("describe_near_anchor", "describe_near_zone", "describe_zone"):
        return 10
    if parsed.intent == "describe_spatial":
        return 6
    return 4

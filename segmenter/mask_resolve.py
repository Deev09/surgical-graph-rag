"""Deterministic overlapping-mask resolution (the Colab-side adapter).

Mask-proposal backends (Mask3D / OpenMask3D class-agnostic stage) emit K
overlapping binary masks with scores over N points. The C1 contract needs ONE
dense assignment: vertex_instance_ids[N] with -1 for unclaimed. This module
is the frozen, tested resolution rule — the Colab notebook imports THIS file
(numpy-only) rather than re-implementing it, so determinism is guaranteed by
local tests, not by notebook discipline.

Rule (frozen; see MaskResolveConfig):
  1. drop masks with score < min_score,
  2. per vertex, the highest-scoring surviving mask that claims it wins
     (ties -> lowest mask index, so resolution is order-stable),
  3. drop resulting instances with < min_vertices vertices (their vertices
     become unclaimed rather than being reassigned — reassignment would leak
     a second-choice heuristic into the contract),
  4. everything unclaimed = -1,
  5. NO semantic-class filtering of any kind (C1 is class-agnostic).

Winning-mask instance ids are the surviving masks' ORIGINAL indices, so an
instance id always names a row of the backend's mask tensor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaskResolveConfig:
    min_score: float = 0.0        # keep everything by default; freeze per run
    min_vertices: int = 20

    def params(self) -> dict:
        return {"min_score": self.min_score, "min_vertices": self.min_vertices}


def resolve_masks(
    masks: np.ndarray,      # bool/0-1 [K, N] — mask k claims vertex n
    scores: np.ndarray,     # float [K]
    config: MaskResolveConfig = MaskResolveConfig(),
) -> np.ndarray:
    """Return vertex_instance_ids int64 [N]; -1 = unclaimed."""
    masks = np.asarray(masks)
    scores = np.asarray(scores, dtype=np.float64)
    if masks.ndim != 2 or scores.shape != (masks.shape[0],):
        raise ValueError(f"shape mismatch: masks {masks.shape}, scores {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("non-finite mask scores")
    k, n = masks.shape
    out = np.full(n, -1, dtype=np.int64)
    if k == 0:
        return out

    keep = scores >= config.min_score
    claimed = masks.astype(bool) & keep[:, None]

    # highest score wins; ties -> lowest original mask index. Iterating in
    # (score desc, index asc) order and only filling still-empty vertices
    # implements exactly that.
    order = np.lexsort((np.arange(k), -scores))
    for idx in order:
        if not keep[idx]:
            continue
        sel = claimed[idx] & (out == -1)
        out[sel] = idx

    # min-vertex filter: too-small instances become unclaimed, not reassigned
    ids, counts = np.unique(out[out >= 0], return_counts=True)
    for i, c in zip(ids, counts):
        if c < config.min_vertices:
            out[out == i] = -1
    return out

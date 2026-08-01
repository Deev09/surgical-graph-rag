"""C1-M2C rule v1 — oracle-free merge-decomposition selection repair.

Protocol: docs/c1_m2c_protocol.md (PREDECLARED, frozen at 9953f04).

Mechanism targeted (measured in docs/c1_composition_ceiling.md): the
backend's high-scoring masks are often MERGES of several objects, and the
near-perfect per-object masks score below the frozen min_score, so
winner-takes-all resolution discards them. Signals used — all oracle-free,
per the protocol's allowlist: mask scores, mask sizes, and the pairwise
overlap/containment structure of the raw masks. Nothing else.

Rule v1 ("merge decomposition"):

  base set   = masks with score >= base_min_score (the frozen operating
               point, 0.2 — an inherited constant, not a rule parameter)
  child of p = any mask c (ANY score) with
                 |p ∩ c| / |c| >= child_containment   (c lies inside p)
                 |c| <= child_max_size_frac * |p|      (c is much smaller)
  kept children of p = greedy scan of p's children by size (desc; ties by
               index) keeping those whose overlap ratio with every
               already-kept child is <= child_disjoint_max
               (ratio = |a ∩ b| / min(|a|,|b|))
  p is a MERGE SUSPECT if it has >= 2 kept children and the sum of their
               intersections with p covers >= parent_cover_min * |p|
               (children are near-disjoint, so the sum approximates the
               union; exactness is not required for a >=-threshold)
  output     = suppress suspects entirely; admit their kept children with
               priority max(child score, parent score) ("the children
               replace their parent"); every other base mask keeps its raw
               score; every other non-base mask stays excluded.

The output feeds the UNCHANGED frozen resolver (mask_resolve.resolve_masks
with min_score=0.0 over the returned priorities; excluded masks carry
priority EXCLUDED < 0). Deterministic: no randomness, no per-scene
constants, single pass (admitted children are not re-decomposed — recorded
as a v1 limitation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EXCLUDED = -1.0
FROZEN_BASE_MIN_SCORE = 0.2


@dataclass(frozen=True)
class SelectionRepairConfig:
    # the four free scalar parameters allowed by the protocol
    child_containment: float = 0.8
    child_max_size_frac: float = 0.5
    child_disjoint_max: float = 0.2
    parent_cover_min: float = 0.5

    def params(self) -> dict:
        return {
            "rule": "merge_decomposition_v1",
            "child_containment": self.child_containment,
            "child_max_size_frac": self.child_max_size_frac,
            "child_disjoint_max": self.child_disjoint_max,
            "parent_cover_min": self.parent_cover_min,
            "base_min_score": FROZEN_BASE_MIN_SCORE,
        }


def _pairwise_intersections(masks: np.ndarray) -> np.ndarray:
    """int64 [K, K] intersection counts (row-by-row; K^2*avg_size bool ops)."""
    k = masks.shape[0]
    inter = np.zeros((k, k), dtype=np.int64)
    for i in range(k):
        idx = np.nonzero(masks[i])[0]
        if len(idx):
            inter[i] = masks[:, idx].sum(axis=1)
    return inter


def repair_selection(
    masks: np.ndarray,          # bool [K, N]
    scores: np.ndarray,         # float [K]
    config: SelectionRepairConfig = SelectionRepairConfig(),
) -> tuple[np.ndarray, dict]:
    """Returns (priorities float64 [K], diagnostics). priorities < 0 means
    excluded; feed to resolve_masks with min_score=0.0."""
    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    k = masks.shape[0]
    sizes = masks.sum(axis=1).astype(np.int64)
    inter = _pairwise_intersections(masks)

    base = (scores >= FROZEN_BASE_MIN_SCORE) & (sizes > 0)
    priorities = np.where(base, scores, EXCLUDED)

    suspects: list[dict] = []
    admitted: dict[int, float] = {}    # child row -> admitted priority
    for p in np.nonzero(base)[0]:
        cand = [c for c in range(k) if c != p and sizes[c] > 0
                and inter[p, c] / sizes[c] >= config.child_containment
                and sizes[c] <= config.child_max_size_frac * sizes[p]]
        # greedy mutually-disjoint subset, largest children first
        cand.sort(key=lambda c: (-sizes[c], c))
        kept: list[int] = []
        for c in cand:
            if all(inter[c, o] / min(sizes[c], sizes[o])
                   <= config.child_disjoint_max for o in kept):
                kept.append(c)
        cover = int(sum(inter[p, c] for c in kept))
        if len(kept) >= 2 and cover >= config.parent_cover_min * sizes[p]:
            priorities[p] = EXCLUDED
            for c in kept:
                pr = max(scores[c], scores[p])
                admitted[c] = max(admitted.get(c, EXCLUDED), pr)
            suspects.append({
                "mask": int(p), "score": round(float(scores[p]), 4),
                "size": int(sizes[p]), "n_children": len(kept),
                "children": [int(c) for c in kept],
                "cover_frac": round(cover / sizes[p], 4),
            })
    for c, pr in admitted.items():
        priorities[c] = max(priorities[c], pr)

    diagnostics = {
        "config": config.params(),
        "n_masks": int(k),
        "n_base": int(base.sum()),
        "n_suspects_suppressed": len(suspects),
        "n_children_admitted": len(admitted),
        "n_admitted_below_base_score": sum(
            1 for c in admitted if scores[c] < FROZEN_BASE_MIN_SCORE),
        "n_final_active": int((priorities >= 0).sum()),
        "suspects": suspects,
        "admitted": {str(c): round(float(pr), 4)
                     for c, pr in sorted(admitted.items())},
    }
    return priorities, diagnostics


# --------------------------------------------------------------------------
# Rule v2 — "corroborated carve-out + retained-fraction suppression".
#
# v1 failure diagnosis (runs/phase8_c1/selection_repair/, report v1): the
# dominant loss is a SMALL real object swallowed by a much bigger mask that
# is otherwise a near-duplicate of a big object (plate inside sofa-mask:
# component sizes 0.08 + 0.96 of the parent, so no >=2-children-<=0.5
# decomposition exists), and v1's admitted children still lost their
# vertices to third-party overlappers. v2 uses two mechanisms:
#
#   carve-out  a mask c (ANY score) that is CORROBORATED (>= 1 other mask
#              with mutual IoU >= corroborator_iou — real objects are
#              proposed repeatedly; measured on the dev scene: 25/30
#              oracle-best masks corroborated) and lies inside a base mask
#              p (containment >= carve_containment, |c| <= carve_max_frac
#              * |p|) is promoted to priority just above its best such
#              parent. The parent keeps everything outside the carve-out.
#   retention  after a first frozen-resolver pass at the declared
#              priorities, any mask retaining < retain_min of its vertices
#              is added to the suppression set (its surviving sliver would
#              be an arbitrary-box fragment), and the frozen resolver runs
#              once more without the suppressed masks. Exactly two passes.
#              This uses the protocol's explicitly-allowed signal "retained
#              fraction under winner-takes-all at a declared priority", and
#              it SUBSUMES v1: both components of a two-object merge get
#              carved out, the merge parent's retention collapses, and the
#              second pass removes it.
#
# Free scalar parameters (4): carve_containment, carve_max_frac,
# corroborator_iou, retain_min. Structural constants (not tuned):
# corroborators_min = 1, promotion epsilon = 1e-6, exactly 2 passes.
# --------------------------------------------------------------------------

_PROMOTE_EPS = 1e-6


@dataclass(frozen=True)
class SelectionRepairV2Config:
    carve_containment: float = 0.8
    carve_max_frac: float = 0.6
    corroborator_iou: float = 0.7
    retain_min: float = 0.5

    def params(self) -> dict:
        return {
            "rule": "corroborated_carveout_v2",
            "carve_containment": self.carve_containment,
            "carve_max_frac": self.carve_max_frac,
            "corroborator_iou": self.corroborator_iou,
            "retain_min": self.retain_min,
            "base_min_score": FROZEN_BASE_MIN_SCORE,
            "structural": {"corroborators_min": 1, "promote_eps": _PROMOTE_EPS,
                           "resolver_passes": 2},
        }


def repair_selection_v2(
    masks: np.ndarray,
    scores: np.ndarray,
    config: SelectionRepairV2Config = SelectionRepairV2Config(),
    *,
    min_vertices: int = 20,
) -> tuple[np.ndarray, dict]:
    """Returns (vertex_instance_ids int64 [N], diagnostics).

    Unlike v1 this returns the RESOLVED dense assignment (the rule includes
    the retention pass, which needs a first resolution); both passes use
    the unchanged frozen resolver.
    """
    from segmenter.mask_resolve import MaskResolveConfig, resolve_masks

    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    k = masks.shape[0]
    sizes = masks.sum(axis=1).astype(np.int64)
    inter = _pairwise_intersections(masks)

    corroborated = np.zeros(k, dtype=bool)
    for i in range(k):
        if sizes[i] == 0:
            continue
        union = sizes[i] + sizes - inter[i]
        iou = np.where(union > 0, inter[i] / np.maximum(union, 1), 0.0)
        iou[i] = 0.0
        corroborated[i] = bool((iou >= config.corroborator_iou).any())

    base = (scores >= FROZEN_BASE_MIN_SCORE) & (sizes > 0)
    priorities = np.where(base, scores, EXCLUDED)

    promotions: list[dict] = []
    for c in range(k):
        if sizes[c] < min_vertices or not corroborated[c]:
            continue
        best_parent_score = None
        for p in np.nonzero(base)[0]:
            if p == c:
                continue
            if (inter[p, c] / sizes[c] >= config.carve_containment
                    and sizes[c] <= config.carve_max_frac * sizes[p]):
                s = float(scores[p])
                if best_parent_score is None or s > best_parent_score:
                    best_parent_score = s
        if best_parent_score is not None:
            new_pr = best_parent_score + _PROMOTE_EPS
            if new_pr > priorities[c]:
                priorities[c] = new_pr
                promotions.append({
                    "mask": int(c), "score": round(float(scores[c]), 4),
                    "size": int(sizes[c]),
                    "promoted_to": round(new_pr, 6),
                })

    cfg = MaskResolveConfig(min_score=0.0, min_vertices=min_vertices)
    ids1 = resolve_masks(masks, priorities, cfg)

    suppressed: list[dict] = []
    pr2 = priorities.copy()
    for m in range(k):
        if pr2[m] < 0 or sizes[m] == 0:
            continue
        retained = int((ids1 == m).sum()) / int(sizes[m])
        if retained < config.retain_min:
            pr2[m] = EXCLUDED
            suppressed.append({"mask": int(m),
                               "score": round(float(scores[m]), 4),
                               "size": int(sizes[m]),
                               "retained_frac": round(retained, 4)})
    ids2 = resolve_masks(masks, pr2, cfg)

    diagnostics = {
        "config": config.params(),
        "n_masks": int(k),
        "n_base": int(base.sum()),
        "n_corroborated": int(corroborated.sum()),
        "n_promoted": len(promotions),
        "n_promoted_below_base_score": sum(
            1 for p in promotions if p["score"] < FROZEN_BASE_MIN_SCORE),
        "n_suppressed_low_retention": len(suppressed),
        "n_final_instances": int(len(set(ids2[ids2 >= 0]))),
        "promotions": promotions,
        "suppressed_low_retention": suppressed,
    }
    return ids2, diagnostics


# --------------------------------------------------------------------------
# Rule v3 — decomposition + restricted carve-out + retention (final budget
# version; protocol allows three).
#
# v2 failure diagnosis (corrected reports, runs/phase8_c1/selection_repair/):
# entity recall reached the gate (24/53) but precision collapsed to 0.35 —
# carve-out at <= 0.6 of the parent also carved PARTS out of real objects
# (a corroborated tabletop-sized sub-mask shatters its own table, which
# then dies of low retention: "on the table" went EMPTY), and requiring
# only one corroborator admitted junk fragments (144 promotions).
#
# v3 composes three mechanisms:
#   1. v1's merge DECOMPOSITION (>= 2 mutually-disjoint children covering
#      the parent), with v1's numbers frozen as structural constants
#      (containment 0.8, child <= 0.5 parent, disjoint <= 0.2, cover >=
#      0.5) — it correctly identified true merges and only true merges on
#      the dev scene. Children are promoted just above the parent.
#   2. small-object CARVE-OUT, restricted: |c| <= carve_max_frac * |p|
#      (0.25 — a real swallowed object is small relative to its swallower;
#      a PART of an object is not) and >= corroborators_min (2)
#      independent duplicate proposals.
#   3. retained-fraction suppression at retain_min (two frozen-resolver
#      passes, as v2) — kills decomposed merge parents and any mask whose
#      surviving sliver would be an arbitrary-box fragment.
#
# Free scalar parameters (4): carve_max_frac, corroborator_iou,
# corroborators_min, retain_min. Everything else is structural.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionRepairV3Config:
    carve_max_frac: float = 0.25
    corroborator_iou: float = 0.7
    corroborators_min: int = 2
    retain_min: float = 0.5

    def params(self) -> dict:
        return {
            "rule": "decompose_carveout_retention_v3",
            "carve_max_frac": self.carve_max_frac,
            "corroborator_iou": self.corroborator_iou,
            "corroborators_min": self.corroborators_min,
            "retain_min": self.retain_min,
            "base_min_score": FROZEN_BASE_MIN_SCORE,
            "structural": {
                "decomposition": {"child_containment": 0.8,
                                  "child_max_size_frac": 0.5,
                                  "child_disjoint_max": 0.2,
                                  "parent_cover_min": 0.5},
                "carve_containment": 0.8,
                "promote_eps": _PROMOTE_EPS,
                "resolver_passes": 2,
            },
        }


def repair_selection_v3(
    masks: np.ndarray,
    scores: np.ndarray,
    config: SelectionRepairV3Config = SelectionRepairV3Config(),
    *,
    min_vertices: int = 20,
) -> tuple[np.ndarray, dict]:
    """Returns (vertex_instance_ids int64 [N], diagnostics)."""
    from segmenter.mask_resolve import MaskResolveConfig, resolve_masks

    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    k = masks.shape[0]
    sizes = masks.sum(axis=1).astype(np.int64)
    inter = _pairwise_intersections(masks)

    n_corr = np.zeros(k, dtype=np.int64)
    for i in range(k):
        if sizes[i] == 0:
            continue
        union = sizes[i] + sizes - inter[i]
        iou = np.where(union > 0, inter[i] / np.maximum(union, 1), 0.0)
        iou[i] = 0.0
        n_corr[i] = int((iou >= config.corroborator_iou).sum())

    base = (scores >= FROZEN_BASE_MIN_SCORE) & (sizes > 0)
    priorities = np.where(base, scores, EXCLUDED)
    promotions: list[dict] = []

    def promote(c: int, parent: int, why: str):
        new_pr = float(scores[parent]) + _PROMOTE_EPS
        if new_pr > priorities[c]:
            priorities[c] = new_pr
            promotions.append({
                "mask": int(c), "score": round(float(scores[c]), 4),
                "size": int(sizes[c]), "parent": int(parent),
                "mechanism": why, "promoted_to": round(new_pr, 6),
            })

    # mechanism 1: v1 merge decomposition (structural constants)
    for p in np.nonzero(base)[0]:
        cand = [c for c in range(k) if c != p and sizes[c] >= min_vertices
                and inter[p, c] / sizes[c] >= 0.8
                and sizes[c] <= 0.5 * sizes[p]]
        cand.sort(key=lambda c: (-sizes[c], c))
        kept: list[int] = []
        for c in cand:
            if all(inter[c, o] / min(sizes[c], sizes[o]) <= 0.2 for o in kept):
                kept.append(c)
        if len(kept) >= 2 and sum(inter[p, c] for c in kept) >= 0.5 * sizes[p]:
            for c in kept:
                promote(c, p, "decomposition")

    # mechanism 2: restricted small-object carve-out
    for c in range(k):
        if sizes[c] < min_vertices or n_corr[c] < config.corroborators_min:
            continue
        for p in np.nonzero(base)[0]:
            if (p != c and inter[p, c] / sizes[c] >= 0.8
                    and sizes[c] <= config.carve_max_frac * sizes[p]):
                promote(c, p, "carveout")

    cfg = MaskResolveConfig(min_score=0.0, min_vertices=min_vertices)
    ids1 = resolve_masks(masks, priorities, cfg)

    # mechanism 3: retained-fraction suppression, one re-resolve
    suppressed: list[dict] = []
    pr2 = priorities.copy()
    for m in range(k):
        if pr2[m] < 0 or sizes[m] == 0:
            continue
        retained = int((ids1 == m).sum()) / int(sizes[m])
        if retained < config.retain_min:
            pr2[m] = EXCLUDED
            suppressed.append({"mask": int(m),
                               "score": round(float(scores[m]), 4),
                               "size": int(sizes[m]),
                               "retained_frac": round(retained, 4)})
    ids2 = resolve_masks(masks, pr2, cfg)

    by_mech: dict[str, int] = {}
    for p in promotions:
        by_mech[p["mechanism"]] = by_mech.get(p["mechanism"], 0) + 1
    diagnostics = {
        "config": config.params(),
        "n_masks": int(k),
        "n_base": int(base.sum()),
        "n_promoted": len(promotions),
        "n_promoted_by_mechanism": by_mech,
        "n_promoted_below_base_score": sum(
            1 for p in promotions if p["score"] < FROZEN_BASE_MIN_SCORE),
        "n_suppressed_low_retention": len(suppressed),
        "n_final_instances": int(len(set(ids2[ids2 >= 0]))),
        "promotions": promotions,
        "suppressed_low_retention": suppressed,
    }
    return ids2, diagnostics

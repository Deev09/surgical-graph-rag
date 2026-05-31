"""BundleCorrespondence builders — phase0_design.md §3.

A BundleCorrespondence maps entities (and surfaces) across two
EntityArtifacts bundles. Required for any cross-bundle benchmark
comparison: object_uid is immutable within a bundle but not across.

Three methods, all returning a `BundleCorrespondence`:

  - shared_source_ref: entities matched by EntityIdentity.source_instance_ref
    equality. Surfaces matched by surface_uid equality. Trivial when both
    bundles came from the same upstream that preserves source refs.

  - iou_match: per §10 of phase0_design.md adoption gates. Two entities
    match when EITHER their AABB IoU ≥ iou_threshold OR the candidate
    centroid lies within min(centroid_scale_factor * oracle_bbox_diag,
    centroid_absolute_cap) of the oracle centroid. Greedy assignment by
    descending IoU; centroid fallback only after no IoU match remains.
    Surface matching is not attempted in Phase 1 (no shared primitive
    for surface IoU yet); surface_pairs is empty.

  - manual: load (entity_pair, surface_pair) lists from a hand-authored
    JSON file. Validates that every uid appears in its respective bundle.

Produced by tools in eval/ only — never by extractors. Stages do not
score themselves; this is the only place that compares bundles.
"""
from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Any

from extractors.base import EntityArtifact, EntityArtifacts, StructuralSurface


# ----- AABB / centroid helpers -----

def _aabb_volume(b: tuple[tuple[float, float, float], tuple[float, float, float]]) -> float:
    lo, hi = b
    dx = max(0.0, hi[0] - lo[0])
    dy = max(0.0, hi[1] - lo[1])
    dz = max(0.0, hi[2] - lo[2])
    return dx * dy * dz


def _aabb_intersection_volume(
    a: tuple[tuple[float, float, float], tuple[float, float, float]],
    b: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> float:
    a_lo, a_hi = a
    b_lo, b_hi = b
    dx = max(0.0, min(a_hi[0], b_hi[0]) - max(a_lo[0], b_lo[0]))
    dy = max(0.0, min(a_hi[1], b_hi[1]) - max(a_lo[1], b_lo[1]))
    dz = max(0.0, min(a_hi[2], b_hi[2]) - max(a_lo[2], b_lo[2]))
    return dx * dy * dz


def aabb_iou(
    a: tuple[tuple[float, float, float], tuple[float, float, float]],
    b: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> float:
    """3D AABB intersection-over-union. Returns 0 when bboxes are
    disjoint or either has zero volume."""
    inter = _aabb_intersection_volume(a, b)
    if inter == 0.0:
        return 0.0
    union = _aabb_volume(a) + _aabb_volume(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def aabb_diag(b: tuple[tuple[float, float, float], tuple[float, float, float]]) -> float:
    lo, hi = b
    return sqrt((hi[0] - lo[0]) ** 2 + (hi[1] - lo[1]) ** 2 + (hi[2] - lo[2]) ** 2)


def centroid_distance(
    a: tuple[float, float, float], b: tuple[float, float, float],
) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


# Local lightweight type to avoid importing BundleCorrespondence at module top.
# (We do import it inside functions to keep this module's import surface tight.)
def _make_correspondence(
    *,
    source: EntityArtifacts,
    target: EntityArtifacts,
    entity_pairs: list[tuple[str, str]],
    surface_pairs: list[tuple[str, str]],
    method: str,
    score: dict[str, float],
):
    from extractors.base import BundleCorrespondence
    matched_src_ent = {p[0] for p in entity_pairs}
    matched_tgt_ent = {p[1] for p in entity_pairs}
    matched_src_surf = {p[0] for p in surface_pairs}
    matched_tgt_surf = {p[1] for p in surface_pairs}
    return BundleCorrespondence(
        source_bundle_hash=source.bundle_hash,
        target_bundle_hash=target.bundle_hash,
        entity_pairs=entity_pairs,
        surface_pairs=surface_pairs,
        method=method,
        score=score,
        unmatched_source_entities=sorted(
            e.identity.object_uid for e in source.entities
            if e.identity.object_uid not in matched_src_ent
        ),
        unmatched_target_entities=sorted(
            e.identity.object_uid for e in target.entities
            if e.identity.object_uid not in matched_tgt_ent
        ),
        unmatched_source_surfaces=sorted(
            s.surface_uid for s in source.structural_surfaces
            if s.surface_uid not in matched_src_surf
        ),
        unmatched_target_surfaces=sorted(
            s.surface_uid for s in target.structural_surfaces
            if s.surface_uid not in matched_tgt_surf
        ),
    )


def _score_key(kind: str, src_uid: str, tgt_uid: str) -> str:
    return f"{kind}:{src_uid}->{tgt_uid}"


# ----- shared_source_ref -----

def correspond_shared_source_ref(
    source: EntityArtifacts,
    target: EntityArtifacts,
):
    """Match entities by EntityIdentity.source_instance_ref equality and
    surfaces by surface_uid equality.

    Empty source_instance_ref values are NOT used as match keys (would
    collide).
    """
    src_by_ref: dict[str, EntityArtifact] = {}
    for e in source.entities:
        ref = e.identity.source_instance_ref
        if not ref:
            continue
        if ref in src_by_ref:
            raise ValueError(
                f"source bundle has duplicate source_instance_ref {ref!r}; "
                "shared_source_ref requires unique refs within a bundle"
            )
        src_by_ref[ref] = e

    entity_pairs: list[tuple[str, str]] = []
    score: dict[str, float] = {}
    for t in target.entities:
        ref = t.identity.source_instance_ref
        if not ref:
            continue
        s = src_by_ref.get(ref)
        if s is None:
            continue
        entity_pairs.append((s.identity.object_uid, t.identity.object_uid))
        score[_score_key("entity", s.identity.object_uid, t.identity.object_uid)] = 1.0

    src_surfaces = {s.surface_uid: s for s in source.structural_surfaces}
    surface_pairs: list[tuple[str, str]] = []
    for ts in target.structural_surfaces:
        if ts.surface_uid in src_surfaces:
            surface_pairs.append((ts.surface_uid, ts.surface_uid))
            score[_score_key("surface", ts.surface_uid, ts.surface_uid)] = 1.0

    return _make_correspondence(
        source=source, target=target,
        entity_pairs=entity_pairs, surface_pairs=surface_pairs,
        method="shared_source_ref", score=score,
    )


# ----- iou_match -----

def correspond_iou_match(
    source: EntityArtifacts,
    target: EntityArtifacts,
    *,
    iou_threshold: float = 0.3,
    centroid_scale_factor: float = 0.5,
    centroid_absolute_cap: float = 0.30,
):
    """Match entities by AABB IoU with a scale-aware centroid-distance
    fallback. `source` is treated as the oracle whose bbox diag sets the
    centroid cap.

    Greedy assignment:
      1. For every (source, target) pair, compute IoU and centroid
         distance.
      2. Pass 1: rank all candidate pairs with IoU ≥ iou_threshold by
         descending IoU; greedily assign, marking entities matched.
      3. Pass 2: for any still-unmatched source entity, find the closest
         unmatched target whose centroid lies within
         min(centroid_scale_factor × source_bbox_diag,
             centroid_absolute_cap). If found, match.

    Surfaces are not matched in Phase 1 (no shared geometric primitive
    for surface IoU yet); surface_pairs is empty. structural_surfaces
    from both bundles are reported under unmatched lists.
    """
    pair_scores: list[tuple[float, float, str, str]] = []
    src_by_uid = {e.identity.object_uid: e for e in source.entities}
    tgt_by_uid = {e.identity.object_uid: e for e in target.entities}

    for s in source.entities:
        for t in target.entities:
            iou = aabb_iou(s.bbox_aabb, t.bbox_aabb)
            d = centroid_distance(s.centroid, t.centroid)
            pair_scores.append((iou, d, s.identity.object_uid, t.identity.object_uid))

    matched_src: set[str] = set()
    matched_tgt: set[str] = set()
    entity_pairs: list[tuple[str, str]] = []
    score: dict[str, float] = {}

    # Pass 1: IoU above threshold, greedy by descending IoU.
    iou_candidates = sorted(
        (p for p in pair_scores if p[0] >= iou_threshold),
        key=lambda p: -p[0],
    )
    for iou, _d, src_uid, tgt_uid in iou_candidates:
        if src_uid in matched_src or tgt_uid in matched_tgt:
            continue
        entity_pairs.append((src_uid, tgt_uid))
        score[_score_key("entity", src_uid, tgt_uid)] = float(iou)
        matched_src.add(src_uid)
        matched_tgt.add(tgt_uid)

    # Pass 2: centroid fallback for still-unmatched source entities.
    for s in source.entities:
        if s.identity.object_uid in matched_src:
            continue
        cap = min(
            centroid_scale_factor * aabb_diag(s.bbox_aabb),
            centroid_absolute_cap,
        )
        best: tuple[float, str] | None = None
        for t in target.entities:
            if t.identity.object_uid in matched_tgt:
                continue
            d = centroid_distance(s.centroid, t.centroid)
            if d <= cap:
                if best is None or d < best[0]:
                    best = (d, t.identity.object_uid)
        if best is None:
            continue
        d, tgt_uid = best
        entity_pairs.append((s.identity.object_uid, tgt_uid))
        # Score = 1 - d/cap so closer matches score higher; bounded [0, 1].
        score[_score_key("entity", s.identity.object_uid, tgt_uid)] = (
            max(0.0, 1.0 - (d / cap)) if cap > 0 else 0.0
        )
        matched_src.add(s.identity.object_uid)
        matched_tgt.add(tgt_uid)

    return _make_correspondence(
        source=source, target=target,
        entity_pairs=entity_pairs, surface_pairs=[],
        method="iou_match", score=score,
    )


# ----- manual -----

def correspond_manual(
    source: EntityArtifacts,
    target: EntityArtifacts,
    pairs_path: Path,
):
    """Load entity_pairs and surface_pairs from a JSON file:

      {
        "entity_pairs":  [["src_uid", "tgt_uid"], ...],
        "surface_pairs": [["src_surface_uid", "tgt_surface_uid"], ...],
        "notes": "..." (optional)
      }

    Validates that every uid exists in its respective bundle. Raises
    ValueError on the first unknown uid (no partial loads).
    """
    payload = json.loads(Path(pairs_path).read_text(encoding="utf-8"))

    src_entity_uids = {e.identity.object_uid for e in source.entities}
    tgt_entity_uids = {e.identity.object_uid for e in target.entities}
    src_surface_uids = {s.surface_uid for s in source.structural_surfaces}
    tgt_surface_uids = {s.surface_uid for s in target.structural_surfaces}

    entity_pairs: list[tuple[str, str]] = []
    for pair in payload.get("entity_pairs", []):
        src_uid, tgt_uid = str(pair[0]), str(pair[1])
        if src_uid not in src_entity_uids:
            raise ValueError(
                f"manual pairs reference unknown source entity {src_uid!r}; "
                f"source bundle has {len(src_entity_uids)} entities"
            )
        if tgt_uid not in tgt_entity_uids:
            raise ValueError(
                f"manual pairs reference unknown target entity {tgt_uid!r}; "
                f"target bundle has {len(tgt_entity_uids)} entities"
            )
        entity_pairs.append((src_uid, tgt_uid))

    surface_pairs: list[tuple[str, str]] = []
    for pair in payload.get("surface_pairs", []):
        src_uid, tgt_uid = str(pair[0]), str(pair[1])
        if src_uid not in src_surface_uids:
            raise ValueError(
                f"manual pairs reference unknown source surface {src_uid!r}"
            )
        if tgt_uid not in tgt_surface_uids:
            raise ValueError(
                f"manual pairs reference unknown target surface {tgt_uid!r}"
            )
        surface_pairs.append((src_uid, tgt_uid))

    score: dict[str, float] = {
        _score_key("entity", s, t): 1.0 for s, t in entity_pairs
    }
    score.update({_score_key("surface", s, t): 1.0 for s, t in surface_pairs})

    return _make_correspondence(
        source=source, target=target,
        entity_pairs=entity_pairs, surface_pairs=surface_pairs,
        method="manual", score=score,
    )

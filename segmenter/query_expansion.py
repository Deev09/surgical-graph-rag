"""Geometry-only selection of absent raw proposals near support anchors.

The caller supplies already-resolved support-anchor surfaces and a bounded set
of score-provisional or composition-lost candidate entities. Selection uses the
same pure rest-contact predicate as ON_ENTITY_SURFACE extraction. It does not
inspect oracle labels, expected answers, or relation edges.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.types import Vec3
from extractors.base import EntityArtifact, EntityIdentity
from extractors.entity_surfaces import EntitySurface
from geometry.rest_contact import RestContactConfig, rest_contact


@dataclass(frozen=True)
class AnchorCheck:
    anchor_uid: str
    on_surface: bool
    failed_clauses: list[str]
    bottom_gap_m: float
    in_plane_gap_m: float


@dataclass(frozen=True)
class ActivationDecision:
    candidate_uid: str
    selected: bool
    matching_anchor_uids: list[str]
    checks: list[AnchorCheck]


def raw_proposal_entities(
    xyz: np.ndarray,
    masks: np.ndarray,
    *,
    proposal_ids: set[int] | frozenset[int],
) -> list[EntityArtifact]:
    """Build anonymous AABB entities from independent, pre-composition masks."""
    xyz = np.asarray(xyz, dtype=np.float64)
    masks = np.asarray(masks, dtype=bool)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape [N, 3], got {xyz.shape}")
    if masks.ndim != 2 or masks.shape[1] != len(xyz):
        raise ValueError(
            f"masks must have shape [K, {len(xyz)}], got {masks.shape}"
        )
    if not np.isfinite(xyz).all():
        raise ValueError("xyz contains non-finite coordinates")
    invalid = {i for i in proposal_ids if i < 0 or i >= len(masks)}
    if invalid:
        raise ValueError(f"proposal ids out of range: {sorted(invalid)}")

    entities: list[EntityArtifact] = []
    for proposal_id in sorted(proposal_ids):
        points = xyz[masks[proposal_id]]
        if not len(points):
            continue
        lo = tuple(float(v) for v in points.min(axis=0))
        hi = tuple(float(v) for v in points.max(axis=0))
        entities.append(EntityArtifact(
            identity=EntityIdentity(
                object_uid=f"obj_{proposal_id}",
                display_label=f"segment_{proposal_id}",
                source_instance_ref=f"raw_proposal:{proposal_id}",
            ),
            bbox_aabb=(lo, hi),
            bbox_obb=None,
            centroid=tuple((lo[i] + hi[i]) / 2.0 for i in range(3)),
            geometry_handle=f"raw_masks.npz#{proposal_id}",
            extraction_diagnostics={
                "n_raw_vertices": int(len(points)),
                "composition_state": "independent_raw_proposal",
            },
        ))
    return entities


def materialize_query_scoped_assignment(
    hard_assignment: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    *,
    selected_ids: set[int] | frozenset[int],
    protected_ids: set[int] | frozenset[int],
    min_vertices: int,
) -> np.ndarray:
    """Locally re-compose selected raw masks over the hard assignment.

    Selected proposals are applied by descending score. They may reclaim
    vertices from accepted non-anchor masks, which is the behavior needed to
    test composition-loss recovery. Protected anchor ids and earlier
    higher-score selected proposals cannot be overwritten. Small leftovers
    are dropped without reassignment, matching the frozen resolver contract.
    """
    hard_assignment = np.asarray(hard_assignment, dtype=np.int64)
    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if hard_assignment.ndim != 1:
        raise ValueError(
            f"hard_assignment must be one-dimensional, got {hard_assignment.shape}"
        )
    if masks.ndim != 2 or masks.shape[1] != len(hard_assignment):
        raise ValueError(
            f"masks must have shape [K, {len(hard_assignment)}], got {masks.shape}"
        )
    if scores.shape != (len(masks),) or not np.isfinite(scores).all():
        raise ValueError(f"invalid scores shape or values: {scores.shape}")
    if min_vertices < 1:
        raise ValueError(f"min_vertices must be at least 1, got {min_vertices}")
    invalid = {
        i for i in set(selected_ids) | set(protected_ids)
        if i < 0 or i >= len(masks)
    }
    if invalid:
        raise ValueError(f"proposal ids out of range: {sorted(invalid)}")

    out = hard_assignment.copy()
    locked = np.isin(out, np.fromiter(protected_ids, dtype=np.int64))
    order = sorted(selected_ids, key=lambda i: (-scores[i], i))
    for proposal_id in order:
        claim = masks[proposal_id] & ~locked
        out[claim] = proposal_id
        locked[claim] = True

    ids, counts = np.unique(out[out >= 0], return_counts=True)
    for instance_id, count in zip(ids, counts):
        if count < min_vertices:
            out[out == instance_id] = -1
    return out


def select_support_region_candidates(
    entities: list[EntityArtifact],
    *,
    candidate_uids: set[str] | frozenset[str],
    anchors: list[EntitySurface],
    gravity: Vec3,
    rest_config: RestContactConfig = RestContactConfig(),
) -> list[ActivationDecision]:
    """Select candidates satisfying rest-contact on at least one anchor.

    Every requested candidate uid must exist in ``entities``. Decisions are
    sorted by uid, and anchor checks preserve caller-provided anchor order.
    """
    by_uid = {entity.identity.object_uid: entity for entity in entities}
    missing = set(candidate_uids) - set(by_uid)
    if missing:
        raise ValueError(
            f"candidate uids missing from geometry bundle: {sorted(missing)}"
        )

    decisions: list[ActivationDecision] = []
    for uid in sorted(candidate_uids):
        entity = by_uid[uid]
        checks: list[AnchorCheck] = []
        matches: list[str] = []
        for anchor in anchors:
            result = rest_contact(
                entity.bbox_aabb,
                entity.centroid,
                anchor.plane,
                anchor.polygon,
                gravity,
                rest_config,
            )
            checks.append(AnchorCheck(
                anchor_uid=anchor.owner_entity_uid,
                on_surface=result.on_surface,
                failed_clauses=list(result.failed_clauses),
                bottom_gap_m=float(result.evidence["bottom_gap_m"]),
                in_plane_gap_m=float(result.evidence["in_plane_gap_m"]),
            ))
            if result.on_surface:
                matches.append(anchor.owner_entity_uid)
        decisions.append(ActivationDecision(
            candidate_uid=uid,
            selected=bool(matches),
            matching_anchor_uids=matches,
            checks=checks,
        ))
    return decisions

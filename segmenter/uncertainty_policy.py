"""Uncertainty-preserving policy over overlapping segmentation proposals.

This module is intentionally upstream of EntityArtifacts / SceneGraphBundle.
It turns one immutable raw proposal bundle into three comparable views:

* hard: only proposals at or above ``accepted_score``;
* inclusive: every proposal is composed and trusted equally;
* uncertainty: accepted proposals remain the default graph, while lower-score
  plausible proposals are retained as query-activatable candidates.

The backend score is preserved as evidence, not described as a calibrated
probability. ``association_confidence`` is a separate composition heuristic:
the fraction of a proposal's raw vertices that survive overlap resolution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from segmenter.mask_resolve import MaskResolveConfig, resolve_masks


ProposalState = Literal["accepted", "provisional", "discarded"]
ScoreTransform = Literal["identity_probability", "sigmoid_logit"]


@dataclass(frozen=True)
class UncertaintyPolicyConfig:
    """Operating policy; thresholds change graph materialization, not truth."""

    accepted_score: float = 0.2
    provisional_score: float = 0.05
    min_vertices: int = 20
    score_transform: ScoreTransform = "identity_probability"

    def validate(self) -> None:
        if not (
            math.isfinite(self.provisional_score)
            and math.isfinite(self.accepted_score)
            and self.provisional_score <= self.accepted_score
        ):
            raise ValueError(
                "expected finite provisional_score <= accepted_score; "
                f"got provisional_score={self.provisional_score}, "
                f"accepted_score={self.accepted_score}"
            )
        if self.score_transform == "identity_probability" and not (
            0.0 <= self.provisional_score <= self.accepted_score <= 1.0
        ):
            raise ValueError(
                "identity_probability thresholds must lie in [0, 1]; "
                f"got provisional_score={self.provisional_score}, "
                f"accepted_score={self.accepted_score}"
            )
        if self.score_transform not in ("identity_probability", "sigmoid_logit"):
            raise ValueError(f"unknown score_transform {self.score_transform!r}")
        if self.min_vertices < 1:
            raise ValueError(
                f"min_vertices must be at least 1, got {self.min_vertices}"
            )


@dataclass(frozen=True)
class EvidenceMass:
    """Additive two-outcome evidence for later multi-observation fusion.

    This is a small Beta/Dirichlet-style accumulator, not a calibration
    claim. A single backend score ``s`` contributes ``s`` positive and
    ``1-s`` negative mass. Repeated, associated observations can be added.
    """

    positive: float = 0.0
    negative: float = 0.0
    observations: int = 0

    def add(self, score: float, *, weight: float = 1.0) -> "EvidenceMass":
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {score}")
        if weight <= 0.0:
            raise ValueError(f"weight must be positive, got {weight}")
        return EvidenceMass(
            positive=self.positive + score * weight,
            negative=self.negative + (1.0 - score) * weight,
            observations=self.observations + 1,
        )

    @property
    def mean(self) -> float | None:
        total = self.positive + self.negative
        return self.positive / total if total > 0.0 else None


@dataclass(frozen=True)
class ProposalEvidence:
    proposal_id: int
    state: ProposalState
    backend_score: float
    existence_evidence: EvidenceMass
    raw_vertices: int
    hard_assigned_vertices: int
    assigned_vertices: int
    inclusive_assigned_vertices: int
    association_confidence: float

    @property
    def materialized(self) -> bool:
        return self.assigned_vertices > 0 and self.state != "discarded"

    @property
    def hard_materialized(self) -> bool:
        return self.hard_assigned_vertices > 0

    @property
    def activation_candidate(self) -> bool:
        return self.materialized and self.state == "provisional"

    @property
    def raw_activation_candidate(self) -> bool:
        return (
            self.raw_vertices > 0
            and self.state != "discarded"
            and not self.hard_materialized
        )

    @property
    def inclusive_materialized(self) -> bool:
        return self.inclusive_assigned_vertices > 0


@dataclass(frozen=True)
class PolicyView:
    """Assignments plus per-proposal evidence for one observation."""

    hard_assignment: np.ndarray
    inclusive_assignment: np.ndarray
    inclusive_score: float
    uncertainty_assignment: np.ndarray
    proposals: list[ProposalEvidence]

    def activated_assignment(
        self,
        provisional_ids: set[int] | frozenset[int],
    ) -> np.ndarray:
        """Return accepted geometry plus selected provisional candidates.

        Unknown, discarded, or non-materialized ids are rejected so a caller
        cannot silently promote evidence that the policy excluded.
        """
        allowed = {
            p.proposal_id for p in self.proposals
            if p.state == "accepted" and p.materialized
        }
        candidates = {
            p.proposal_id for p in self.proposals if p.activation_candidate
        }
        unknown = set(provisional_ids) - candidates
        if unknown:
            raise ValueError(
                "requested ids are not provisional activation candidates: "
                f"{sorted(unknown)}"
            )
        allowed.update(provisional_ids)
        out = self.uncertainty_assignment.copy()
        if len(out):
            keep = np.isin(out, np.fromiter(allowed, dtype=np.int64))
            out[~keep] = -1
        return out


def _validate_inputs(
    masks: np.ndarray,
    scores: np.ndarray,
    config: UncertaintyPolicyConfig,
) -> tuple[np.ndarray, np.ndarray]:
    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if masks.ndim != 2 or scores.shape != (masks.shape[0],):
        raise ValueError(f"shape mismatch: masks {masks.shape}, scores {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("non-finite proposal scores")
    if config.score_transform == "identity_probability" and (
        (scores < 0.0) | (scores > 1.0)
    ).any():
        raise ValueError("uncertainty policy requires proposal scores in [0, 1]")
    return masks, scores


def classify_score(score: float, config: UncertaintyPolicyConfig) -> ProposalState:
    config.validate()
    if not math.isfinite(score):
        raise ValueError(f"score must be finite, got {score}")
    if config.score_transform == "identity_probability" and not 0.0 <= score <= 1.0:
        raise ValueError(f"probability score must be in [0, 1], got {score}")
    if score >= config.accepted_score:
        return "accepted"
    if score >= config.provisional_score:
        return "provisional"
    return "discarded"


def _evidence_score(score: float, transform: ScoreTransform) -> float:
    if transform == "identity_probability":
        return score
    if transform == "sigmoid_logit":
        if score >= 0.0:
            return 1.0 / (1.0 + math.exp(-score))
        exp_score = math.exp(score)
        return exp_score / (1.0 + exp_score)
    raise ValueError(f"unknown score transform {transform!r}")


def build_policy_view(
    masks: np.ndarray,
    scores: np.ndarray,
    config: UncertaintyPolicyConfig = UncertaintyPolicyConfig(),
) -> PolicyView:
    """Compose hard, inclusive, and uncertainty-preserving policy views."""
    config.validate()
    masks, scores = _validate_inputs(masks, scores, config)

    hard = resolve_masks(
        masks,
        scores,
        MaskResolveConfig(
            min_score=config.accepted_score,
            min_vertices=config.min_vertices,
        ),
    )
    inclusive_score = float(scores.min()) if len(scores) else 0.0
    inclusive = resolve_masks(
        masks,
        scores,
        MaskResolveConfig(
            min_score=inclusive_score,
            min_vertices=config.min_vertices,
        ),
    )
    uncertainty = resolve_masks(
        masks,
        scores,
        MaskResolveConfig(
            min_score=config.provisional_score,
            min_vertices=config.min_vertices,
        ),
    )

    raw_counts = np.count_nonzero(masks, axis=1)
    hard_ids = hard[hard >= 0]
    uncertainty_ids = uncertainty[uncertainty >= 0]
    inclusive_ids = inclusive[inclusive >= 0]
    hard_counts = np.bincount(hard_ids, minlength=len(scores))
    uncertainty_counts = np.bincount(
        uncertainty_ids, minlength=len(scores)
    )
    inclusive_counts = np.bincount(
        inclusive_ids, minlength=len(scores)
    )

    proposals: list[ProposalEvidence] = []
    for proposal_id, score in enumerate(scores):
        raw_vertices = int(raw_counts[proposal_id])
        hard_assigned_vertices = int(hard_counts[proposal_id])
        assigned_vertices = int(uncertainty_counts[proposal_id])
        inclusive_assigned_vertices = int(inclusive_counts[proposal_id])
        association = (
            assigned_vertices / raw_vertices if raw_vertices > 0 else 0.0
        )
        score_float = float(score)
        proposals.append(ProposalEvidence(
            proposal_id=proposal_id,
            state=classify_score(score_float, config),
            backend_score=score_float,
            existence_evidence=EvidenceMass().add(
                _evidence_score(score_float, config.score_transform)
            ),
            raw_vertices=raw_vertices,
            hard_assigned_vertices=hard_assigned_vertices,
            assigned_vertices=assigned_vertices,
            inclusive_assigned_vertices=inclusive_assigned_vertices,
            association_confidence=association,
        ))

    return PolicyView(
        hard_assignment=hard,
        inclusive_assignment=inclusive,
        inclusive_score=inclusive_score,
        uncertainty_assignment=uncertainty,
        proposals=proposals,
    )


def assignment_summary(assignment: np.ndarray) -> dict[str, int | float]:
    """Compact, JSON-safe materialization summary."""
    assignment = np.asarray(assignment)
    if assignment.ndim != 1:
        raise ValueError(f"assignment must be one-dimensional, got {assignment.shape}")
    claimed = int(np.count_nonzero(assignment >= 0))
    ids = np.unique(assignment[assignment >= 0])
    total = int(len(assignment))
    return {
        "n_materialized_nodes": int(len(ids)),
        "n_claimed_vertices": claimed,
        "claimed_vertex_fraction": claimed / total if total else 0.0,
    }

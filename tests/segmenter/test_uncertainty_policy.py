"""Tests for uncertainty-preserving raw proposal materialization."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmenter.uncertainty_policy import (
    EvidenceMass,
    UncertaintyPolicyConfig,
    assignment_summary,
    build_policy_view,
)


CFG = UncertaintyPolicyConfig(
    accepted_score=0.2,
    provisional_score=0.05,
    min_vertices=2,
)


def _example():
    masks = np.zeros((4, 11), dtype=bool)
    masks[0, 0:4] = True        # accepted
    masks[1, 4:7] = True        # provisional
    masks[2, 7:9] = True        # discarded
    masks[3, [2, 3, 9, 10]] = True  # provisional; loses overlap to mask 0
    scores = np.array([0.9, 0.15, 0.04, 0.1])
    return masks, scores


def test_policy_views_preserve_default_and_activation_pool():
    masks, scores = _example()
    view = build_policy_view(masks, scores, CFG)
    if view.hard_assignment.tolist() != [0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1]:
        raise AssertionError(f"unexpected hard assignment: {view.hard_assignment}")
    if view.uncertainty_assignment.tolist() != \
            [0, 0, 0, 0, 1, 1, 1, -1, -1, 3, 3]:
        raise AssertionError(
            f"unexpected uncertainty assignment: {view.uncertainty_assignment}"
        )
    if view.inclusive_assignment.tolist() != \
            [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3]:
        raise AssertionError(
            f"unexpected inclusive assignment: {view.inclusive_assignment}"
        )
    states = [p.state for p in view.proposals]
    if states != ["accepted", "provisional", "discarded", "provisional"]:
        raise AssertionError(f"unexpected states: {states}")
    if view.proposals[2].materialized or not view.proposals[2].inclusive_materialized:
        raise AssertionError(
            "discarded evidence must stay out of the uncertainty view while "
            "remaining observable in the inclusive comparison"
        )


def test_activation_is_explicit_and_rejects_untrusted_ids():
    masks, scores = _example()
    view = build_policy_view(masks, scores, CFG)
    activated = view.activated_assignment({3})
    if activated.tolist() != [0, 0, 0, 0, -1, -1, -1, -1, -1, 3, 3]:
        raise AssertionError(f"unexpected activation result: {activated}")
    for invalid in ({2}, {99}):
        try:
            view.activated_assignment(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"activation must reject invalid ids: {invalid}")


def test_existence_and_association_uncertainty_stay_separate():
    masks, scores = _example()
    view = build_policy_view(masks, scores, CFG)
    overlapped = view.proposals[3]
    if overlapped.existence_evidence.mean != 0.1:
        raise AssertionError(
            f"existence evidence must preserve backend score: {overlapped}"
        )
    if overlapped.association_confidence != 0.5:
        raise AssertionError(
            f"association confidence must expose overlap loss: {overlapped}"
        )
    fused = EvidenceMass().add(0.8).add(0.6, weight=2.0)
    if fused.observations != 2 or abs(fused.mean - (2.0 / 3.0)) > 1e-12:
        raise AssertionError(f"unexpected accumulated evidence: {fused}")


def test_composition_lost_accepted_mask_is_raw_activation_candidate():
    masks = np.array([
        [1, 1, 1, 1],
        [0, 1, 1, 0],
    ])
    view = build_policy_view(
        masks,
        np.array([0.9, 0.8]),
        UncertaintyPolicyConfig(min_vertices=1),
    )
    swallowed = view.proposals[1]
    if swallowed.hard_materialized:
        raise AssertionError(f"nested mask should be composition-lost: {swallowed}")
    if not swallowed.raw_activation_candidate:
        raise AssertionError(
            f"accepted composition loss must remain query-activatable: {swallowed}"
        )


def test_logit_scores_use_native_thresholds_and_sigmoid_evidence():
    masks = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ])
    config = UncertaintyPolicyConfig(
        accepted_score=0.2,
        provisional_score=0.05,
        min_vertices=1,
        score_transform="sigmoid_logit",
    )
    view = build_policy_view(masks, np.array([0.3, -1.0]), config)
    if [p.state for p in view.proposals] != ["accepted", "discarded"]:
        raise AssertionError("state thresholds must remain in native logit space")
    expected = 1.0 / (1.0 + np.exp(-0.3))
    if abs(view.proposals[0].existence_evidence.mean - expected) > 1e-12:
        raise AssertionError("logit evidence must be explicitly sigmoid transformed")
    if set(view.inclusive_assignment.tolist()) != {0, 1}:
        raise AssertionError("inclusive policy must include negative logit scores")


def test_validation_and_empty_summary():
    masks, scores = _example()
    for config in (
        UncertaintyPolicyConfig(accepted_score=0.2, provisional_score=0.3),
        UncertaintyPolicyConfig(min_vertices=0),
    ):
        try:
            build_policy_view(masks, scores, config)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid config must raise: {config}")
    bad_scores = scores.copy()
    bad_scores[0] = 1.1
    try:
        build_policy_view(masks, bad_scores, CFG)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range uncertainty scores must raise")
    if assignment_summary(np.array([], dtype=np.int64)) != {
        "n_materialized_nodes": 0,
        "n_claimed_vertices": 0,
        "claimed_vertex_fraction": 0.0,
    }:
        raise AssertionError("empty summary must be stable")


TESTS = [
    test_policy_views_preserve_default_and_activation_pool,
    test_activation_is_explicit_and_rejects_untrusted_ids,
    test_existence_and_association_uncertainty_stay_separate,
    test_composition_lost_accepted_mask_is_raw_activation_candidate,
    test_logit_scores_use_native_thresholds_and_sigmoid_evidence,
    test_validation_and_empty_summary,
]


def main() -> int:
    failed = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
            print()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

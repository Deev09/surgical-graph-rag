"""Tests for the frozen overlapping-mask resolution rule (Colab adapter).

Run: python tests/segmenter/test_mask_resolve.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmenter.mask_resolve import MaskResolveConfig, resolve_masks

CFG = MaskResolveConfig(min_score=0.5, min_vertices=2)


def test_highest_score_wins():
    # min_vertices=1 so the single leftover vertex isn't filtered — this
    # test isolates the score rule only
    masks = np.array([[1, 1, 1, 0], [0, 1, 1, 1]])
    out = resolve_masks(masks, np.array([0.6, 0.9]),
                        MaskResolveConfig(min_score=0.5, min_vertices=1))
    if out.tolist() != [0, 1, 1, 1]:
        raise AssertionError(f"higher-scoring mask must win overlaps: {out}")
    # and WITH the default filter the 1-vertex leftover unclaims
    out2 = resolve_masks(masks, np.array([0.6, 0.9]), CFG)
    if out2.tolist() != [-1, 1, 1, 1]:
        raise AssertionError(f"filtered leftover must unclaim: {out2}")


def test_tie_goes_to_lowest_index():
    masks = np.array([[1, 1, 0], [1, 1, 1]])
    out = resolve_masks(masks, np.array([0.7, 0.7]),
                        MaskResolveConfig(min_score=0.5, min_vertices=1))
    if out.tolist() != [0, 0, 1]:
        raise AssertionError(f"tie must resolve to the lowest mask index: {out}")


def test_min_score_filters_whole_mask():
    masks = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    out = resolve_masks(masks, np.array([0.4, 0.9]), CFG)  # mask 0 under 0.5
    if out.tolist() != [-1, -1, 1, 1]:
        raise AssertionError(f"sub-threshold mask must claim nothing: {out}")


def test_min_vertices_unclaims_not_reassigns():
    # mask 0 wins vertex 0 only (1 vertex < min_vertices=2) even though
    # mask 1 also claims it — the vertex must become -1, NOT fall to mask 1
    masks = np.array([[1, 0, 0], [1, 1, 1]])
    out = resolve_masks(masks, np.array([0.9, 0.6]), CFG)
    if out.tolist() != [-1, 1, 1]:
        raise AssertionError(f"small instance must unclaim, never reassign: {out}")


def test_ids_are_original_mask_indices():
    masks = np.array([[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]])
    out = resolve_masks(masks, np.array([0.1, 0.8, 0.8]), CFG)  # mask 0 filtered
    if sorted(set(out.tolist())) != [1, 2]:
        raise AssertionError(f"ids must be original mask rows: {out}")


def test_shape_and_finite_validation():
    for masks, scores in (
        (np.zeros((2, 3)), np.zeros(3)),               # score length mismatch
        (np.zeros((2, 3)), np.array([0.5, np.nan])),   # non-finite score
    ):
        try:
            resolve_masks(masks, scores, CFG)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid inputs must raise")
    out = resolve_masks(np.zeros((0, 4)), np.zeros(0), CFG)
    if out.tolist() != [-1, -1, -1, -1]:
        raise AssertionError(f"zero masks -> all unclaimed: {out}")


TESTS = [
    test_highest_score_wins,
    test_tie_goes_to_lowest_index,
    test_min_score_filters_whole_mask,
    test_min_vertices_unclaims_not_reassigns,
    test_ids_are_original_mask_indices,
    test_shape_and_finite_validation,
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

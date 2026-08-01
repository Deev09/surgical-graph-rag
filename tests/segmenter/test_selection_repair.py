"""Tests for segmenter/selection_repair.py (rule v1, merge decomposition).

Run: python tests/segmenter/test_selection_repair.py
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
from segmenter.selection_repair import repair_selection


def test_merge_parent_suppressed_children_admitted():
    # merged A+B (0.95, base) decomposes into exact A (0.9, base) and
    # exact B (0.05, below base) -> parent suppressed, B admitted at 0.95
    masks = np.zeros((3, 16), dtype=bool)
    masks[0, 0:8] = True
    masks[1, :] = True
    masks[2, 8:16] = True
    scores = np.array([0.9, 0.95, 0.05])
    pr, diag = repair_selection(masks, scores)
    if pr[1] >= 0:
        raise AssertionError(f"merged mask must be suppressed: {pr}")
    if pr[0] != 0.95 or pr[2] != 0.95:
        raise AssertionError(f"children must inherit parent priority: {pr}")
    if diag["n_suspects_suppressed"] != 1 or diag["n_admitted_below_base_score"] != 1:
        raise AssertionError(f"diagnostics wrong: {diag}")
    ids = resolve_masks(masks, pr, MaskResolveConfig(min_score=0.0, min_vertices=4))
    if not (len(set(ids[0:8])) == 1 and len(set(ids[8:16])) == 1
            and ids[0] != ids[8] and ids[0] >= 0 and ids[8] >= 0):
        raise AssertionError(f"resolution must yield two clean instances: {ids}")


def test_single_child_is_not_a_merge():
    # a mask with ONE contained child (a duplicate/refinement) is not a
    # merge suspect — nothing is suppressed
    masks = np.zeros((2, 16), dtype=bool)
    masks[0, 0:12] = True
    masks[1, 0:5] = True
    scores = np.array([0.9, 0.05])
    pr, diag = repair_selection(masks, scores)
    if pr[0] != 0.9 or diag["n_suspects_suppressed"] != 0:
        raise AssertionError(f"single-child parent must survive: {pr} {diag}")
    if pr[1] >= 0:
        raise AssertionError(f"non-admitted sub-base mask stays excluded: {pr}")


def test_overlapping_children_do_not_count_twice():
    # two heavily-overlapping children are one region, not two objects —
    # greedy disjoint filter keeps one, so the parent is NOT a suspect
    masks = np.zeros((3, 16), dtype=bool)
    masks[0, 0:12] = True
    masks[1, 0:6] = True
    masks[2, 1:6] = True     # overlap ratio with mask1 = 5/5 = 1.0
    scores = np.array([0.9, 0.3, 0.25])
    pr, diag = repair_selection(masks, scores)
    if diag["n_suspects_suppressed"] != 0 or pr[0] != 0.9:
        raise AssertionError(f"overlapping children must not flag a merge: {diag}")


def test_low_cover_children_do_not_flag():
    # two tiny disjoint children covering < parent_cover_min of the parent
    masks = np.zeros((3, 32), dtype=bool)
    masks[0, 0:32] = True
    masks[1, 0:4] = True
    masks[2, 8:12] = True
    scores = np.array([0.9, 0.3, 0.3])
    pr, diag = repair_selection(masks, scores)
    if diag["n_suspects_suppressed"] != 0:
        raise AssertionError(f"cover 8/32 must not flag: {diag}")


def _plate_in_sofa():
    """40 verts: sofa = 0..29, plate = 30..39. The merge (sofa+plate) has
    the top score; the exact plate mask scores below base but has a
    duplicate corroborator; the exact sofa mask is a base near-duplicate."""
    masks = np.zeros((4, 40), dtype=bool)
    masks[0, :] = True        # sofa+plate merge, 1.5
    masks[1, 30:40] = True    # plate exact, -0.3 (below base)
    masks[2, 30:40] = True    # plate duplicate (corroborator), -0.5
    masks[3, 0:30] = True     # sofa exact, 0.9
    scores = np.array([1.5, -0.3, -0.5, 0.9])
    return masks, scores


def test_v2_carveout_recovers_swallowed_plate():
    from segmenter.selection_repair import SelectionRepairV2Config, repair_selection_v2
    masks, scores = _plate_in_sofa()
    ids, diag = repair_selection_v2(masks, scores, SelectionRepairV2Config(),
                                    min_vertices=5)
    if len(set(ids[30:40])) != 1 or len(set(ids[0:30])) != 1 \
            or ids[30] == ids[0] or ids[30] < 0 or ids[0] < 0:
        raise AssertionError(f"plate must be carved out of the merge: {ids}")
    if diag["n_promoted"] < 1:
        raise AssertionError(f"plate mask must be promoted: {diag}")


def test_v3_carveout_and_decomposition():
    from segmenter.selection_repair import SelectionRepairV3Config, repair_selection_v3
    # plate/sofa carve-out (plate = 0.25 of merge exactly); v3 requires
    # >= 2 corroborators, so the plate needs two duplicate proposals
    base_masks, base_scores = _plate_in_sofa()
    masks = np.vstack([base_masks, np.zeros((1, 40), dtype=bool)])
    masks[4, 30:40] = True    # second plate duplicate
    scores = np.append(base_scores, -0.6)
    ids, diag = repair_selection_v3(masks, scores, SelectionRepairV3Config(),
                                    min_vertices=5)
    if ids[30] == ids[0] or ids[30] < 0 or ids[0] < 0:
        raise AssertionError(f"v3 carve-out failed: {ids} {diag}")
    # two-component merge: decomposition mechanism (components are 0.5 of
    # the parent — beyond carve_max_frac, so only decomposition fires)
    masks2 = np.zeros((3, 16), dtype=bool)
    masks2[0, :] = True       # merge A+B, 0.95
    masks2[1, 0:8] = True     # exact A, 0.9 (base)
    masks2[2, 8:16] = True    # exact B, 0.05 (below base)
    scores2 = np.array([0.95, 0.9, 0.05])
    ids2, diag2 = repair_selection_v3(masks2, scores2,
                                      SelectionRepairV3Config(),
                                      min_vertices=4)
    if diag2["n_promoted_by_mechanism"].get("decomposition", 0) < 2:
        raise AssertionError(f"decomposition must fire: {diag2}")
    if ids2[0] == ids2[8] or ids2[0] < 0 or ids2[8] < 0:
        raise AssertionError(f"v3 must split the merge: {ids2}")


TESTS = [
    test_merge_parent_suppressed_children_admitted,
    test_single_child_is_not_a_merge,
    test_overlapping_children_do_not_count_twice,
    test_low_cover_children_do_not_flag,
    test_v2_carveout_recovers_swallowed_plate,
    test_v3_carveout_and_decomposition,
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

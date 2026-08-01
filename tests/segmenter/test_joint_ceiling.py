"""Tests for tools/c1_joint_ceiling.py (joint oracle-guided selection).

Run: python tests/segmenter/test_joint_ceiling.py

Synthetic two-cube scene where the frozen composition provably loses cube B:
a high-scoring MERGED mask (A+B, score 0.95) outranks cube A's exact mask
(0.9), and cube B's exact mask sits BELOW the frozen min_score (0.05). Joint
selection must nominate both exact masks, materialize them through the
frozen resolver, and recover 2/2 entities in both variants — with the
merged mask reduced to nothing in selected_plus_rest (its vertices are all
claimed by boosted exact masks, so min_vertices unclaims it).
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.segmenter.test_c1_pipeline import _scene
from tools.c1_joint_ceiling import measure_joint, select_joint_masks


def _merged_scene(tmp: Path):
    """Delivered pred: the merged mask wins everything (one 16-vertex
    instance). Raw masks: exact A (0.9), merged A+B (0.95), exact B (0.05)."""
    pred = np.array([30] * 16)   # what frozen WTA @0.2 produces here
    room, bundle = _scene(tmp, pred)
    masks = np.zeros((3, 16), dtype=bool)
    masks[0, 0:8] = True     # exact cube A
    masks[1, :] = True       # merged A+B — highest score
    masks[2, 8:16] = True    # exact cube B — below frozen min_score
    scores = np.array([0.9, 0.95, 0.05])
    np.savez_compressed(bundle / "raw_masks.npz",
                        masks_packed=np.packbits(masks, axis=1),
                        n_vertices=np.int64(16), scores=scores)
    return room, bundle


def test_select_joint_masks_prefers_exact_over_merged():
    oracle = np.array([1] * 8 + [2] * 8)
    masks = np.zeros((3, 16), dtype=bool)
    masks[0, 0:8] = True
    masks[1, :] = True
    masks[2, 8:16] = True
    sel = select_joint_masks(masks, oracle, [1, 2])
    if sel[1]["mask"] != 0 or sel[2]["mask"] != 2:
        raise AssertionError(f"exact masks must win nomination: {sel}")
    if sel[1]["iou"] != 1.0 or sel[2]["iou"] != 1.0:
        raise AssertionError(f"nominated IoUs must be perfect: {sel}")
    if sel[1]["rank"] != 0 or sel[2]["rank"] != 0:
        raise AssertionError(f"no collision fallback expected: {sel}")


def test_joint_variants_recover_both_cubes():
    with tempfile.TemporaryDirectory() as td:
        room, bundle = _merged_scene(Path(td))
        r = measure_joint(room, bundle, "synthetic_scene", None,
                          Path(td) / "out", min_vertices=4)
        s = r["selection"]
        if s["n_selected"] != 2 or s["n_collision_fallbacks"] != 0:
            raise AssertionError(f"selection wrong: {s}")
        if s["n_selected_below_frozen_min_score"] != 1:
            raise AssertionError(f"cube B's 0.05 mask must be flagged: {s}")
        rows = r["rows"]
        if rows["delivered"]["entity_matches_at_05"] == "2/2":
            raise AssertionError("premise broken: delivered must NOT be perfect")
        for v in ("joint_selected_only", "joint_selected_plus_rest"):
            if rows[v]["entity_matches_at_05"] != "2/2":
                raise AssertionError(f"{v} must recover both cubes: {rows[v]}")
            if rows[v]["n_pred_instances"] != 2:
                raise AssertionError(f"{v}: merged mask must end up empty "
                                     f"(min_vertices unclaims): {rows[v]}")
        if rows["delivered"]["qa_vs_human_key"] is not None:
            raise AssertionError("no key given — QA section must be None")


TESTS = [
    test_select_joint_masks_prefers_exact_over_merged,
    test_joint_variants_recover_both_cubes,
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

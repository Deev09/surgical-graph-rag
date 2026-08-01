"""Tests for tools/c1_composition_ceiling.py (oracle-guided union ceiling).

Run: python tests/segmenter/test_composition_ceiling.py

Reuses the synthetic two-cube scene from test_c1_pipeline. Cube A gets one
perfect raw mask; cube B exists only as three fragments (3+3+2 vertices), so
its best SINGLE mask sits at IoU 0.375 while unions climb 0.375 -> 0.75 ->
1.0. The dense prediction delivers the fragments as separate instances, so
cube B is winnable by union but not by selection — exactly the distinction
the tool exists to measure.
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
from tools.c1_composition_ceiling import greedy_union_curve, measure


def _fragmented_scene(tmp: Path):
    """Cube A = one instance + one perfect mask; cube B = three fragment
    instances and three fragment masks (3, 3, 2 vertices)."""
    pred = np.array([10] * 8 + [21] * 3 + [22] * 3 + [23] * 2)
    room, bundle = _scene(tmp, pred)
    masks = np.zeros((4, 16), dtype=bool)
    masks[0, 0:8] = True     # cube A, perfect
    masks[1, 8:11] = True    # cube B fragment, IoU 3/8
    masks[2, 11:14] = True   # cube B fragment, IoU 3/8
    masks[3, 14:16] = True   # cube B fragment, IoU 2/8
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    np.savez_compressed(bundle / "raw_masks.npz",
                        masks_packed=np.packbits(masks, axis=1),
                        n_vertices=np.int64(16), scores=scores)
    return room, bundle


def test_greedy_union_curve_climbs_and_stops():
    ent = np.zeros(16, dtype=bool)
    ent[8:16] = True
    frags = [np.zeros(16, dtype=bool) for _ in range(4)]
    frags[0][8:11] = True
    frags[1][11:14] = True
    frags[2][14:16] = True
    frags[3][0:8] = True     # pure neighbor mask: adding it only hurts IoU
    curve = greedy_union_curve(ent, frags)
    if curve != [0.375, 0.75, 1.0]:
        raise AssertionError(f"expected [0.375, 0.75, 1.0], got {curve}")


def test_selection_vs_union_ceilings():
    with tempfile.TemporaryDirectory() as td:
        room, bundle = _fragmented_scene(Path(td))
        r = measure(room, bundle)
        c = r["recall_ceilings_at_iou"]
        if c["single"]["0.5"] != 0.5:
            raise AssertionError(f"only cube A reachable by selection: {c}")
        if c["union2"]["0.5"] != 1.0 or c["union3"]["0.75"] != 1.0:
            raise AssertionError(f"unions must recover cube B: {c}")
        if c["dense_greedy_iou"]["0.5"] != 0.5:
            raise AssertionError(f"delivered must reflect the fragmented pred: {c}")
        if r["n_winnable_by_union3"] != 1:
            raise AssertionError(f"exactly cube B is winnable: {r['winnable_by_union3']}")
        w = r["winnable_by_union3"][0]
        if w["class"] != "chair" or w["single"] >= 0.5 or w["union3"] != 1.0:
            raise AssertionError(f"winnable row wrong: {w}")


TESTS = [
    test_greedy_union_curve_climbs_and_stops,
    test_selection_vs_union_ceilings,
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

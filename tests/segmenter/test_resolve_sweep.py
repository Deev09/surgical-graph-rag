"""Tests for tools/c1_resolve_sweep.py (raw-mask re-resolution sweep).

Run: python tests/segmenter/test_resolve_sweep.py

Reuses the synthetic two-cube scene from test_c1_pipeline; raw masks are
three proposals: cube A (score .9), cube B (score .5), and a low-score
duplicate of cube B (.2). Sweeping min_score over [0.0, 0.4, 0.6] must show
monotonically fewer instances and a recall drop once cube B's mask is cut.
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

from segmenter.base import SegmentationOutput, save_segmentation_output, sha256_file
from segmenter.mask_resolve import MaskResolveConfig, resolve_masks
from tests.segmenter.test_c1_pipeline import _scene
from tools.c1_resolve_sweep import load_raw_masks, sweep


def _bundle_with_raw_masks(tmp: Path):
    """Synthetic room + a bundle whose sidecar AND raw_masks.npz agree."""
    masks = np.zeros((3, 16), dtype=bool)
    masks[0, 0:8] = True     # cube A, score .9
    masks[1, 8:16] = True    # cube B, score .5
    masks[2, 8:12] = True    # low-score partial duplicate of B, score .2
    scores = np.array([0.9, 0.5, 0.2])
    ids = resolve_masks(masks, scores, MaskResolveConfig(min_score=0.0, min_vertices=4))
    room, bundle = _scene(tmp, ids)
    np.savez_compressed(bundle / "raw_masks.npz",
                        masks_packed=np.packbits(masks, axis=1),
                        n_vertices=np.int64(16), scores=scores)
    return room, bundle, masks, scores


def test_raw_masks_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        _, bundle, masks, scores = _bundle_with_raw_masks(Path(td))
        m2, s2 = load_raw_masks(bundle)
        if not np.array_equal(m2, masks) or not np.array_equal(s2, scores):
            raise AssertionError("packbits round-trip must be exact")


def test_sweep_monotone_and_recall_drop():
    with tempfile.TemporaryDirectory() as td:
        room, bundle, _, _ = _bundle_with_raw_masks(Path(td))
        r = sweep(room, bundle, [0.0, 0.4, 0.6], min_vertices=4)
        v = {x["min_score"]: x for x in r["variants"]}
        if v[0.0]["n_pred_instances"] < v[0.4]["n_pred_instances"] or \
           v[0.4]["n_pred_instances"] < v[0.6]["n_pred_instances"]:
            raise AssertionError(f"instance count must not grow with min_score: {v}")
        if v[0.4]["entity_recall_at_iou"]["0.5"] != 1.0:
            raise AssertionError(f"both cubes must be recovered at 0.4: {v[0.4]}")
        if v[0.6]["entity_recall_at_iou"]["0.5"] >= 1.0:
            raise AssertionError(f"cutting cube B's mask (score .5) must drop recall: {v[0.6]}")


TESTS = [
    test_raw_masks_roundtrip,
    test_sweep_monotone_and_recall_drop,
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

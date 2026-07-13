"""C1 resolution-threshold sweep — how much of the recall gap is MIN_SCORE?

    python3 tools/c1_resolve_sweep.py <room_dir> <bundle_dir> \
        [--min-scores 0.0 0.1 0.2 0.3 0.4 0.5 0.6] [--min-vertices 20]

Consumes the raw mask evidence (`raw_masks.npz`: packbits bool masks +
scores, written by the Colab notebook alongside the sidecar), re-resolves
the SAME saved masks at each MIN_SCORE with the frozen rule
(segmenter/mask_resolve.py), and scores every variant with the exact
evaluator (tools/c1_exact_eval.py). One variable moves: the score cutoff.
No GPU, no re-inference.

Output: runs/phase8_c1/<scene>_resolve_sweep.json + a stdout table. This is
a measurement, not a tuning pass — pick an operating point AFTER seeing the
distribution across scenes, and record the choice as a benchmark-definition
decision (docs/mesh_pipeline_contract.md honesty rules apply).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmenter.base import (
    SegmentationOutput, load_segmentation_output, save_segmentation_output,
)
from segmenter.mask_resolve import MaskResolveConfig, resolve_masks
from tools.c1_exact_eval import evaluate


def load_raw_masks(bundle_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (masks [K, N] bool, scores [K]) from raw_masks.npz."""
    z = np.load(bundle_dir / "raw_masks.npz")
    n = int(z["n_vertices"])
    masks = np.unpackbits(z["masks_packed"], axis=1, count=n).astype(bool)
    return masks, z["scores"].astype(float)


def sweep(room_dir: Path, bundle_dir: Path,
          min_scores: list[float], min_vertices: int) -> dict:
    base = load_segmentation_output(bundle_dir)   # validates hashes/meta
    masks, scores = load_raw_masks(bundle_dir)
    if masks.shape[1] != base.n_vertices:
        raise ValueError(f"raw masks are {masks.shape[1]} vertices, "
                         f"sidecar says {base.n_vertices}")

    variants = []
    for ms in min_scores:
        cfg = MaskResolveConfig(min_score=ms, min_vertices=min_vertices)
        ids = resolve_masks(masks, scores, cfg)
        seg = SegmentationOutput(
            input_mesh_sha256=base.input_mesh_sha256,
            n_vertices=base.n_vertices,
            segmenter_name=base.segmenter_name,
            segmenter_version=base.segmenter_version,
            config_params_json=json.dumps(
                {"resolve_sweep_variant": cfg.params()}, sort_keys=True),
            vertex_instance_ids=ids,
            # instance ids are original mask rows -> confidence = raw score
            instance_confidence={int(i): float(scores[i])
                                 for i in np.unique(ids) if i >= 0},
        ).finalize()
        with tempfile.TemporaryDirectory() as td:
            vdir = Path(td) / "variant"
            save_segmentation_output(seg, vdir)
            r = evaluate(room_dir, vdir)
        variants.append({
            "min_score": ms,
            "min_vertices": min_vertices,
            "n_pred_instances": r["n_pred_instances"],
            "n_matched": r["n_matched"],
            "entity_recall_at_iou": r["recall_at_iou"],
            "support_owner_recall_at_iou": r["support_owner"]["recall_at_iou"],
            "unassigned_vertex_frac": r["unassigned_vertex_frac_pred"],
            "n_oracle_objects_split": r["over_segmentation"]["n_oracle_objects_split"],
            "n_predictions_merging": r["under_segmentation"]["n_predictions_merging"],
        })
    return {
        "schema": "c1_resolve_sweep_v1",
        "scene_dir": str(room_dir),
        "bundle_dir": str(bundle_dir),
        "segmenter": {"name": base.segmenter_name, "version": base.segmenter_version},
        "n_raw_masks": int(masks.shape[0]),
        "note": ("same saved masks re-resolved per variant; a measurement of "
                 "the score cutoff, not a tuned benchmark setting"),
        "variants": variants,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("room_dir", type=Path)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--min-scores", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--min-vertices", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        report = sweep(args.room_dir, args.bundle_dir,
                       args.min_scores, args.min_vertices)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[c1_resolve_sweep] HARD FAIL: {exc}")
        return 1

    out = args.out
    if out is None:
        out = REPO_ROOT / "runs" / "phase8_c1" / f"{args.room_dir.name}_resolve_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"[c1_resolve_sweep] {args.room_dir.name}: {report['n_raw_masks']} raw masks")
    print(f"{'min_score':>9} {'n_pred':>6} {'matched':>7} {'ent@.5':>7} "
          f"{'task@.5':>7} {'unassigned':>10} {'split':>5} {'merged':>6}")
    for v in report["variants"]:
        task = v["support_owner_recall_at_iou"]["0.5"]
        print(f"{v['min_score']:9.2f} {v['n_pred_instances']:6d} {v['n_matched']:7d} "
              f"{v['entity_recall_at_iou']['0.5']:7.2f} "
              f"{'  n/a' if task is None else f'{task:7.2f}'} "
              f"{v['unassigned_vertex_frac']:10.3f} "
              f"{v['n_oracle_objects_split']:5d} {v['n_predictions_merging']:6d}")
    print(f"[c1_resolve_sweep] report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Locally re-resolve a segmentation bundle at a chosen MIN_SCORE (zero GPU).

    python3 tools/c1_reresolve.py <bundle_dir> <out_dir> \
        [--min-score 0.2] [--min-vertices 20]

Reads raw_masks.npz from the source bundle, resolves with the frozen rule,
and writes a COMPLETE new bundle (sidecar + copied raw_masks.npz) with:
  - instance_confidence populated from each surviving mask's raw score,
  - provenance recording both the original Colab-side resolve params and
    this local re-resolution (config key "reresolved_locally").

Use case: freezing an operating point after the cross-scene sweep — record
that choice as an OPERATING-POINT / BENCHMARK-DEFINITION decision, never as
a model improvement (docs/mesh_pipeline_contract.md honesty rules).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmenter.base import (
    SegmentationOutput, load_segmentation_output, save_segmentation_output,
)
from segmenter.mask_resolve import MaskResolveConfig, resolve_masks
from tools.c1_resolve_sweep import load_raw_masks


def reresolve(bundle_dir: Path, out_dir: Path,
              min_score: float, min_vertices: int) -> SegmentationOutput:
    base = load_segmentation_output(bundle_dir)      # validates the source
    masks, scores = load_raw_masks(bundle_dir)
    if masks.shape[1] != base.n_vertices:
        raise ValueError(f"raw masks {masks.shape[1]} vertices != "
                         f"sidecar {base.n_vertices}")
    cfg = MaskResolveConfig(min_score=min_score, min_vertices=min_vertices)
    ids = resolve_masks(masks, scores, cfg)
    original = json.loads(base.config_params_json)
    seg = SegmentationOutput(
        input_mesh_sha256=base.input_mesh_sha256,
        n_vertices=base.n_vertices,
        segmenter_name=base.segmenter_name,
        segmenter_version=base.segmenter_version,
        config_params_json=json.dumps({
            **original,
            "reresolved_locally": {
                **cfg.params(),
                "source_bundle_output_sha256": base.output_sha256,
                "note": "operating-point re-resolution from raw_masks.npz; "
                        "benchmark-definition choice, not a model change",
            },
        }, sort_keys=True),
        vertex_instance_ids=ids,
        instance_confidence={int(i): float(scores[i])
                             for i in np.unique(ids) if i >= 0},
        runtime_seconds=base.runtime_seconds,
        hardware=base.hardware,
    ).finalize()
    save_segmentation_output(seg, out_dir)
    shutil.copy(bundle_dir / "raw_masks.npz", out_dir / "raw_masks.npz")
    return seg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--min-score", type=float, default=0.2)
    parser.add_argument("--min-vertices", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        seg = reresolve(args.bundle_dir, args.out_dir,
                        args.min_score, args.min_vertices)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[c1_reresolve] HARD FAIL: {exc}")
        return 1
    n = len(seg.instance_ids())
    print(f"[c1_reresolve] {args.bundle_dir} -> {args.out_dir}: "
          f"{n} instances at min_score={args.min_score}, "
          f"output {seg.output_sha256[:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())

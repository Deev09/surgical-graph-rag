"""C1-M2C stage 0 — composition ceiling measurement (EVALUATION-ONLY).

  python3 tools/c1_composition_ceiling.py --room-dir DIR --bundle DIR [--out FILE]

Measures, per oracle entity, the best IoU achievable from the SAVED raw
masks under increasingly powerful composition:

  single     best individual raw mask (= the known selection ceiling)
  union<=2   greedy oracle-guided union of at most 2 masks
  union<=3   greedy oracle-guided union of at most 3 masks
  union_any  greedy union until no mask improves IoU (capped at 8 steps)

This is a DIAGNOSTIC UPPER-BOUND probe, not a composer: mask selection here
reads the oracle membership directly, which no deployable system may do.
Its only purpose is to answer, before any composer is designed, "is there
enough signal in the saved fragments for fragment ASSEMBLY to beat mask
SELECTION, and by how much?" Greedy union is a lower bound on the true
best-union ceiling (exhaustive subset search is intractable), so the real
ceiling is >= the numbers reported here.

Scope: unions only. Mask differences and Segmentator-component recombination
are NOT probed — if unions already close the gap they are unnecessary; if
unions don't, that is a finding worth its own probe.

Candidate cap: per entity, only the 64 masks with the largest intersection
are considered (report records how often the cap binds — no silent caps).

Zero GPU; runs on the saved raw_masks.npz. Same entity filter as
tools/c1_failure_classes.py (STRUCTURAL_OR_DROPPED excluded).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.replica_mesh_import import _parse_semantic_ply
from segmenter.base import load_segmentation_output
from tools.c1_exact_eval import (
    STRUCTURAL_OR_DROPPED, evaluate, oracle_vertex_membership,
)
from tools.c1_resolve_sweep import load_raw_masks

CANDIDATE_CAP = 64
UNION_STEP_CAP = 8
IOU_THRESHOLDS = (0.5, 0.75)


def greedy_union_curve(ent: np.ndarray, cands: list[np.ndarray]) -> list[float]:
    """Best-IoU-so-far after greedily adding 1, 2, ... UNION_STEP_CAP masks.

    Greedy: each step adds the candidate mask that maximizes IoU of the
    running union against the oracle entity; stops early when no candidate
    improves it. Returns the (non-decreasing) IoU curve; curve[0] is the
    best single mask.
    """
    ent_size = int(ent.sum())
    cur = np.zeros_like(ent)
    cur_iou = 0.0
    used: set[int] = set()
    curve: list[float] = []
    for _ in range(UNION_STEP_CAP):
        best_k, best_iou, best_union = -1, cur_iou, None
        for k, m in enumerate(cands):
            if k in used:
                continue
            u = cur | m
            inter = int((u & ent).sum())
            iou = inter / (int(u.sum()) + ent_size - inter)
            if iou > best_iou:
                best_k, best_iou, best_union = k, iou, u
        if best_k < 0:
            break
        used.add(best_k)
        cur, cur_iou = best_union, best_iou
        curve.append(round(cur_iou, 4))
    return curve or [0.0]


def measure(room_dir: Path, bundle_dir: Path) -> dict:
    report = evaluate(room_dir, bundle_dir)
    seg = load_segmentation_output(bundle_dir)
    masks, _scores = load_raw_masks(bundle_dir)
    _, vidx, oid = _parse_semantic_ply(room_dir / "habitat" / "mesh_semantic.ply")
    oracle = oracle_vertex_membership(vidx, oid, seg.n_vertices)

    # per-mask x per-oracle intersection counts, one pass
    n_oids = int(oracle.max()) + 1
    inter = np.zeros((masks.shape[0], n_oids), dtype=np.int64)
    for k in range(masks.shape[0]):
        ids = oracle[masks[k]]
        inter[k] = np.bincount(ids[ids >= 0], minlength=n_oids)

    per_object = []
    n_cap_bound = 0
    for oid_str, cov in report["oracle_coverage"].items():
        if not cov["class"] or cov["class"] in STRUCTURAL_OR_DROPPED:
            continue
        o = int(oid_str)
        ent = oracle == o
        overlapping = np.nonzero(inter[:, o])[0]
        if len(overlapping) > CANDIDATE_CAP:
            n_cap_bound += 1
            overlapping = overlapping[
                np.argsort(inter[overlapping, o])[::-1][:CANDIDATE_CAP]]
        curve = greedy_union_curve(ent, [masks[k] for k in overlapping])
        per_object.append({
            "oracle_id": o,
            "class": cov["class"],
            "size": cov["size"],
            "dense_greedy_iou": cov["greedy_iou"],   # what composition delivered
            "single": curve[0],
            "union2": max(curve[:2]),
            "union3": max(curve[:3]),
            "union_any": curve[-1],
            "n_masks_used_at_best": len(curve),
            "n_overlapping_masks": int(len(overlapping)),
        })
    per_object.sort(key=lambda r: -r["size"])

    n = len(per_object)
    def recall(field: str, t: float) -> float | None:
        return round(sum(1 for r in per_object if r[field] >= t) / n, 4) if n else None

    ceilings = {field: {str(t): recall(field, t) for t in IOU_THRESHOLDS}
                for field in ("dense_greedy_iou", "single", "union2", "union3",
                              "union_any")}
    winnable = [r for r in per_object
                if r["union3"] >= 0.5 and r["dense_greedy_iou"] < 0.5]
    return {
        "schema": "c1_composition_ceiling_v1",
        "purpose": ("oracle-guided achievability bound over saved raw masks; "
                    "NOT a composer, NOT a deployable result"),
        "scene_dir": str(room_dir),
        "bundle_dir": str(bundle_dir),
        "n_oracle_entities": n,
        "n_raw_masks": int(masks.shape[0]),
        "candidate_cap": CANDIDATE_CAP,
        "n_entities_where_cap_bound": n_cap_bound,
        "recall_ceilings_at_iou": ceilings,
        "n_winnable_by_union3": len(winnable),
        "winnable_by_union3": [
            {k: r[k] for k in ("oracle_id", "class", "size",
                               "dense_greedy_iou", "single", "union2", "union3")}
            for r in winnable],
        "per_object": per_object,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--room-dir", required=True, type=Path)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    r = measure(args.room_dir, args.bundle)
    print(f"entities={r['n_oracle_entities']}  raw_masks={r['n_raw_masks']}  "
          f"cap_bound={r['n_entities_where_cap_bound']}")
    print(f"{'ceiling':>16}  {'R@0.5':>6}  {'R@0.75':>7}")
    for field in ("dense_greedy_iou", "single", "union2", "union3", "union_any"):
        c = r["recall_ceilings_at_iou"][field]
        name = "delivered" if field == "dense_greedy_iou" else field
        print(f"{name:>16}  {c['0.5']:>6}  {c['0.75']:>7}")
    print(f"winnable by union<=3 (>=0.5 possible, <0.5 delivered): "
          f"{r['n_winnable_by_union3']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(r, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

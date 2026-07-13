"""C1 failure classification — zero-GPU, per oracle entity, four-way.

    python3 tools/c1_failure_classes.py <room_dir> <bundle_dir>

For every oracle ENTITY-class instance, decides which stage lost it, using
the raw mask evidence (raw_masks.npz) against the dense assignment:

  recovered         greedy dense match at IoU >= 0.5
  merged            not recovered, but >=50% of its vertices sit inside a
                    prediction that is greedy-matched to a DIFFERENT object
  lost_by_resolver  some raw mask (any score) covers it at IoU >= 0.5, but
                    winner-takes-all resolution destroyed that proposal
  no_raw_proposal   no raw mask ever reached IoU 0.5 — a model limitation
                    no threshold or resolver change can recover

Relation-level consequences (recovered-but-relation-changed) are reported by
tools/c1_run.py's merge-aware attribution, not here — this tool is about
instance existence, that one about downstream effect.
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


def best_raw_iou_per_oracle(masks: np.ndarray, oracle: np.ndarray) -> dict[int, float]:
    """Max IoU any raw mask (any score) achieves against each oracle object."""
    oracle_sizes = {int(i): int(c) for i, c in
                    zip(*np.unique(oracle[oracle >= 0], return_counts=True))}
    best: dict[int, float] = {o: 0.0 for o in oracle_sizes}
    n_oids = int(oracle.max()) + 1
    for k in range(masks.shape[0]):
        m = masks[k]
        m_size = int(m.sum())
        if m_size == 0:
            continue
        ids = oracle[m]
        counts = np.bincount(ids[ids >= 0], minlength=n_oids)
        for o in np.nonzero(counts)[0]:
            inter = int(counts[o])
            iou = inter / (m_size + oracle_sizes[int(o)] - inter)
            if iou > best[int(o)]:
                best[int(o)] = iou
    return best


def classify(room_dir: Path, bundle_dir: Path) -> dict:
    report = evaluate(room_dir, bundle_dir)          # hard gates + coverage
    seg = load_segmentation_output(bundle_dir)
    masks, _scores = load_raw_masks(bundle_dir)

    _, vidx, oid = _parse_semantic_ply(room_dir / "habitat" / "mesh_semantic.ply")
    oracle = oracle_vertex_membership(vidx, oid, seg.n_vertices)
    best_raw = best_raw_iou_per_oracle(masks, oracle)

    matched_owner = {m["pred_id"]: m["oracle_id"] for m in report["matches"]}
    per_object = []
    counts = {"recovered": 0, "merged": 0, "lost_by_resolver": 0, "no_raw_proposal": 0}
    for oid_str, cov in report["oracle_coverage"].items():
        if not cov["class"] or cov["class"] in STRUCTURAL_OR_DROPPED:
            continue
        o = int(oid_str)
        raw = best_raw.get(o, 0.0)
        if cov["greedy_iou"] >= 0.5:
            cls = "recovered"
        elif (cov["top_pred"] is not None and cov["covered_frac"] >= 0.5
              and matched_owner.get(cov["top_pred"]) not in (None, o)):
            cls = "merged"
        elif raw >= 0.5:
            cls = "lost_by_resolver"
        else:
            cls = "no_raw_proposal"
        counts[cls] += 1
        per_object.append({
            "oracle_id": o, "class": cov["class"], "size": cov["size"],
            "failure_class": cls,
            "best_raw_iou": round(raw, 3),
            "dense_greedy_iou": cov["greedy_iou"],
            "covered_frac": cov["covered_frac"],
            "top_pred": cov["top_pred"],
        })
    per_object.sort(key=lambda r: (r["failure_class"], -r["size"]))
    # merged-vs-resolver overlap: a merged object that ALSO had a viable raw
    # proposal (IoU >= 0.5) was destroyed by winner-takes-all resolution,
    # not by the model — the office_0 desk-organizer case. Counted
    # separately so "merged" isn't read as purely a proposal problem.
    n_merged_viable = sum(1 for r in per_object
                          if r["failure_class"] == "merged"
                          and r["best_raw_iou"] >= 0.5)
    return {
        "schema": "c1_failure_classes_v1",
        "scene_dir": str(room_dir),
        "bundle_dir": str(bundle_dir),
        "n_oracle_entities": len(per_object),
        "counts": counts,
        "n_merged_with_viable_raw_proposal": n_merged_viable,
        "per_object": per_object,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("room_dir", type=Path)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        r = classify(args.room_dir, args.bundle_dir)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[c1_failure_classes] HARD FAIL: {exc}")
        return 1
    out = args.out or (REPO_ROOT / "runs" / "phase8_c1"
                       / f"{args.room_dir.name}_failure_classes.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1), encoding="utf-8")
    c = r["counts"]
    print(f"[c1_failure_classes] {args.room_dir.name}: "
          f"{r['n_oracle_entities']} oracle entities -> "
          f"recovered={c['recovered']} merged={c['merged']} "
          f"lost_by_resolver={c['lost_by_resolver']} "
          f"no_raw_proposal={c['no_raw_proposal']}")
    print(f"[c1_failure_classes] report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

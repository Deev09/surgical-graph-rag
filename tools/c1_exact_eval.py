"""C1.00 — exact segmentation evaluator (vertex-index correspondence).

Scores a segmenter's dense vertex assignment against Replica's oracle
instances WITHOUT spatial matching: mesh.ply and habitat/mesh_semantic.ply
carry identical vertex arrays in identical index order (verified + hard-
checked here as gate G1), so oracle membership transfers per vertex index.

    python3 tools/c1_exact_eval.py <room_dir> <segmentation_bundle_dir> \
        [--out report.json]

Oracle vertex membership: a vertex's oracle object_id is the majority id over
its incident semantic-mesh faces (ties -> smallest id); vertices touched by
no face are background (-1).

Report (no pass/fail accuracy threshold by design — the first C1 runs measure
the distribution; adoption thresholds get chosen afterwards):
  - per matched instance: vertex IoU, sizes, oracle class label
  - matched/unmatched predictions, oracle object recall (matched fraction,
    plus informational recall at IoU 0.25/0.50/0.75)
  - over-segmentation  (oracle objects covered >=10% by >=2 predictions)
  - under-segmentation (predictions covering >=10% of >=2 oracle objects)
  - background/unassigned vertex fractions on both sides
  - support-OWNER class recall (recovery of tables/chairs/shelves — the
    furniture support questions target; NOT the supported items themselves)

Exit codes: 0 = evaluated; 1 = hard invariant failed (vertex arrays differ,
bundle/mesh hash mismatch, length mismatch) — those are contract violations,
not low scores.
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
from segmenter.base import load_segmentation_output, sha256_file
from segmenter.ply import parse_vertices

# classes the question battery's support questions target as OWNERS (see
# demo/question_battery.SUPPORT_Q). NOTE the honest name: recall on these
# measures recovery of the supporting furniture (tables/chairs/shelves),
# NOT of the objects sitting on them — supported-item recovery shows up in
# c1_run's per-question answer recall instead.
SUPPORT_OWNER_CLASSES = ("table", "desk", "shelf", "counter", "stool",
                         "bench", "sofa", "chair", "plant-stand")

STRUCTURAL_OR_DROPPED = ("floor", "wall", "ceiling", "undefined", "non-plane", "plane")


def oracle_vertex_membership(vidx: np.ndarray, oid: np.ndarray, n_vertices: int) -> np.ndarray:
    """Majority incident-face object_id per vertex; ties -> smallest id;
    untouched vertices -> -1."""
    k = vidx.shape[1]
    v = vidx.reshape(-1)
    o = np.repeat(oid, k)
    pairs = np.stack([v, o], axis=1)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    uv, uo = uniq[:, 0], uniq[:, 1]
    # order: vertex asc, count desc, oid asc -> first row per vertex wins
    order = np.lexsort((uo, -counts, uv))
    uv, uo = uv[order], uo[order]
    first = np.unique(uv, return_index=True)[1]
    out = np.full(n_vertices, -1, dtype=np.int64)
    out[uv[first]] = uo[first]
    return out


def overlap_counts(pred: np.ndarray, oracle: np.ndarray):
    """{(pred_id, oracle_id): n_shared_vertices} over vertices assigned on
    both sides, plus per-id totals."""
    both = (pred >= 0) & (oracle >= 0)
    pairs = np.stack([pred[both], oracle[both]], axis=1)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    overlaps = {(int(p), int(o)): int(c) for (p, o), c in zip(uniq, counts)}
    pred_sizes = {int(i): int(c) for i, c in
                  zip(*np.unique(pred[pred >= 0], return_counts=True))}
    oracle_sizes = {int(i): int(c) for i, c in
                    zip(*np.unique(oracle[oracle >= 0], return_counts=True))}
    return overlaps, pred_sizes, oracle_sizes


def greedy_match(overlaps: dict, pred_sizes: dict, oracle_sizes: dict):
    """Maximum-overlap greedy 1:1 matching. Returns
    [(pred_id, oracle_id, n_overlap, iou)] sorted by overlap desc."""
    matches = []
    used_p: set[int] = set()
    used_o: set[int] = set()
    for (p, o), n in sorted(overlaps.items(), key=lambda kv: (-kv[1], kv[0])):
        if p in used_p or o in used_o:
            continue
        used_p.add(p)
        used_o.add(o)
        union = pred_sizes[p] + oracle_sizes[o] - n
        matches.append((p, o, n, n / union if union else 0.0))
    return matches


def evaluate(room_dir: Path, bundle_dir: Path) -> dict:
    raw_path = room_dir / "mesh.ply"
    sem_path = room_dir / "habitat" / "mesh_semantic.ply"

    seg = load_segmentation_output(bundle_dir)
    raw_sha = sha256_file(raw_path)
    if raw_sha != seg.input_mesh_sha256:
        raise ValueError(f"bundle was produced from a different mesh: "
                         f"{seg.input_mesh_sha256[:16]}... vs {raw_sha[:16]}...")

    xyz_raw = parse_vertices(raw_path)
    xyz_sem, vidx, oid = _parse_semantic_ply(sem_path)
    if len(xyz_raw) != len(xyz_sem) or not np.array_equal(xyz_raw, xyz_sem):
        raise ValueError("G1 violated: raw and semantic vertex arrays are not "
                         "identical by index — exact correspondence unavailable")
    if len(seg.vertex_instance_ids) != len(xyz_raw):
        raise ValueError(f"assignment length {len(seg.vertex_instance_ids)} != "
                         f"mesh vertices {len(xyz_raw)}")

    pred = seg.vertex_instance_ids
    oracle = oracle_vertex_membership(vidx, oid, len(xyz_raw))

    info = json.loads((room_dir / "habitat" / "info_semantic.json").read_text())
    id_to_class = {int(o["id"]): str(o.get("class_name", "")).strip()
                   for o in info["objects"]}

    overlaps, pred_sizes, oracle_sizes = overlap_counts(pred, oracle)
    matches = greedy_match(overlaps, pred_sizes, oracle_sizes)
    matched_p = {m[0] for m in matches}
    matched_o = {m[1] for m in matches}

    # over/under-segmentation at the 10%-of-oracle-object rule
    covering: dict[int, set[int]] = {}
    covered_by: dict[int, set[int]] = {}
    for (p, o), n in overlaps.items():
        if n >= 0.10 * oracle_sizes[o]:
            covering.setdefault(o, set()).add(p)
            covered_by.setdefault(p, set()).add(o)
    split_oracle = sorted(o for o, ps in covering.items() if len(ps) >= 2)
    merged_pred = sorted(p for p, os_ in covered_by.items() if len(os_) >= 2)

    entity_oracle = {o: c for o, c in ((o, id_to_class.get(o, "")) for o in oracle_sizes)
                     if c and c not in STRUCTURAL_OR_DROPPED}
    iou_by_oracle = {m[1]: m[3] for m in matches}

    def recall_at(t: float, ids) -> float | None:
        """None (JSON null) when the id set is empty — a scene with zero
        support-owner objects has no recall, not a recall of 0."""
        ids = list(ids)
        if not ids:
            return None
        return sum(1 for o in ids if iou_by_oracle.get(o, 0.0) >= t) / len(ids)

    owner_ids = [o for o, c in entity_oracle.items() if c in SUPPORT_OWNER_CLASSES]

    # matches-at-IoU counts, entity classes only — n_matched alone is
    # any-overlap greedy pairing incl. structural classes, NOT detection
    # recall; these counts are the honest companions
    n_entity_matches_at_iou = {
        str(t): sum(1 for o in entity_oracle if iou_by_oracle.get(o, 0.0) >= t)
        for t in (0.25, 0.50, 0.75)
    }

    # merge-aware coverage: for EVERY oracle instance, the single prediction
    # covering most of its vertices (matched or not) and the covered
    # fraction — the data c1_run needs to attribute a lost answer to
    # "merged into pred X" vs "no viable proposal" instead of a blanket
    # "missed by segmenter"
    top_cover: dict[int, tuple[int, int]] = {}
    for (p, o), n in overlaps.items():
        if o not in top_cover or n > top_cover[o][1]:
            top_cover[o] = (p, n)
    greedy_pred_of = {m[1]: m[0] for m in matches}
    oracle_coverage = {
        str(o): {
            "class": id_to_class.get(o, ""),
            "size": oracle_sizes[o],
            "top_pred": (top_cover[o][0] if o in top_cover else None),
            "covered_frac": (round(top_cover[o][1] / oracle_sizes[o], 4)
                             if o in top_cover else 0.0),
            "greedy_pred": greedy_pred_of.get(o),
            "greedy_iou": round(iou_by_oracle.get(o, 0.0), 4),
        }
        for o in oracle_sizes
    }

    report = {
        "schema": "c1_exact_eval_v1",
        "scene_dir": str(room_dir),
        "bundle_dir": str(bundle_dir),
        "segmenter": {"name": seg.segmenter_name, "version": seg.segmenter_version,
                      "config_params_json": seg.config_params_json,
                      "output_sha256": seg.output_sha256},
        "g1_vertex_arrays_identical": True,
        "n_vertices": int(len(xyz_raw)),
        "n_pred_instances": len(pred_sizes),
        "n_oracle_instances": len(oracle_sizes),
        "n_oracle_entity_instances": len(entity_oracle),
        "unassigned_vertex_frac_pred": float((pred < 0).mean()),
        "background_vertex_frac_oracle": float((oracle < 0).mean()),
        "n_matched": len(matches),
        "n_matched_note": "any-overlap greedy 1:1 pairs incl. structural "
                          "classes — NOT detection recall; see "
                          "n_entity_matches_at_iou",
        "n_entity_matches_at_iou": n_entity_matches_at_iou,
        "n_unmatched_predictions": len(pred_sizes) - len(matched_p),
        "n_unmatched_oracle": len(oracle_sizes) - len(matched_o),
        "oracle_recall_matched_frac": (len(matched_o) / len(oracle_sizes)
                                       if oracle_sizes else 0.0),
        "recall_at_iou": {str(t): recall_at(t, entity_oracle)
                          for t in (0.25, 0.50, 0.75)},
        "support_owner": {
            "note": "recovery of the SUPPORTING furniture only, not of the "
                    "objects on it (see c1_run per-question answer recall)",
            "classes": list(SUPPORT_OWNER_CLASSES),
            "n_oracle_objects": len(owner_ids),
            "recall_at_iou": {str(t): recall_at(t, owner_ids)
                              for t in (0.25, 0.50, 0.75)},
        },
        "oracle_coverage": oracle_coverage,
        "over_segmentation": {"n_oracle_objects_split": len(split_oracle),
                              "split_oracle_ids": split_oracle[:50]},
        "under_segmentation": {"n_predictions_merging": len(merged_pred),
                               "merging_pred_ids": merged_pred[:50]},
        "matches": [
            {"pred_id": p, "oracle_id": o, "overlap": n, "iou": round(iou, 4),
             "oracle_class": id_to_class.get(o, ""),
             "pred_size": pred_sizes[p], "oracle_size": oracle_sizes[o]}
            for p, o, n, iou in matches
        ],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("room_dir", type=Path)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        report = evaluate(args.room_dir, args.bundle_dir)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[c1_exact_eval] HARD FAIL: {exc}")
        return 1

    out = args.out
    if out is None:
        out = REPO_ROOT / "runs" / "phase8_c1" / f"{args.room_dir.name}_exact_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    ious = [m["iou"] for m in report["matches"]]
    print(f"[c1_exact_eval] {args.room_dir.name}: pred={report['n_pred_instances']} "
          f"oracle={report['n_oracle_instances']} matched={report['n_matched']} "
          f"median_iou={float(np.median(ious)) if ious else 0.0:.3f} "
          f"split={report['over_segmentation']['n_oracle_objects_split']} "
          f"merged={report['under_segmentation']['n_predictions_merging']}")
    print(f"[c1_exact_eval] report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

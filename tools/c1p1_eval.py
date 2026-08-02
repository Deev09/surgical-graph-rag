"""C1-P1 evaluator: proposal bank vs oracle + human key, gates G1-G6.

  python3 tools/c1p1_eval.py --scene replica_room_2

Opens oracle membership and the human key ONLY after the finalized,
hash-stamped bank exists (hard-checked). Reports per-entity best IoU for
Mask3D raw masks, the P1 bank, and the pooled union; evaluates the
predeclared development gates verbatim.
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
from tools.c3_surface_run import SCENE_TO_SHORT, _load_generation_inputs

GATES_DEV = {
    "G1_pooled_viable_min": 30,
    "G2_new_viable_min": 10,
    "G3_new_viable_in_key_min": 4,
    "G4_new_viable_on_furniture_min": 2,
    "G5_entity_evidence_coverage_min": 0.80,
    "G6_max_proposals": 2000,
    "G6_max_bytes": 2 * 1024 ** 3,
}

# Stage-2 transfer gates (protocol verbatim): improvement >= 5 pooled
# viable entities over the FROZEN Mask3D baseline, >= 2 newly viable
# in-key entities, same evidence-coverage guard and bank caps.
TRANSFER_BASELINES = {"replica_office_0": (13, 47), "replica_room_1": (21, 45)}
GATES_TRANSFER = {
    "H_improvement_min": 5,
    "H_new_in_key_min": 2,
    "H_evidence_coverage_min": 0.80,
    "H_max_proposals": 2000,
    "H_max_bytes": 2 * 1024 ** 3,
}


def load_bank(root: Path, scene: str):
    bj = json.loads((root / f"{scene}_bank.json").read_text())
    z = np.load(root / f"bank_{scene}.npz")
    verts, offsets = z["vertices"], z["offsets"]
    props = [verts[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
    return props, bj


def best_iou_per_entity(proposals, oracle: np.ndarray,
                        entity_oids: list[int]) -> dict[int, float]:
    sizes = {o: int((oracle == o).sum()) for o in entity_oids}
    best = {o: 0.0 for o in entity_oids}
    for p in proposals:
        ids, counts = np.unique(oracle[p], return_counts=True)
        for o, c in zip(ids.tolist(), counts.tolist()):
            if o in best:
                iou = c / (len(p) + sizes[o] - c)
                if iou > best[o]:
                    best[o] = iou
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out-root", type=Path,
                    default=REPO_ROOT / "runs" / "phase8_c1p1")
    args = ap.parse_args(argv)
    scene, root = args.scene, args.out_root
    short = SCENE_TO_SHORT[scene]

    # bank must be finalized before any oracle read
    props, bank_meta = load_bank(root, scene)

    lock = json.loads((REPO_ROOT / "tools" / "replica_scenes.lock.json")
                      .read_text())
    data_root = Path(lock["data_root_relative_to_repo"])
    room = data_root / short
    bundle = REPO_ROOT / "runs" / "phase8_c1" / "bundles_ms02" / short
    seg = load_segmentation_output(bundle)
    m3d_masks, _ = load_raw_masks(bundle)
    ev = evaluate(room, bundle)
    _, vidx, oid = _parse_semantic_ply(room / "habitat" / "mesh_semantic.ply")
    oracle = oracle_vertex_membership(vidx, oid, seg.n_vertices)
    entity_oids = [int(o) for o, cov in ev["oracle_coverage"].items()
                   if cov["class"] and cov["class"] not in STRUCTURAL_OR_DROPPED]
    classes = {int(o): cov["class"] for o, cov in ev["oracle_coverage"].items()}

    m3d_props = [np.flatnonzero(m3d_masks[k]) for k in range(len(m3d_masks))]
    best_m3d = best_iou_per_entity(m3d_props, oracle, entity_oids)
    best_p1 = best_iou_per_entity(props, oracle, entity_oids)
    best_pool = {o: max(best_m3d[o], best_p1[o]) for o in entity_oids}

    viable_m3d = {o for o in entity_oids if best_m3d[o] >= 0.5}
    viable_pool = {o for o in entity_oids if best_pool[o] >= 0.5}
    new_viable = sorted(viable_pool - viable_m3d)

    key = json.loads((REPO_ROOT / "eval" / "questions" / "phase8"
                      / f"{scene}_qa.json").read_text())
    key_cited, furniture_cited = set(), set()
    for q in key["questions"]:
        for u in q["expected_must_contain"]:
            key_cited.add(int(u.split("_")[1]))
            if q.get("relation") == "ON_ENTITY_SURFACE":
                furniture_cited.add(int(u.split("_")[1]))
    new_in_key = sorted(set(new_viable) & key_cited)
    new_on_furniture = sorted(set(new_viable) & furniture_cited)

    # G5: evidence coverage from the id buffers
    ids = np.load(root / f"views_{scene}" / "ids.npz")
    seen = np.zeros(seg.n_vertices, dtype=bool)
    for k in ids.files:
        buf = ids[k]
        seen[np.unique(buf[buf >= 0])] = True
    cov_ok = 0
    for o in entity_oids:
        member = oracle == o
        n = int(member.sum())
        if n and int((member & seen).sum()) / n >= 0.10:
            cov_ok += 1
    evidence_cov = cov_ok / len(entity_oids)

    stage = "transfer" if scene in TRANSFER_BASELINES else "dev"
    if stage == "dev":
        g = GATES_DEV
        gates = {
            "G1": len(viable_pool) >= g["G1_pooled_viable_min"],
            "G2": len(new_viable) >= g["G2_new_viable_min"],
            "G3": len(new_in_key) >= g["G3_new_viable_in_key_min"],
            "G4": len(new_on_furniture) >= g["G4_new_viable_on_furniture_min"],
            "G5": evidence_cov >= g["G5_entity_evidence_coverage_min"],
            "G6": (bank_meta["n_proposals"] <= g["G6_max_proposals"]
                   and bank_meta["serialized_bytes"] <= g["G6_max_bytes"]),
        }
    else:
        g = GATES_TRANSFER
        base_viable, base_n = TRANSFER_BASELINES[scene]
        if len(viable_m3d) != base_viable or len(entity_oids) != base_n:
            raise AssertionError(
                f"frozen anchor drift: recomputed Mask3D viable "
                f"{len(viable_m3d)}/{len(entity_oids)} vs protocol "
                f"{base_viable}/{base_n}")
        gates = {
            "H_improvement": (len(viable_pool) - base_viable
                              >= g["H_improvement_min"]),
            "H_new_in_key": len(new_in_key) >= g["H_new_in_key_min"],
            "H_evidence": evidence_cov >= g["H_evidence_coverage_min"],
            "H_caps": (bank_meta["n_proposals"] <= g["H_max_proposals"]
                       and bank_meta["serialized_bytes"] <= g["H_max_bytes"]),
        }
    report = {
        "schema": "c1p1_eval_v1",
        "protocol": "docs/c1_p1_multiview_proposals_protocol.md",
        "scene_id": scene,
        "stage": stage,
        "bank_npz_sha256": bank_meta["bank_npz_sha256"],
        "n_entities": len(entity_oids),
        "viable_mask3d": len(viable_m3d),
        "viable_p1_alone": sum(1 for o in entity_oids if best_p1[o] >= 0.5),
        "viable_pooled": len(viable_pool),
        "new_viable": [{"oid": o, "class": classes[o],
                        "p1_iou": round(best_p1[o], 3),
                        "in_key": o in key_cited,
                        "on_furniture": o in furniture_cited}
                       for o in new_viable],
        "n_new_viable": len(new_viable),
        "n_new_in_key": len(new_in_key),
        "n_new_on_furniture": len(new_on_furniture),
        "entity_evidence_coverage": round(evidence_cov, 4),
        "gate_thresholds": g,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "per_entity": [{"oid": o, "class": classes[o],
                        "mask3d": round(best_m3d[o], 3),
                        "p1": round(best_p1[o], 3),
                        "pooled": round(best_pool[o], 3)}
                       for o in entity_oids],
    }
    out = root / f"{scene}_eval.json"
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(f"pooled viable: {len(viable_pool)}/{len(entity_oids)} "
          f"(Mask3D {len(viable_m3d)}, P1-alone {report['viable_p1_alone']}); "
          f"new: {len(new_viable)} (in-key {len(new_in_key)}, "
          f"furniture {len(new_on_furniture)}); "
          f"evidence-coverage {evidence_cov:.2f}")
    for k in sorted(gates):
        print(f"  {k}: {'PASS' if gates[k] else 'FAIL'}")
    print(f"ALL GATES: {'PASS' if report['all_gates_pass'] else 'FAIL'}")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

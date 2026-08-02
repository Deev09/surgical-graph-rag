"""C1-P2 Stage P2.0 — oracle-guided QA ceiling of the pooled bank.

  python3 tools/c1p2_ceiling.py --scene replica_room_2

Protocol: docs/c1_p2_composer_protocol.md (approved). Evaluation-only
diagnostic: each oracle entity nominates its best single proposal from
the pooled `P1 ∪ Mask3D-raw` set (IoU priority; collision fallback to
the next viable proposal), the nominated set is materialized through the
FROZEN resolver, and the REAL downstream (derived bundle → graph →
Router) is scored against the human key. Nomination reads the oracle —
no deployable system may do this; the output is the reachable ceiling
that Stage-P2.1 rule gates are set against.

Also recomputes the frozen C1 reference row and hard-asserts it matches
the protocol anchors, then applies the predeclared proceed/stop rule.
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

from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer
from demo.replica_mesh_import import _parse_semantic_ply
from segmenter.base import SegmentationOutput, load_segmentation_output, save_segmentation_output
from segmenter.derived import build_c1_eval_bundle
from segmenter.mask_resolve import MaskResolveConfig, resolve_masks
from tools.c1_exact_eval import (
    STRUCTURAL_OR_DROPPED, evaluate, oracle_vertex_membership,
)
from tools.c1_joint_ceiling import _qa_for_arts, score_against_key
from tools.c1_resolve_sweep import load_raw_masks
from tools.c1p1_eval import load_bank
from tools.c3_surface_run import SCENE_TO_SHORT

# frozen anchors + predeclared proceed/stop numbers (protocol verbatim)
C1_ANCHORS = {"replica_room_2": {"micro_recall": 0.2449, "support": 2}}
PROCEED_RECALL_MIN = 0.285
PROCEED_SUPPORT_MIN = 4


def pooled_proposals(bundle_dir: Path, bank_root: Path, scene: str):
    """[(source, index, sorted vertex array)] for P1 ∪ Mask3D-raw."""
    m3d_masks, _ = load_raw_masks(bundle_dir)
    out = [("m3d", k, np.flatnonzero(m3d_masks[k]).astype(np.int64))
           for k in range(len(m3d_masks))]
    p1, _ = load_bank(bank_root, scene)
    out += [("p1", j, np.asarray(v, dtype=np.int64))
            for j, v in enumerate(p1)]
    return out


def nominate(proposals, oracle: np.ndarray, entity_oids: list[int]):
    """Oracle-guided per-entity nomination with collision fallback.
    Returns {oid: (proposal_index, iou)} for entities with best IoU>=0.5."""
    sizes = {o: int((oracle == o).sum()) for o in entity_oids}
    ranked: dict[int, list[tuple[float, int]]] = {o: [] for o in entity_oids}
    for pi, (_, _, verts) in enumerate(proposals):
        ids, counts = np.unique(oracle[verts], return_counts=True)
        for o, c in zip(ids.tolist(), counts.tolist()):
            if o in ranked:
                iou = c / (len(verts) + sizes[o] - c)
                if iou >= 0.5:
                    ranked[o].append((iou, pi))
    for o in ranked:
        ranked[o].sort(reverse=True)
    taken: set[int] = set()
    selected: dict[int, tuple[int, float]] = {}
    order = sorted(ranked, key=lambda o: -(ranked[o][0][0] if ranked[o] else 0))
    for o in order:
        for iou, pi in ranked[o]:
            if pi not in taken:
                taken.add(pi)
                selected[o] = (pi, iou)
                break
    return selected


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="replica_room_2")
    ap.add_argument("--bank-root", type=Path,
                    default=REPO_ROOT / "runs" / "phase8_c1p1")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "runs" / "phase8_c1p2")
    args = ap.parse_args(argv)
    scene, short = args.scene, SCENE_TO_SHORT[args.scene]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lock = json.loads((REPO_ROOT / "tools" / "replica_scenes.lock.json")
                      .read_text())
    room = Path(lock["data_root_relative_to_repo"]) / short
    bundle = REPO_ROOT / "runs" / "phase8_c1" / "bundles_ms02" / short
    key = json.loads((REPO_ROOT / "eval" / "questions" / "phase8"
                      / f"{scene}_qa.json").read_text())
    seg = load_segmentation_output(bundle)
    ev = evaluate(room, bundle)
    _, vidx, oid = _parse_semantic_ply(room / "habitat" / "mesh_semantic.ply")
    oracle = oracle_vertex_membership(vidx, oid, seg.n_vertices)
    entity_oids = [int(o) for o, cov in ev["oracle_coverage"].items()
                   if cov["class"] and cov["class"] not in STRUCTURAL_OR_DROPPED]

    proposals = pooled_proposals(bundle, args.bank_root, scene)
    selected = nominate(proposals, oracle, entity_oids)
    n_p1 = sum(1 for pi, _ in selected.values()
               if proposals[pi][0] == "p1")

    # materialize nominated proposals via the frozen resolver (IoU as the
    # declared deterministic priority — oracle-guided, as this stage is)
    sel = sorted(selected.items())
    masks = np.zeros((len(sel), seg.n_vertices), dtype=bool)
    scores = np.zeros(len(sel))
    for row, (o, (pi, iou)) in enumerate(sel):
        masks[row, proposals[pi][2]] = True
        scores[row] = iou
    ids = resolve_masks(masks, scores,
                        MaskResolveConfig(min_score=0.0, min_vertices=20))
    out = SegmentationOutput(
        input_mesh_sha256=seg.input_mesh_sha256,
        n_vertices=seg.n_vertices,
        segmenter_name="c1p2_ceiling_oracle_guided",
        segmenter_version="0.0-diagnostic",
        config_params_json=json.dumps({
            "provenance": "ORACLE-GUIDED pooled-bank ceiling (P2.0)",
            "n_nominated": len(sel), "n_from_p1": n_p1,
            "source_bundle_output_sha256": seg.output_sha256,
        }, sort_keys=True),
        vertex_instance_ids=ids,
    ).finalize()
    bdir = args.out_dir / f"ceiling_bundle_{scene}"
    save_segmentation_output(out, bdir)

    router = Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                    verbalizer=StandardVerbalizer())
    ctx = ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))

    def qa_row(bundle_dir: Path):
        e = evaluate(room, bundle_dir)
        arts, _ = build_c1_eval_bundle(room, bundle_dir, scene,
                                       min_vertices=20)
        uid_map = {f"obj_{m['pred_id']}": f"obj_{m['oracle_id']}"
                   for m in e["matches"]}
        qa, n_edges, _ = _qa_for_arts(arts, key, router, ctx, uid_map)
        n5 = e["n_entity_matches_at_iou"]["0.5"]
        return {
            "entity_matches_at_05": f"{n5}/{e['n_oracle_entity_instances']}",
            "micro_precision": (None if qa["micro_precision"] is None
                                else round(qa["micro_precision"], 4)),
            "micro_recall": (None if qa["micro_recall"] is None
                             else round(qa["micro_recall"], 4)),
            "support_hits": (qa["per_relation"].get("ON_ENTITY_SURFACE")
                             or {}).get("n_hit"),
            "n_graph_edges": n_edges,
            "per_relation": {r: {"n_hit": v["n_hit"],
                                 "n_expected": v["n_expected"],
                                 "n_cited": v["n_cited"]}
                             for r, v in qa["per_relation"].items()},
        }

    rows = {"C1_reference": qa_row(bundle), "P2_0_ceiling": qa_row(bdir)}
    anchor = C1_ANCHORS.get(scene)
    if anchor:
        got = rows["C1_reference"]
        if (got["micro_recall"] != anchor["micro_recall"]
                or got["support_hits"] != anchor["support"]):
            raise AssertionError(f"C1 anchor drift: {got} vs {anchor}")

    ceil = rows["P2_0_ceiling"]
    proceed = (ceil["micro_recall"] is not None
               and (ceil["micro_recall"] >= PROCEED_RECALL_MIN
                    or (ceil["support_hits"] or 0) >= PROCEED_SUPPORT_MIN))
    report = {
        "schema": "c1p2_ceiling_v1",
        "protocol": "docs/c1_p2_composer_protocol.md",
        "scene_id": scene,
        "purpose": ("oracle-guided pooled-bank QA ceiling; diagnostic only, "
                    "never deployable"),
        "n_pooled_proposals": len(proposals),
        "n_nominated": len(sel),
        "n_nominations_from_p1": n_p1,
        "ceiling_bundle_output_sha256": out.output_sha256,
        "rows": rows,
        "proceed_rule": {"recall_min": PROCEED_RECALL_MIN,
                         "support_min": PROCEED_SUPPORT_MIN},
        "decision": "PROCEED_TO_P2_1" if proceed else "STOP_P2",
    }
    p = args.out_dir / f"{scene}_p2_ceiling.json"
    p.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n",
                 encoding="utf-8")
    for name, r in rows.items():
        print(f"{name:>14}: ent@0.5={r['entity_matches_at_05']} "
              f"P={r['micro_precision']} R={r['micro_recall']} "
              f"support={r['support_hits']} edges={r['n_graph_edges']}")
    print(f"nominated {len(sel)} proposals ({n_p1} from P1)")
    print(f"DECISION: {report['decision']}")
    print(f"report -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

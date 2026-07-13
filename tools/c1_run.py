"""C1.04 — one-command local C1 run for a returned segmentation bundle.

    python3 tools/c1_run.py <room_dir> <segmentation_bundle_dir> <scene_id> \
        [--out-dir runs/phase8_c1]

Pipeline (all local, per docs/mesh_pipeline_contract.md):
  1. load + validate the bundle (hashes, lengths) and run the exact
     evaluator (C1.00) -> <scene>_exact_eval.json
  2. build the oracle-enriched C1 evaluation bundle (C1.03: labels via
     exact correspondence, variant A's surfaces verbatim)
  3. build A (JSON), B (semantic mesh), and C1 graphs with the SAME frozen
     battery configuration and diff battery answers pairwise
     -> <scene>_c1_run.json
  4. attribute each B<->C1 answer change to the predicted segment(s)
     responsible (via the pred<->oracle match table)

Honesty: every emitted report carries the isolation statement — C1 labels
and surfaces are INJECTED; only instance boundaries are learned. A<->B
differences are box-source effects (already characterized); B<->C1
differences are instance-extraction effects. No accuracy pass threshold is
applied (first runs measure the distribution).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.question_battery import STRUCTURAL_Q, SUPPORT_Q, _runs
from demo.replica_habitat_import import import_habitat_room
from demo.replica_mesh_import import import_mesh_room
from graph.builder import build_graph
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer
from segmenter.derived import build_c1_eval_bundle
from tools.c1_exact_eval import evaluate


def _answers(arts, router, ctx):
    bundle, _ = build_graph(arts, _runs(), density_policy="phase2_telemetry_only")
    return {q: sorted(router.answer(q, bundle, ctx).cited_uids)
            for q in STRUCTURAL_Q + SUPPORT_Q}


def _diff(name_a, ans_a, name_b, ans_b, uid_map=None):
    """Pairwise answer diff. uid_map translates side-b uids into side-a uid
    space (oracle space) before comparison; untranslatable uids stay as-is."""
    out = []
    for q in ans_a:
        a = set(ans_a[q])
        b = {uid_map.get(u, u) for u in ans_b[q]} if uid_map else set(ans_b[q])
        if a != b:
            out.append({"question": q,
                        f"only_{name_a}": sorted(a - b),
                        f"only_{name_b}": sorted(b - a)})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("room_dir", type=Path)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("scene_id")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "runs" / "phase8_c1")
    parser.add_argument("--min-vertices", type=int, default=20,
                        help="candidate min_vertices (contract default 20)")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        exact = evaluate(args.room_dir, args.bundle_dir)
        c1_arts, injection = build_c1_eval_bundle(
            args.room_dir, args.bundle_dir, args.scene_id,
            min_vertices=args.min_vertices)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[c1_run] HARD FAIL: {exc}")
        return 1

    (args.out_dir / f"{args.scene_id}_exact_eval.json").write_text(
        json.dumps(exact, indent=1), encoding="utf-8")

    A = import_habitat_room(args.room_dir, args.scene_id)
    B = import_mesh_room(args.room_dir, args.scene_id)
    router = Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                    verbalizer=StandardVerbalizer())
    ctx = ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))

    ans_a = _answers(A, router, ctx)
    ans_b = _answers(B, router, ctx)
    ans_c = _answers(c1_arts, router, ctx)

    # translate C1's pred-space uids into oracle uid space for the diff, and
    # keep the map in the report so each change names its predicted segment
    pred_to_oracle = {f"obj_{m['pred_id']}": f"obj_{m['oracle_id']}"
                      for m in exact["matches"]}
    oracle_to_pred = {v: k for k, v in pred_to_oracle.items()}
    coverage = exact["oracle_coverage"]

    def attribute_lost(oracle_uid: str) -> str:
        """Merge-aware attribution of a lost B answer, using full coverage —
        NOT just the greedy 1:1 map (an object absorbed into a merged segment
        is not 'missed')."""
        oid = oracle_uid.split("_", 1)[1]
        cov = coverage.get(oid)
        if cov is None:
            return f"{oracle_uid}: not-in-oracle-coverage"
        gp, frac, top = cov["greedy_pred"], cov["covered_frac"], cov["top_pred"]
        if gp is not None and cov["greedy_iou"] >= 0.5:
            return (f"{oracle_uid}: recovered-by pred obj_{gp} "
                    f"(iou {cov['greedy_iou']:.2f}) but relation changed")
        if top is not None and frac >= 0.5:
            owner = coverage_owner(top)
            if owner is not None and f"obj_{owner}" != oracle_uid:
                return (f"{oracle_uid}: merged-into pred obj_{top} "
                        f"(greedy-matched to obj_{owner}; {frac:.0%} covered)")
            return (f"{oracle_uid}: absorbed-by pred obj_{top} "
                    f"({frac:.0%} covered, low-IoU/unmatched)")
        return f"{oracle_uid}: no-viable-proposal ({frac:.0%} best coverage)"

    def coverage_owner(pred_id: int):
        """oracle id this prediction is greedy-matched to, if any"""
        for m in exact["matches"]:
            if m["pred_id"] == pred_id:
                return m["oracle_id"]
        return None

    def attribute_gained(uid: str) -> str:
        """Identify the prediction behind a gained C1 answer. `uid` is post-
        translation: matched preds appear as their ORACLE uid; unmatched
        preds keep their pred-space uid."""
        if uid in oracle_to_pred:
            m = next(m for m in exact["matches"]
                     if f"obj_{m['oracle_id']}" == uid)
            return (f"{uid}: via matched pred obj_{m['pred_id']} "
                    f"(iou {m['iou']:.2f}) — learned-box relation")
        return f"{uid}: unmatched anonymous segment"

    diffs_bc = _diff("B", ans_b, "C1", ans_c, uid_map=pred_to_oracle)
    for d in diffs_bc:
        d["lost_attribution"] = [attribute_lost(u) for u in d["only_B"]]
        d["gained_attribution"] = [attribute_gained(u) for u in d["only_C1"]]

    # B-relative per-question precision/recall (B = reference; C1 answers
    # translated into oracle uid space). None where the reference/answer set
    # is empty — an empty set has no rate, not a rate of 0 or 1.
    per_question_pr = {}
    tot_inter = tot_b = tot_c = 0
    sup_inter = sup_b = 0
    for q in ans_b:
        bset = set(ans_b[q])
        cset = {pred_to_oracle.get(u, u) for u in ans_c[q]}
        inter = len(bset & cset)
        per_question_pr[q] = {
            "precision_vs_B": (inter / len(cset)) if cset else None,
            "recall_vs_B": (inter / len(bset)) if bset else None,
            "n_B": len(bset), "n_C1": len(cset), "n_common": inter,
        }
        tot_inter += inter; tot_b += len(bset); tot_c += len(cset)
        if q in SUPPORT_Q:
            sup_inter += inter; sup_b += len(bset)
    answer_pr = {
        "micro_recall_vs_B": (tot_inter / tot_b) if tot_b else None,
        "micro_precision_vs_B": (tot_inter / tot_c) if tot_c else None,
        "support_answer_recall_vs_B": (sup_inter / sup_b) if sup_b else None,
        "n_support_answers_B": sup_b,
    }

    report = {
        "schema": "c1_run_v2",
        "scene_id": args.scene_id,
        "answer_pr_vs_B": answer_pr,
        "per_question_pr_vs_B": per_question_pr,
        "isolation_statement": c1_arts.notes["isolation_statement"],
        "exact_eval_summary": {k: exact[k] for k in (
            "n_pred_instances", "n_oracle_instances", "n_matched",
            "recall_at_iou", "n_entity_matches_at_iou", "support_owner",
            "over_segmentation", "under_segmentation")},
        "oracle_injection_summary": {k: injection[k] for k in (
            "n_injected_labels", "n_unmatched_kept_anonymous",
            "n_removed_structural_or_dropped")},
        "answer_diffs": {
            "A_vs_B_box_source": _diff("A", ans_a, "B", ans_b),
            "B_vs_C1_instance_extraction": diffs_bc,
        },
        "uid_map_pred_to_oracle": pred_to_oracle,
        "answers": {"A": ans_a, "B": ans_b, "C1": ans_c},
    }
    out = args.out_dir / f"{args.scene_id}_c1_run.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    n_ab = len(report["answer_diffs"]["A_vs_B_box_source"])
    n_bc = len(diffs_bc)
    ent5 = exact["n_entity_matches_at_iou"]["0.5"]
    n_ent = exact["n_oracle_entity_instances"]
    sup = answer_pr["support_answer_recall_vs_B"]
    mic = answer_pr["micro_recall_vs_B"]
    print(f"[c1_run] {args.scene_id}: entity-matches@0.5={ent5}/{n_ent}  "
          f"answer-recall-vs-B={'n/a' if mic is None else f'{mic:.2f}'}  "
          f"support-answer-recall={'n/a' if sup is None else f'{sup:.2f}'}  "
          f"A~B diffs={n_ab}  B~C1 diffs={n_bc}  "
          f"(labels+surfaces injected for isolation)")
    for d in diffs_bc:
        print(f"  B~C1 {d['question']!r}:")
        for a in d["lost_attribution"]:
            print(f"    lost  {a}")
        for a in d["gained_attribution"]:
            print(f"    gained {a}")
    print(f"[c1_run] report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

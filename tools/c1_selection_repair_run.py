"""C1-M2C rule runner — apply oracle-free selection repair, score all gates.

  python3 tools/c1_selection_repair_run.py \
      --room-dir DIR --bundle S3D_BUNDLE --control-bundle M3D_BUNDLE \
      --scene-id replica_room_2 --key eval/questions/phase8/replica_room_2_qa.json

Applies segmenter/selection_repair.py to the development bundle AND
unchanged to the control (Mask3D) bundle, materializes both through the
frozen resolver, runs the real downstream (exact eval -> derived bundle ->
graph -> Router), scores against the human key, and reports the six
PREDECLARED gates of docs/c1_m2c_protocol.md verbatim. The rule sees only
raw masks + scores; oracle data enters ONLY in scoring.
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
from segmenter.base import SegmentationOutput, load_segmentation_output, save_segmentation_output
from segmenter.derived import build_c1_eval_bundle
from segmenter.mask_resolve import MaskResolveConfig, resolve_masks
from segmenter.selection_repair import (
    SelectionRepairConfig, SelectionRepairV2Config, SelectionRepairV3Config,
    repair_selection, repair_selection_v2, repair_selection_v3,
)
from tools.c1_exact_eval import evaluate
from tools.c1_joint_ceiling import FROZEN_MIN_VERTICES, _qa_for_arts
from tools.c1_resolve_sweep import load_raw_masks

# Predeclared gates, frozen in docs/c1_m2c_protocol.md (dev scene room_2).
GATES = {
    "G1_entity_matches_at_05_min": 24,
    "G2_micro_precision_min": 0.85,
    "G3_micro_recall_min": 0.25,
    "G4_support_hits_min": 4,
    "G5_must_not_violations_max": 1,
    "G6_control_entity_min": 18,
    "G6_control_micro_precision_min": 0.95,
    "G6_control_micro_recall_min": 0.24,
}


def apply_rule_and_run(room_dir: Path, src_bundle: Path, scene_id: str,
                       key: dict | None, out_dir: Path, tag: str,
                       router, ctx, rule_version: str) -> dict:
    seg_src = load_segmentation_output(src_bundle)
    masks, scores = load_raw_masks(src_bundle)
    if rule_version == "v1":
        priorities, diag = repair_selection(masks, scores,
                                            SelectionRepairConfig())
        ids = resolve_masks(masks, priorities,
                            MaskResolveConfig(min_score=0.0,
                                              min_vertices=FROZEN_MIN_VERTICES))
    elif rule_version == "v2":
        ids, diag = repair_selection_v2(masks, scores,
                                        SelectionRepairV2Config(),
                                        min_vertices=FROZEN_MIN_VERTICES)
    elif rule_version == "v3":
        ids, diag = repair_selection_v3(masks, scores,
                                        SelectionRepairV3Config(),
                                        min_vertices=FROZEN_MIN_VERTICES)
    else:
        raise ValueError(f"unknown rule version {rule_version!r}")
    out = SegmentationOutput(
        input_mesh_sha256=seg_src.input_mesh_sha256,
        n_vertices=seg_src.n_vertices,
        segmenter_name=f"selection_repair_{rule_version}",
        segmenter_version=f"{rule_version}-oracle-free",
        config_params_json=json.dumps({
            "source_bundle_output_sha256": seg_src.output_sha256,
            "rule": diag["config"],
            "resolver": {"min_score": 0.0,
                         "min_vertices": FROZEN_MIN_VERTICES},
            "provenance": "ORACLE-FREE deployable rule (c1_m2c protocol)",
        }, sort_keys=True),
        vertex_instance_ids=ids,
    ).finalize()
    bundle_dir = out_dir / f"bundle_{tag}"
    save_segmentation_output(out, bundle_dir)

    ev = evaluate(room_dir, bundle_dir)
    arts, _ = build_c1_eval_bundle(room_dir, bundle_dir, scene_id,
                                   min_vertices=FROZEN_MIN_VERTICES)
    uid_map = {f"obj_{m['pred_id']}": f"obj_{m['oracle_id']}"
               for m in ev["matches"]}
    qa, n_edges, n_entities = _qa_for_arts(arts, key, router, ctx, uid_map)
    n5 = ev["n_entity_matches_at_iou"]["0.5"]
    return {
        "rule_diagnostics": diag,
        "entity_matches_at_05": n5,
        "n_oracle_entities": ev["n_oracle_entity_instances"],
        "support_owner_recall_at_05": ev["support_owner"]["recall_at_iou"]["0.5"],
        "n_pred_instances": ev["n_pred_instances"],
        "n_graph_edges": n_edges,
        "n_entities": n_entities,
        "qa_vs_human_key": qa,
    }


def _qa_bits(row):
    qa = row["qa_vs_human_key"] or {}
    sup = (qa.get("per_relation") or {}).get("ON_ENTITY_SURFACE") or {}
    viol = sum(len(q["must_not_violations"])
               for q in (qa.get("per_question") or {}).values())
    return qa.get("micro_precision"), qa.get("micro_recall"), \
        sup.get("n_hit"), viol


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--room-dir", required=True, type=Path)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--control-bundle", type=Path, default=None)
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--key", type=Path, default=None)
    ap.add_argument("--rule", default="v1", choices=("v1", "v2", "v3"))
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "runs" / "phase8_c1" / "selection_repair")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    key = json.loads(args.key.read_text()) if args.key else None
    if key is not None and key.get("answer_key_type") != "human_verified":
        raise ValueError(f"{args.key} is not a human_verified key")
    router = Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                    verbalizer=StandardVerbalizer())
    ctx = ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))

    dev = apply_rule_and_run(args.room_dir, args.bundle, args.scene_id,
                             key, args.out_dir, f"dev_{args.rule}",
                             router, ctx, args.rule)
    ctl = None
    if args.control_bundle:
        ctl = apply_rule_and_run(args.room_dir, args.control_bundle,
                                 args.scene_id, key, args.out_dir,
                                 f"control_{args.rule}", router, ctx, args.rule)

    p, r, sup, viol = _qa_bits(dev)
    gates = {
        "G1": dev["entity_matches_at_05"] >= GATES["G1_entity_matches_at_05_min"],
        "G2": p is not None and p >= GATES["G2_micro_precision_min"],
        "G3": r is not None and r >= GATES["G3_micro_recall_min"],
        "G4": sup is not None and sup >= GATES["G4_support_hits_min"],
        "G5": viol <= GATES["G5_must_not_violations_max"],
    }
    if ctl is not None:
        cp, cr, _, _ = _qa_bits(ctl)
        gates["G6"] = (
            ctl["entity_matches_at_05"] >= GATES["G6_control_entity_min"]
            and cp is not None and cp >= GATES["G6_control_micro_precision_min"]
            and cr is not None and cr >= GATES["G6_control_micro_recall_min"])

    report = {
        "schema": "c1_selection_repair_run_v1",
        "rule_version": args.rule,
        "protocol": "docs/c1_m2c_protocol.md (frozen 9953f04)",
        "scene_id": args.scene_id,
        "dev_bundle": str(args.bundle),
        "control_bundle": str(args.control_bundle) if args.control_bundle else None,
        "human_key": str(args.key) if args.key else None,
        "gate_thresholds": GATES,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "dev": dev,
        "control": ctl,
    }
    out = args.out_dir / f"{args.scene_id}_{args.rule}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    d = dev["rule_diagnostics"]
    if args.rule == "v1":
        print(f"[rule v1] base={d['n_base']} suppressed={d['n_suspects_suppressed']} "
              f"children_admitted={d['n_children_admitted']} "
              f"(below-0.2: {d['n_admitted_below_base_score']}) "
              f"active={d['n_final_active']}")
    else:
        mech = d.get("n_promoted_by_mechanism", "")
        print(f"[rule {args.rule}] base={d['n_base']} "
              f"promoted={d['n_promoted']} {mech} "
              f"(below-0.2: {d['n_promoted_below_base_score']}) "
              f"retention-suppressed={d['n_suppressed_low_retention']} "
              f"instances={d['n_final_instances']}")
    print(f"dev: ent@0.5={dev['entity_matches_at_05']}/{dev['n_oracle_entities']} "
          f"P={'-' if p is None else f'{p:.2f}'} R={'-' if r is None else f'{r:.2f}'} "
          f"support={sup} viol={viol} edges={dev['n_graph_edges']}")
    if ctl is not None:
        cp, cr, csup, cviol = _qa_bits(ctl)
        print(f"ctl: ent@0.5={ctl['entity_matches_at_05']}/{ctl['n_oracle_entities']} "
              f"P={'-' if cp is None else f'{cp:.2f}'} "
              f"R={'-' if cr is None else f'{cr:.2f}'} "
              f"support={csup} viol={cviol} edges={ctl['n_graph_edges']}")
    for g in sorted(gates):
        print(f"  {g}: {'PASS' if gates[g] else 'FAIL'}")
    print(f"ALL GATES: {'PASS' if report['all_gates_pass'] else 'FAIL'}")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

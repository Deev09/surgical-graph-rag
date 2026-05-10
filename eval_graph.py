"""Graph runner over the rich-schema benchmark.

Pipeline: parser -> candidate_filter -> top_k_prune. Same code path as
eval_scene.py / eval_paraphrase.py, just rewrapped to:
  - read benchmark/questions/<scene>.json (rich Question schema)
  - emit RunnerOutput per question (entity_ids + Evidence + latency)
  - compute ScoredResult via benchmark.runner.score_output (uniform across runners)

eval_scene.py is unchanged and remains the legacy-dict path. eval_graph.py is
the new-schema sibling that other runners (vlm, hybrid) will plug alongside.

Lenient top1/topk on the bathroom scene reproduces the v1 frozen baseline
(100/100/0.0) exactly because score_output reduces to v1 semantics when
ambiguity_policy=one_of and target_kind=entity.

Usage:
    python eval_graph.py \\
        --questions benchmark/questions/graffiti_bathroom.json \\
        --scene-graph baselines/v1/scene_graph.json \\
        --parser regex --scorer v1 \\
        --out-dir runs/graph/bathroom
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.runner import (
    Evidence,
    RunnerOutput,
    rollup,
    score_output,
    scored_result_to_dict,
)
from benchmark.schema import Question, load_questions
from parsers.dispatch import MODES, parse
from parsers.vocab import rebuild_label_vocab_from_scene
from scoring.filter import candidate_filter
from scoring.geom import _dist3, _find_anchor_node, _get_xyz
from scoring.topk import top_k_for_intent, top_k_prune
from scoring.v1 import score_node as score_v1
from scoring.v2 import score_node as score_v2
from tiny_graph_demo import scene_nodes_from_record


SCORERS = {"v1": score_v1, "v2": score_v2}


def _evidence_for(node: dict[str, Any], parsed, scene_nodes) -> Evidence:
    entity = str(node.get("label", ""))
    rel = parsed.spatial_relation
    anchor = parsed.relation_anchor
    if rel and anchor:
        anchor_node = _find_anchor_node(scene_nodes, anchor)
        relation_path = [{"from": entity, "type": rel, "to": anchor}]
        if anchor_node is not None:
            d = _dist3(_get_xyz(node), _get_xyz(anchor_node))
            return Evidence(
                entity_ids=[entity],
                relation_path=relation_path,
                distance_m=round(d, 3),
            )
        return Evidence(entity_ids=[entity], relation_path=relation_path)
    return Evidence(entity_ids=[entity])


def _run_one(q: Question, scene_nodes, parser_mode: str, score_fn) -> RunnerOutput:
    t0 = time.perf_counter()
    dr = parse(q.text, parser_mode)
    if dr.parsed is None:
        return RunnerOutput(
            question_id=q.question_id,
            abstained=False,
            answer_entity_ids=[],
            answer_text="",
            error="unparseable",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    parsed = dr.parsed
    k = top_k_for_intent(parsed)
    candidates = candidate_filter(scene_nodes, parsed, score_fn)
    pruned = top_k_prune(candidates, k=k)
    answer_ids = [str(p["label"]) for p in pruned]

    if pruned:
        evidence = _evidence_for(pruned[0], parsed, scene_nodes)
        if len(answer_ids) > 1:
            evidence.entity_ids = answer_ids[:]
    else:
        evidence = Evidence()

    return RunnerOutput(
        question_id=q.question_id,
        abstained=False,
        answer_entity_ids=answer_ids,
        answer_text=", ".join(answer_ids),
        evidence=evidence,
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _scene_objects_dict(scene_record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(o["label"]): o for o in scene_record.get("objects", [])}


def _print_summary(payload: dict[str, Any]) -> None:
    s = payload["summary"]
    cfg = payload["runner_config"]
    print(f"\n=== graph parser={cfg['parser']} scorer={cfg['scorer']} ===")
    print(f"  n={s['n_questions']}  top1={s['top1_accuracy']:.2%}  topk={s['topk_recall']:.2%}  "
          f"policy={s['policy_satisfied_rate']:.2%}  FP/q={s['avg_false_positives_per_query']:.2f}")
    print(f"  abstention: {s['abstention_outcome_counts']}")
    print(f"  failures:   {s['failure_attribution_counts']}")
    cats = s.get("per_category", {})
    if cats:
        rows = [
            f"{c}: top1={st['top1']:.0%}  topk={st['topk']:.0%}  policy={st['policy_satisfied']:.0%}  "
            f"FP={st['mean_fp']:.2f}  (n={st['n']})"
            for c, st in cats.items()
        ]
        print("  per-category:")
        for r in rows:
            print(f"    {r}")
    print(f"  latency mean={s['mean_latency_ms']:.1f}ms  max={s['max_latency_ms']:.1f}ms")


def main() -> None:
    ap = argparse.ArgumentParser(description="Graph runner over rich-schema questions.")
    ap.add_argument("--questions", required=True, type=Path, help="benchmark/questions/<scene>.json")
    ap.add_argument("--scene-graph", required=True, type=Path)
    ap.add_argument("--parser", choices=(*MODES, "all"), default="regex")
    ap.add_argument("--scorer", choices=("v1", "v2", "both"), default="v2")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    questions = load_questions(args.questions)
    scene_record = json.loads(args.scene_graph.read_text(encoding="utf-8"))
    scene_nodes = scene_nodes_from_record(scene_record)
    scene_objects = _scene_objects_dict(scene_record)
    rebuild_label_vocab_from_scene(scene_record)
    scene_id = questions[0].scene_id if questions else None

    parser_modes = MODES if args.parser == "all" else (args.parser,)
    scorer_ids = ("v1", "v2") if args.scorer == "both" else (args.scorer,)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_by_config: dict[str, dict[str, Any]] = {}
    for scorer_id in scorer_ids:
        score_fn = SCORERS[scorer_id]
        for mode in parser_modes:
            results = []
            for q in questions:
                out = _run_one(q, scene_nodes, mode, score_fn)
                scored = score_output(
                    q, out, scene_objects,
                    runner_name="graph",
                    runner_config={"parser": mode, "scorer": scorer_id},
                )
                results.append(scored)

            payload = {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "scene_id": scene_id,
                "runner_name": "graph",
                "runner_config": {"parser": mode, "scorer": scorer_id},
                "summary": rollup(results, questions),
                "results": [scored_result_to_dict(r) for r in results],
            }
            out_path = args.out_dir / f"eval_graph.{mode}.{scorer_id}.json"
            out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            _print_summary(payload)
            print(f"  -> {out_path}")
            summary_by_config[f"{mode}.{scorer_id}"] = payload["summary"]

    if len(summary_by_config) > 1:
        rollup_all = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "scene_id": scene_id,
            "runner_name": "graph",
            "configs": summary_by_config,
        }
        sum_path = args.out_dir / "eval_graph_summary.json"
        sum_path.write_text(json.dumps(rollup_all, indent=2) + "\n", encoding="utf-8")
        print(f"\nA/B summary: {sum_path}")


if __name__ == "__main__":
    main()

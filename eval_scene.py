"""Cross-scene benchmark harness.

Runs the existing parser/scorer/eval pipeline against an arbitrary scene,
not the bathroom global. Reuses tiny_graph_demo's candidate_filter,
top_k_prune, top_k_for_intent, node_matches_expectation, and the parser
modes from parsers.dispatch. Strict scoring via strict_eval.

Does NOT modify parser behavior, scoring, thresholds, or v1 artifacts.
The parser's vocabulary is whatever was snapshotted at module-import time
(bathroom labels). That is intentional — vocab gap is a real signal.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tiny_graph_demo import (
    node_matches_expectation,
    scene_nodes_from_record,
)

from scoring.filter import candidate_filter
from scoring.topk import top_k_for_intent, top_k_prune
from scoring.v1 import score_node as score_node_v1
from scoring.v2 import score_node as score_node_v2

from parsers.dispatch import MODES, parse
from parsers.vocab import rebuild_label_vocab_from_scene
from strict_eval import score_one_strict


SCORERS = {"v1": score_node_v1, "v2": score_node_v2}


def _attribute_failure(row: dict[str, Any]) -> str:
    """Coarse attribution: parser / relation_extraction / scorer / dense_graph / ambiguous / none."""
    if row["topk_ok"]:
        if row["top1_ok"]:
            return "none"
        return "scorer"  # right answer present in top-k but not top-1
    if row["parser_source"] == "unparseable":
        return "parser"
    if row["k"] is None:
        return "parser"
    if not row["pruned_labels"]:
        return "relation_extraction"  # parsed fine, no candidates survived
    if row["false_positives"] >= row["k"]:
        return "dense_graph"  # full pruned list is irrelevant
    return "scorer"


def _score_one(parsed, expected: list[str], scene_nodes, score_fn) -> dict[str, Any]:
    k = top_k_for_intent(parsed)
    candidates = candidate_filter(scene_nodes, parsed, score_fn)
    pruned = top_k_prune(candidates, k=k)
    labels_pruned = [str(p["label"]) for p in pruned]

    def matches_any(node: dict[str, Any]) -> bool:
        return any(node_matches_expectation(node, e) for e in expected)

    if expected == []:
        # True-negative case: success means returning nothing.
        top1_ok = not pruned
        topk_ok = not pruned
        fp = len(pruned)
    else:
        top1_ok = bool(pruned) and matches_any(pruned[0])
        topk_ok = any(matches_any(p) for p in pruned)
        fp = sum(1 for p in pruned if not matches_any(p))

    strict = score_one_strict(parsed, expected, pruned)
    if strict["hit_category"] == "pending_lenient":
        strict["hit_category"] = "metric_cushion" if top1_ok else "miss"

    return {
        "k": k,
        "top1_ok": top1_ok,
        "topk_ok": topk_ok,
        "false_positives": fp,
        "top1_label": labels_pruned[0] if labels_pruned else None,
        "topk_labels": labels_pruned,
        "pruned_labels": labels_pruned,
        **strict,
    }


def _row_for_unparseable(query: str, expected: list[str], dr, mode: str) -> dict[str, Any]:
    strict_none = score_one_strict(None, expected, [])
    return {
        "query": query,
        "expected": expected,
        "k": None,
        "top1_ok": (expected == []),  # vacuously OK only if expected was empty
        "topk_ok": (expected == []),
        "false_positives": 0,
        "top1_label": None,
        "topk_labels": [],
        "pruned_labels": [],
        **strict_none,
        "parsed_intent": None,
        "parsed_relation": None,
        "parsed_anchor": None,
        "parser_mode": mode,
        "parser_source": dr.source,
        "parser_error": dr.error,
        "parser_retry_count": dr.retry_count,
        "parser_latency_ms": dr.latency_ms,
        "failure_attribution": "parser",
    }


def _run_mode(
    mode: str,
    scene_nodes,
    expected_map: dict[str, list[str]],
    scene_id: str,
    score_fn,
    scorer_id: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for query, expected in expected_map.items():
        dr = parse(query, mode)
        if dr.parsed is None:
            row = _row_for_unparseable(query, expected, dr, mode)
            row["scorer"] = scorer_id
            rows.append(row)
            continue

        scored = _score_one(dr.parsed, expected, scene_nodes, score_fn)
        row = {
            "query": query,
            "expected": expected,
            **scored,
            "parsed_intent": dr.parsed.intent,
            "parsed_relation": dr.parsed.spatial_relation,
            "parsed_anchor": dr.parsed.relation_anchor,
            "parser_mode": mode,
            "parser_source": dr.source,
            "parser_error": dr.error,
            "parser_retry_count": dr.retry_count,
            "parser_latency_ms": dr.latency_ms,
            "scorer": scorer_id,
        }
        row["failure_attribution"] = _attribute_failure(row)
        rows.append(row)

    n = len(rows)
    parsed_rows = [r for r in rows if r["parsed_intent"] is not None]
    n_unparseable = n - len(parsed_rows)
    latencies = [r["parser_latency_ms"] for r in rows]
    retries = [r["parser_retry_count"] for r in rows]
    source_counts = Counter(r["parser_source"] for r in rows)
    hit_cat_counts = Counter(r["hit_category"] for r in rows)
    answer_type_counts = Counter(r["answer_type"] for r in rows)
    failure_counts = Counter(r["failure_attribution"] for r in rows)

    real_hits = sum(1 for r in rows if r["strict_top1_ok"] and r["intent_congruent"])
    strict_topk_hits = sum(1 for r in rows if r["strict_topk_ok"] and r["intent_congruent"])
    intent_congruent = sum(1 for r in rows if r["intent_congruent"])

    summary = {
        "n_queries": n,
        "top1_accuracy": sum(1 for r in rows if r["top1_ok"]) / n if n else 0.0,
        "topk_recall": sum(1 for r in rows if r["topk_ok"]) / n if n else 0.0,
        "avg_false_positives_per_query": (sum(r["false_positives"] for r in rows) / n if n else 0.0),
        "strict_top1_accuracy": real_hits / n if n else 0.0,
        "strict_topk_recall": strict_topk_hits / n if n else 0.0,
        "intent_congruent_rate": intent_congruent / n if n else 0.0,
        "hit_category_counts": dict(hit_cat_counts),
        "answer_type_counts": dict(answer_type_counts),
        "failure_attribution_counts": dict(failure_counts),
        "unparseable_count": n_unparseable,
        "parser_source_breakdown": dict(source_counts),
        "mean_retry_count": statistics.mean(retries) if retries else 0.0,
        "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency_ms": (
            statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies, default=0.0)
        ),
    }

    return {
        "scene_id": scene_id,
        "parser_mode": mode,
        "scorer": scorer_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "rows": rows,
    }


def _print_summary(payload: dict[str, Any]) -> None:
    s = payload["summary"]
    mode = payload["parser_mode"]
    scorer = payload.get("scorer", "v1")
    print(f"\n=== scorer={scorer}  parser={mode} ===")
    print(f"  n={s['n_queries']}  unparseable={s['unparseable_count']}")
    print(f"  lenient top-1: {s['top1_accuracy']:.2%}   topk: {s['topk_recall']:.2%}   FP@k: {s['avg_false_positives_per_query']:.2f}")
    print(f"  strict  top-1: {s['strict_top1_accuracy']:.2%}   topk: {s['strict_topk_recall']:.2%}   intent-congruent: {s['intent_congruent_rate']:.2%}")
    print(f"  hit categories: {s['hit_category_counts']}")
    print(f"  failure attributions: {s['failure_attribution_counts']}")
    print(f"  parser source: {s['parser_source_breakdown']}")
    print(f"  retry mean={s['mean_retry_count']:.2f}  latency mean={s['mean_latency_ms']:.1f}ms  p95={s['p95_latency_ms']:.1f}ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-graph", required=True, type=Path)
    ap.add_argument("--expected-answers", required=True, type=Path)
    ap.add_argument("--parser", choices=(*MODES, "all"), default="all")
    ap.add_argument("--scorer", choices=("v1", "v2", "both"), default="v1")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--scene-id", default=None, help="Override scene_id label in output (defaults to scene_graph['scene'])")
    args = ap.parse_args()

    scene_record = json.loads(args.scene_graph.read_text(encoding="utf-8"))
    scene_nodes = scene_nodes_from_record(scene_record)
    scene_id = args.scene_id or scene_record.get("scene", args.scene_graph.stem)
    expected_map = json.loads(args.expected_answers.read_text(encoding="utf-8"))
    rebuild_label_vocab_from_scene(scene_record)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    modes = MODES if args.parser == "all" else (args.parser,)
    scorer_ids = ("v1", "v2") if args.scorer == "both" else (args.scorer,)
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    for scorer_id in scorer_ids:
        score_fn = SCORERS[scorer_id]
        for m in modes:
            payload = _run_mode(m, scene_nodes, expected_map, scene_id, score_fn, scorer_id)
            suffix = f".{m}" if len(scorer_ids) == 1 else f".{scorer_id}.{m}"
            out_path = args.out_dir / f"evaluation_table{suffix}.json"
            out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            _print_summary(payload)
            print(f"  -> {out_path}")
            payloads[(scorer_id, m)] = payload

    if args.parser == "all" or args.scorer == "both":
        rollup = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "scene_id": scene_id,
            "scene_graph": str(args.scene_graph),
            "expected_answers": str(args.expected_answers),
            "scorers": list(scorer_ids),
            "modes": {
                f"{s}.{m}": {"summary": p["summary"]} for (s, m), p in payloads.items()
            },
        }
        suffix = "" if len(scorer_ids) == 1 else "_ab"
        summary_path = args.out_dir / f"scene_eval_summary{suffix}.json"
        summary_path.write_text(json.dumps(rollup, indent=2) + "\n", encoding="utf-8")
        print(f"\nA/B summary: {summary_path}")


if __name__ == "__main__":
    main()

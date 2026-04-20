"""Paraphrase stress test for the v1 pruning pipeline, parser A/B-aware.

Runs the same downstream semantics used by tiny_graph_demo.run_benchmark
(candidate_filter -> top_k_prune -> node_matches_expectation) but lets the
caller choose which parser feeds it: v1 regex, LLM, LLM-with-regex-fallback,
or strict LLM. Outputs a per-mode evaluation table + an optional summary
across all modes.

v1 artifacts in baselines/v1/ are never touched.
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
    candidate_filter,
    node_matches_expectation,
    parse_query,
    scene,
    top_k_for_intent,
    top_k_prune,
)

from parsers.dispatch import MODES, parse
from strict_eval import resolve_answer_type, score_one_strict


ROOT = Path(__file__).resolve().parent
IN_PATH = ROOT / "baselines" / "v1_paraphrase" / "paraphrased_queries.json"
OUT_DIR = ROOT / "baselines" / "v1_paraphrase"
LEGACY_REGEX_PATH = OUT_DIR / "evaluation_table.json"


def _out_path(mode: str) -> Path:
    return OUT_DIR / f"evaluation_table.{mode}.json"


def _load_records() -> list[dict[str, Any]]:
    records = json.loads(IN_PATH.read_text(encoding="utf-8"))
    queries = [r["query"] for r in records]
    if len(set(queries)) != len(queries):
        raise ValueError("duplicate paraphrase strings in input")
    return records


def _score_one(parsed, expected: list[str]) -> dict[str, Any]:
    """Run the same inner pipeline run_benchmark uses; emit both lenient and strict fields."""
    k = top_k_for_intent(parsed)
    candidates = candidate_filter(scene, parsed)
    pruned = top_k_prune(candidates, k=k)
    labels_pruned = [str(p["label"]) for p in pruned]

    def matches_any(node: dict[str, Any]) -> bool:
        return any(node_matches_expectation(node, e) for e in expected)

    top1_ok = bool(pruned) and matches_any(pruned[0])
    topk_ok = any(matches_any(p) for p in pruned)
    fp = sum(1 for p in pruned if not matches_any(p))

    strict = score_one_strict(parsed, expected, pruned)
    # Resolve pending_lenient using the lenient top1_ok.
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


def _run_mode(mode: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        q = rec["query"]
        src = rec["source_query"]
        expected = rec["expected"]

        dr = parse(q, mode)
        p_src = parse_query(src)  # source-intent baseline for drift

        if dr.parsed is None:
            # Unparseable — explicit, no synthetic fallback. Counts as a miss.
            strict_none = score_one_strict(None, expected, [])
            rows.append(
                {
                    "query": q,
                    "source_query": src,
                    "expected": expected,
                    "k": None,
                    "top1_ok": False,
                    "topk_ok": False,
                    "false_positives": 0,
                    "top1_label": None,
                    "topk_labels": [],
                    "pruned_labels": [],
                    **strict_none,
                    "source_intent": p_src.intent,
                    "paraphrase_intent": None,
                    "paraphrase_relation": None,
                    "paraphrase_anchor": None,
                    "paraphrase_zone_family": None,
                    "parser_drift": None,
                    "parser_mode": mode,
                    "parser_source": dr.source,
                    "parser_error": dr.error,
                    "parser_retry_count": dr.retry_count,
                    "parser_latency_ms": dr.latency_ms,
                }
            )
            continue

        scored = _score_one(dr.parsed, expected)
        drift = dr.parsed.intent != p_src.intent
        rows.append(
            {
                "query": q,
                "source_query": src,
                "expected": expected,
                **scored,
                "source_intent": p_src.intent,
                "paraphrase_intent": dr.parsed.intent,
                "paraphrase_relation": dr.parsed.spatial_relation,
                "paraphrase_anchor": dr.parsed.relation_anchor,
                "paraphrase_zone_family": dr.parsed.zone_family,
                "parser_drift": drift,
                "parser_mode": mode,
                "parser_source": dr.source,
                "parser_error": dr.error,
                "parser_retry_count": dr.retry_count,
                "parser_latency_ms": dr.latency_ms,
            }
        )

    n = len(rows)
    parsed_rows = [r for r in rows if r["paraphrase_intent"] is not None]
    n_parsed = len(parsed_rows)
    n_unparseable = n - n_parsed
    drift_count = sum(1 for r in parsed_rows if r["parser_drift"])

    latencies = [r["parser_latency_ms"] for r in rows]
    retries = [r["parser_retry_count"] for r in rows]
    source_counts = Counter(r["parser_source"] for r in rows)
    hit_cat_counts = Counter(r["hit_category"] for r in rows)
    answer_type_counts = Counter(r["answer_type"] for r in rows)

    real_hits = sum(
        1 for r in rows if r["strict_top1_ok"] and r["intent_congruent"]
    )
    strict_topk_hits = sum(
        1 for r in rows if r["strict_topk_ok"] and r["intent_congruent"]
    )
    intent_congruent = sum(1 for r in rows if r["intent_congruent"])

    summary = {
        "n_queries": n,
        "top1_accuracy": sum(1 for r in rows if r["top1_ok"]) / n if n else 0.0,
        "topk_recall": sum(1 for r in rows if r["topk_ok"]) / n if n else 0.0,
        "avg_false_positives_per_query": (
            sum(r["false_positives"] for r in rows) / n if n else 0.0
        ),
        "strict_top1_accuracy": real_hits / n if n else 0.0,
        "strict_topk_recall": strict_topk_hits / n if n else 0.0,
        "intent_congruent_rate": intent_congruent / n if n else 0.0,
        "hit_category_counts": dict(hit_cat_counts),
        "answer_type_counts": dict(answer_type_counts),
        "unparseable_count": n_unparseable,
        "parser_drift_count_over_parsed": drift_count,
        "parser_drift_rate_over_parsed": drift_count / n_parsed if n_parsed else 0.0,
        "parser_source_breakdown": dict(source_counts),
        "mean_retry_count": statistics.mean(retries) if retries else 0.0,
        "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency_ms": (
            statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies, default=0.0)
        ),
    }

    return {
        "baseline_id": "v1_paraphrase",
        "source_baseline": "v1",
        "parser_mode": mode,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "rows": rows,
    }


def _write_payload(payload: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _out_path(payload["parser_mode"])
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _regex_parity_check(regex_payload: dict[str, Any]) -> list[str]:
    """Compare regex-mode rows to the existing un-suffixed evaluation_table.json.

    Returns a list of discrepancy descriptions (empty = identical behavior).
    Fields compared: top1_ok, topk_ok, false_positives, pruned_labels, paraphrase_intent.
    """
    if not LEGACY_REGEX_PATH.exists():
        return ["legacy evaluation_table.json not found; skipping parity check"]

    legacy = json.loads(LEGACY_REGEX_PATH.read_text(encoding="utf-8"))
    legacy_by_q = {r["query"]: r for r in legacy["rows"]}
    new_by_q = {r["query"]: r for r in regex_payload["rows"]}

    discrepancies: list[str] = []
    fields = ("top1_ok", "topk_ok", "false_positives", "pruned_labels", "paraphrase_intent")
    for q, new_row in new_by_q.items():
        old_row = legacy_by_q.get(q)
        if old_row is None:
            discrepancies.append(f"{q!r}: not in legacy table")
            continue
        for f in fields:
            if old_row.get(f) != new_row.get(f):
                discrepancies.append(
                    f"{q!r}: field {f} changed: legacy={old_row.get(f)!r} new={new_row.get(f)!r}"
                )
    return discrepancies


def _print_mode_summary(payload: dict[str, Any]) -> None:
    s = payload["summary"]
    mode = payload["parser_mode"]
    print(f"\n=== mode: {mode} ===")
    print(f"  n={s['n_queries']}  unparseable={s['unparseable_count']}")
    print(f"  lenient top-1: {s['top1_accuracy']:.2%}   topk: {s['topk_recall']:.2%}   FP@k: {s['avg_false_positives_per_query']:.2f}")
    print(f"  strict  top-1: {s['strict_top1_accuracy']:.2%}   topk: {s['strict_topk_recall']:.2%}   intent-congruent: {s['intent_congruent_rate']:.2%}")
    print(f"  hit categories: {s['hit_category_counts']}")
    print(
        f"  parser drift (over parsed): "
        f"{s['parser_drift_count_over_parsed']}/{s['n_queries'] - s['unparseable_count']} "
        f"({s['parser_drift_rate_over_parsed']:.2%})"
    )
    print(f"  parser source: {s['parser_source_breakdown']}")
    print(f"  retry mean={s['mean_retry_count']:.2f}  "
          f"latency mean={s['mean_latency_ms']:.1f}ms  p95={s['p95_latency_ms']:.1f}ms")


def _write_ab_summary(payloads: dict[str, dict[str, Any]]) -> Path:
    rollup = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_baseline": "v1",
        "paraphrase_set": str(IN_PATH.name),
        "modes": {
            m: {
                "output_file": _out_path(m).name,
                "summary": p["summary"],
            }
            for m, p in payloads.items()
        },
    }
    path = OUT_DIR / "parser_ab_summary.json"
    path.write_text(json.dumps(rollup, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parser",
        choices=(*MODES, "all"),
        default="regex",
        help="Which parser to run. 'all' runs every mode sequentially and writes an A/B summary.",
    )
    args = ap.parse_args()

    records = _load_records()

    if args.parser == "all":
        payloads: dict[str, dict[str, Any]] = {}
        for mode in MODES:
            payload = _run_mode(mode, records)
            path = _write_payload(payload)
            payloads[mode] = payload
            _print_mode_summary(payload)
            print(f"  -> {path}")
            if mode == "regex":
                discrepancies = _regex_parity_check(payload)
                if discrepancies:
                    print("  REGEX PARITY WARNING — A/B numbers may not be comparable:")
                    for d in discrepancies:
                        print(f"    - {d}")
                else:
                    print("  regex parity OK (identical to legacy evaluation_table.json)")
        summary_path = _write_ab_summary(payloads)
        print(f"\nA/B summary: {summary_path}")
        return

    payload = _run_mode(args.parser, records)
    path = _write_payload(payload)
    _print_mode_summary(payload)
    print(f"\nReport: {path}")

    if args.parser == "regex":
        discrepancies = _regex_parity_check(payload)
        if discrepancies:
            print("\nREGEX PARITY WARNING — A/B numbers may not be comparable:")
            for d in discrepancies:
                print(f"  - {d}")
        else:
            print("\nregex parity OK (identical to legacy evaluation_table.json)")


if __name__ == "__main__":
    main()

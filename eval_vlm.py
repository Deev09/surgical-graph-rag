"""VLM runner: question + scene photos + canonical-id list -> structured answer.

Plugs into the same RunnerOutput / score_output contract as eval_graph.py, so
the harness can compare graph and VLM head-to-head on the same Question +
expected_targets.

Pipeline per question:
  1. Build prompt payload (system + user + entity list + tool schema)
  2. Compute cache key over (model_id, prompt_template_version, question_id,
     image_sha256s, prompt_payload)
  3. Hit DiskCache; on miss, call VLM and store
  4. Convert VLM response to RunnerOutput; run score_output against the Question

Modes:
  --mock      MockVLMClient. Always abstains. No API key, no images required.
              Used to validate the harness without spending money.
  default     AnthropicVLMClient. Requires ANTHROPIC_API_KEY and a populated
              views.json manifest pointing at real images.

The VLM never sees the expected answer or any expected_targets metadata. It
only sees:
  - the question text
  - the canonical-id list of every object in the scene (with display labels)
  - the answer_type and category
  - the photos
That's how we test perception, not test-leak.

Usage:
    python eval_vlm.py \\
        --questions benchmark/questions/replica_room_0.json \\
        --scene-graph scenes/replica_room_0/scene_graph.json \\
        --views scenes/replica_room_0/views.json \\
        --out-dir runs/vlm/replica
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.cache import CacheKey, DiskCache, hash_file
from benchmark.runner import (
    Evidence,
    RunnerOutput,
    rollup,
    score_output,
    scored_result_to_dict,
)
from benchmark.schema import Question, load_questions
from benchmark.views import SceneViews, Viewpoint, load_views
from benchmark.vlm_client import (
    DEFAULT_MODEL_ID,
    PROMPT_TEMPLATE_VERSION,
    AnthropicVLMClient,
    BaseVLMClient,
    MockVLMClient,
    VLMResponse,
)


def _scene_objects_dict(scene_record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(o["label"]): o for o in scene_record.get("objects", [])}


def _vlm_response_to_runner_output(
    q: Question,
    resp: VLMResponse,
    viewpoints: list[Viewpoint],
) -> RunnerOutput:
    if resp.error:
        return RunnerOutput(
            question_id=q.question_id,
            abstained=False,
            answer_entity_ids=[],
            answer_text="",
            error=resp.error,
            latency_ms=resp.latency_ms,
        )

    out = resp.output or {}
    answer_ids = list(out.get("answer_entity_ids") or [])
    abstained = bool(out.get("abstained", False))
    answer_text = str(out.get("answer_text", ""))
    ev_in = out.get("evidence") or {}

    src_idx = ev_in.get("source_frame_idx")
    if src_idx is not None and (not isinstance(src_idx, int) or src_idx < 0 or src_idx >= len(viewpoints)):
        src_idx = None

    crop = ev_in.get("crop_bbox")
    if crop is not None and (not isinstance(crop, list) or len(crop) != 4):
        crop = None

    confidence = ev_in.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None

    evidence = Evidence(
        entity_ids=list(ev_in.get("entity_ids") or answer_ids),
        relation_path=[],
        distance_m=None,
        source_frame_idx=src_idx,
        crop_bbox=crop,
    )

    return RunnerOutput(
        question_id=q.question_id,
        abstained=abstained,
        answer_entity_ids=answer_ids,
        answer_text=answer_text,
        confidence=confidence,
        evidence=evidence,
        latency_ms=resp.latency_ms,
        error=None,
    )


def _run_one(
    q: Question,
    scene_objects_list: list[dict[str, Any]],
    viewpoints: list[Viewpoint],
    client: BaseVLMClient,
    cache: DiskCache,
    image_sha256s: list[str],
    cache_disabled: bool,
) -> tuple[RunnerOutput, dict[str, Any] | None]:
    payload = client.build_prompt_payload(
        question_text=q.text,
        category=q.category,
        answer_type=q.answer_type,
        scene_objects=scene_objects_list,
    )
    key = CacheKey(
        question_id=q.question_id,
        model_id=client.model_id,
        prompt_template_version=client.prompt_template_version,
        image_sha256s=image_sha256s,
        prompt_payload=payload,
    )
    cached = None if cache_disabled else cache.get(key)
    if cached is not None:
        resp = VLMResponse(
            output=cached.get("output", {}),
            raw_response=cached.get("raw_response", ""),
            latency_ms=float(cached.get("latency_ms", 0.0)),
            tokens=cached.get("tokens"),
            error=cached.get("error"),
        )
        cache_status = "hit"
    else:
        resp = client.call(
            question_text=q.text,
            category=q.category,
            answer_type=q.answer_type,
            scene_objects=scene_objects_list,
            image_paths=[v.path for v in viewpoints],
        )
        cache.put(key, {
            "output": resp.output,
            "raw_response": resp.raw_response,
            "latency_ms": resp.latency_ms,
            "tokens": resp.tokens,
            "error": resp.error,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        })
        cache_status = "miss"

    runner_output = _vlm_response_to_runner_output(q, resp, viewpoints)
    diag = {
        "cache_status": cache_status,
        "tokens": resp.tokens,
    }
    return runner_output, diag


def _print_summary(payload: dict[str, Any]) -> None:
    s = payload["summary"]
    cfg = payload["runner_config"]
    print(f"\n=== vlm model={cfg['model_id']} mock={cfg.get('mock', False)} ===")
    print(f"  n={s['n_questions']}  top1={s['top1_accuracy']:.2%}  topk={s['topk_recall']:.2%}  "
          f"policy={s['policy_satisfied_rate']:.2%}  FP/q={s['avg_false_positives_per_query']:.2f}")
    print(f"  abstention: {s['abstention_outcome_counts']}")
    print(f"  failures:   {s['failure_attribution_counts']}")
    cats = s.get("per_category", {})
    if cats:
        print("  per-category:")
        for c, st in cats.items():
            print(f"    {c}: top1={st['top1']:.0%}  topk={st['topk']:.0%}  policy={st['policy_satisfied']:.0%}  "
                  f"FP={st['mean_fp']:.2f}  (n={st['n']})")
    print(f"  latency mean={s['mean_latency_ms']:.1f}ms  max={s['max_latency_ms']:.1f}ms")


def main() -> None:
    ap = argparse.ArgumentParser(description="VLM runner over rich-schema questions.")
    ap.add_argument("--questions", required=True, type=Path)
    ap.add_argument("--scene-graph", required=True, type=Path)
    ap.add_argument("--views", type=Path, default=None,
                    help="Path to views.json. Required unless --mock.")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL_ID)
    ap.add_argument("--mock", action="store_true", help="Use MockVLMClient; skip API + image loading.")
    ap.add_argument("--no-cache", action="store_true", help="Disable disk cache.")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    scene_record = json.loads(args.scene_graph.read_text(encoding="utf-8"))
    scene_objects_dict = _scene_objects_dict(scene_record)
    scene_objects_list = list(scene_record.get("objects", []))
    scene_id = questions[0].scene_id if questions else None

    if args.mock:
        client: BaseVLMClient = MockVLMClient()
        viewpoints: list[Viewpoint] = []
        image_sha256s: list[str] = []
    else:
        if args.views is None:
            raise SystemExit("--views <path/to/views.json> is required (or pass --mock).")
        scene_views: SceneViews = load_views(args.views)
        viewpoints = scene_views.viewpoints
        image_sha256s = [hash_file(v.path) for v in viewpoints]
        client = AnthropicVLMClient(model_id=args.model)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = DiskCache(args.out_dir / "cache")

    results = []
    diags = []
    t0_run = time.perf_counter()
    for q in questions:
        out, diag = _run_one(
            q=q,
            scene_objects_list=scene_objects_list,
            viewpoints=viewpoints,
            client=client,
            cache=cache,
            image_sha256s=image_sha256s,
            cache_disabled=args.no_cache,
        )
        scored = score_output(
            q, out, scene_objects_dict,
            runner_name="vlm",
            runner_config={
                "model_id": client.model_id,
                "prompt_template_version": client.prompt_template_version,
                "mock": args.mock,
                "n_viewpoints": len(viewpoints),
            },
        )
        results.append(scored)
        diags.append({"question_id": q.question_id, **diag})

    total_latency = (time.perf_counter() - t0_run) * 1000.0
    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "scene_id": scene_id,
        "runner_name": "vlm",
        "runner_config": {
            "model_id": client.model_id,
            "prompt_template_version": client.prompt_template_version,
            "mock": args.mock,
            "n_viewpoints": len(viewpoints),
            "viewpoints": [{"id": v.id, "note": v.note} for v in viewpoints],
        },
        "summary": rollup(results, questions),
        "results": [scored_result_to_dict(r) for r in results],
        "diagnostics": diags,
        "total_wall_ms": total_latency,
    }
    cache_hits = sum(1 for d in diags if d.get("cache_status") == "hit")
    payload["summary"]["cache_hits"] = cache_hits
    payload["summary"]["cache_misses"] = len(diags) - cache_hits

    out_path = args.out_dir / f"eval_vlm.{client.model_id.replace('/', '_')}{'.mock' if args.mock else ''}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _print_summary(payload)
    print(f"  cache: {cache_hits} hits, {len(diags) - cache_hits} misses")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()

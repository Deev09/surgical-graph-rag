"""Compare v0.1 vs v0.2 runner semantics on bathroom + Replica fixtures.

P1.11.post one-shot script.

What this does:
  1. Promotes the frozen v0.1 question fixtures to v0.2 (writes COPIES under
     runs/p1_11_post/questions/; the original benchmark/questions/*.json
     files are not touched).
  2. Re-scores the preserved RunnerOutputs from prior eval_graph runs using
     the P1.11-fixed score_output. The model outputs are NOT regenerated —
     we only re-apply scoring. Parser, scorer, top-k, thresholds, graph,
     extractor, and ranking are all unchanged.
  3. Emits a comparison artifact with old vs new summaries, per-question
     deltas, and attribution (any_of_subset / scalar / schema_promotion /
     other) for each changed outcome.
  4. Records change_type="benchmark_semantics_change" in the artifact.

What this does NOT do:
  - Modify benchmark/questions/*.json (the frozen v0.1 files).
  - Modify runs/graph/* (the prior eval artifacts).
  - Re-execute parser, scorer, or any graph behavior. No model change.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.runner import (
    Evidence, RunnerOutput, rollup, score_output, scored_result_to_dict,
)
from benchmark.schema import (
    SCHEMA_VERSION, Question, load_questions, save_questions,
)


PRIOR_RUNS = [
    {
        "scene_id": "graffiti_bathroom",
        "fixture_v01": REPO_ROOT / "benchmark/questions/graffiti_bathroom.json",
        "scene_graph": REPO_ROOT / "baselines/v1/scene_graph.json",
        "prior_artifacts": [
            REPO_ROOT / "runs/graph/bathroom/eval_graph.regex.v1.json",
        ],
    },
    {
        "scene_id": "replica_room_0",
        "fixture_v01": REPO_ROOT / "benchmark/questions/replica_room_0.json",
        "scene_graph": REPO_ROOT / "scenes/replica_room_0/scene_graph.json",
        "prior_artifacts": [
            REPO_ROOT / "runs/graph/replica/eval_graph.regex.v1.json",
            REPO_ROOT / "runs/graph/replica/eval_graph.regex.v2.json",
        ],
    },
]


def promote_v01_to_v02(src: Path, dst: Path) -> None:
    """Load v0.1, save as v0.2 (the loader does the promotion in-memory;
    save writes the current SCHEMA_VERSION)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        questions = load_questions(src)
    scene_id = questions[0].scene_id if questions else "unknown"
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_questions(scene_id, questions, dst)


def _reconstruct_runner_output(d: dict[str, Any]) -> RunnerOutput:
    ev = d.get("evidence", {}) or {}
    return RunnerOutput(
        question_id=d["question_id"],
        abstained=bool(d.get("abstained", False)),
        answer_entity_ids=list(d.get("answer_entity_ids", [])),
        answer_text=str(d.get("answer_text", "")),
        answer_count=d.get("answer_count"),
        answer_yes_no=d.get("answer_yes_no"),
        confidence=d.get("confidence"),
        evidence=Evidence(
            entity_ids=list(ev.get("entity_ids", [])),
            relation_path=[dict(r) for r in ev.get("relation_path", [])],
            distance_m=ev.get("distance_m"),
            source_frame_idx=ev.get("source_frame_idx"),
            crop_bbox=list(ev.get("crop_bbox")) if ev.get("crop_bbox") is not None else None,
        ),
        latency_ms=float(d.get("latency_ms", 0.0)),
        error=d.get("error"),
    )


def _scene_objects(scene_graph_path: Path) -> dict[str, dict[str, Any]]:
    record = json.loads(scene_graph_path.read_text(encoding="utf-8"))
    return {str(o["label"]): o for o in record.get("objects", [])}


def _attribute_change(
    q: Question,
    old_scored: dict[str, Any],
    new_scored_dict: dict[str, Any],
) -> list[str]:
    """Attribute which P1.11 change caused each delta. Returns a list of
    tags (possibly empty if no flag changed for this question)."""
    flags = ("top1_correct", "topk_correct", "policy_satisfied",
             "false_positives", "abstention_outcome", "failure_attribution")
    changed_flags = [
        f for f in flags if old_scored.get(f) != new_scored_dict.get(f)
    ]
    if not changed_flags:
        return []
    tags: list[str] = []
    if q.answer_type in ("count", "yes_no"):
        tags.append("scalar_scoring")
    if "policy_satisfied" in changed_flags and q.ambiguity_policy == "any_of_subset":
        tags.append("any_of_subset_fix")
    if not tags:
        tags.append("other")
    return tags


def compare_one(
    *,
    scene_id: str,
    questions_v02: list[Question],
    scene_objects: dict[str, dict[str, Any]],
    prior_path: Path,
) -> dict[str, Any]:
    """Re-score one prior artifact under v0.2 semantics. Returns a per-
    artifact comparison dict."""
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    runner_name = prior.get("runner_name", "graph")
    runner_config = prior.get("runner_config", {})
    prior_results = prior.get("results", [])
    old_summary = prior.get("summary", {})

    q_by_id = {q.question_id: q for q in questions_v02}

    rescored_results = []
    per_question_diffs: list[dict[str, Any]] = []
    n_unchanged = 0
    for old in prior_results:
        qid = old["question_id"]
        q = q_by_id.get(qid)
        if q is None:
            raise ValueError(
                f"question {qid!r} in {prior_path} not found in v0.2 fixture"
            )
        output = _reconstruct_runner_output(old["output"])
        new_scored = score_output(
            q, output, scene_objects,
            runner_name=runner_name, runner_config=runner_config,
        )
        new_dict = scored_result_to_dict(new_scored)
        rescored_results.append(new_scored)
        tags = _attribute_change(q, old, new_dict)
        flags = ("top1_correct", "topk_correct", "policy_satisfied",
                 "false_positives", "abstention_outcome", "failure_attribution")
        deltas = {
            f: {"old": old.get(f), "new": new_dict.get(f)}
            for f in flags if old.get(f) != new_dict.get(f)
        }
        if deltas:
            per_question_diffs.append({
                "question_id": qid,
                "question_text": q.text,
                "answer_type": q.answer_type,
                "ambiguity_policy": q.ambiguity_policy,
                "attribution": tags,
                "deltas": deltas,
            })
        else:
            n_unchanged += 1

    new_summary = rollup(rescored_results, questions_v02)
    summary_deltas = {
        k: {"old": old_summary.get(k), "new": new_summary.get(k)}
        for k in (
            "top1_accuracy", "topk_recall", "policy_satisfied_rate",
            "avg_false_positives_per_query",
        )
        if old_summary.get(k) != new_summary.get(k)
    }

    return {
        "prior_artifact_path": str(prior_path.relative_to(REPO_ROOT)),
        "runner_name": runner_name,
        "runner_config": runner_config,
        "old_summary": old_summary,
        "new_summary": new_summary,
        "summary_deltas": summary_deltas,
        "per_question_diffs": per_question_diffs,
        "n_questions_total": len(prior_results),
        "n_questions_unchanged": n_unchanged,
        "n_questions_changed": len(per_question_diffs),
    }


def main() -> int:
    out_dir = REPO_ROOT / "runs/p1_11_post"
    out_dir.mkdir(parents=True, exist_ok=True)
    questions_dir = out_dir / "questions"

    promoted: list[dict[str, str]] = []
    scenes: list[dict[str, Any]] = []

    for entry in PRIOR_RUNS:
        scene_id = entry["scene_id"]
        v01_path = Path(entry["fixture_v01"])
        v02_path = questions_dir / f"{scene_id}.v0_2.json"
        promote_v01_to_v02(v01_path, v02_path)
        promoted.append({
            "scene_id": scene_id,
            "v01_path": str(v01_path.relative_to(REPO_ROOT)),
            "v02_path": str(v02_path.relative_to(REPO_ROOT)),
        })

        questions_v02 = load_questions(v02_path)
        scene_objects = _scene_objects(Path(entry["scene_graph"]))

        artifact_comparisons = []
        for prior_path in entry["prior_artifacts"]:
            artifact_comparisons.append(
                compare_one(
                    scene_id=scene_id,
                    questions_v02=questions_v02,
                    scene_objects=scene_objects,
                    prior_path=Path(prior_path),
                )
            )
        scenes.append({
            "scene_id": scene_id,
            "scene_graph": str(Path(entry["scene_graph"]).relative_to(REPO_ROOT)),
            "n_questions": len(questions_v02),
            "policy_breakdown": {
                p: sum(1 for q in questions_v02 if q.ambiguity_policy == p)
                for p in {q.ambiguity_policy for q in questions_v02}
            },
            "answer_type_breakdown": {
                t: sum(1 for q in questions_v02 if q.answer_type == t)
                for t in {q.answer_type for q in questions_v02}
            },
            "prior_artifact_comparisons": artifact_comparisons,
        })

    artifact = {
        "change_type": "benchmark_semantics_change",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "from_schema_version": "v0.1",
        "to_schema_version": SCHEMA_VERSION,
        "what_changed": [
            "any_of_subset policy now requires zero false positives in addition "
            "to >= 1 expected hit (previously identical to one_of by bug; see "
            "phase0_design.md §7.2 and benchmark/runner.py:_policy_satisfied)",
            "count and yes_no answer_types are now scored directly from "
            "RunnerOutput.answer_count / answer_yes_no (previously unscored; "
            "see phase0_design.md §7.1)",
            "schema_version bumped v0.1 → v0.2 with optional expected_count and "
            "expected_yes_no fields on Question (entity-typed questions are "
            "unchanged in shape)",
        ],
        "what_did_not_change": [
            "graph construction (no extractor edits)",
            "relation extraction logic (directional + proximity unchanged in compat mode)",
            "parser behavior (regex / llm parser paths untouched)",
            "scorer/ranking behavior (scoring.v1 and scoring.v2 untouched)",
            "top-k logic (scoring.topk untouched)",
            "model outputs (RunnerOutputs are replayed from cache; no re-execution)",
        ],
        "method": (
            "RunnerOutputs are reconstructed from the preserved 'output' field "
            "of each scored result in the prior artifacts. New score_output is "
            "applied verbatim. No parser, scorer, or graph code runs during "
            "this comparison."
        ),
        "promoted_fixtures": promoted,
        "scenes": scenes,
    }

    artifact_path = out_dir / "comparison.json"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote: {artifact_path.relative_to(REPO_ROOT)}")
    print(f"Promoted v0.2 fixtures: {questions_dir.relative_to(REPO_ROOT)}/")
    for scene in scenes:
        print(f"\nScene: {scene['scene_id']}")
        for comp in scene["prior_artifact_comparisons"]:
            print(f"  Prior: {comp['prior_artifact_path']}")
            print(f"    Config: {comp['runner_config']}")
            print(f"    Changed: {comp['n_questions_changed']} of {comp['n_questions_total']} questions")
            if comp["summary_deltas"]:
                for k, v in comp["summary_deltas"].items():
                    print(f"      {k}: {v['old']} -> {v['new']}")
            else:
                print("      (no summary deltas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Runner I/O types and shared scoring/rollup logic.

Three structs:
  - Evidence:      what the runner cites as the basis for its answer
  - RunnerOutput:  what a single runner produced for a single question
  - ScoredResult:  RunnerOutput + scoring against the Question's expected_targets

Scoring is computed once, here, regardless of which runner produced the output.
That's the property that makes graph / vlm / hybrid results comparable: same
question, same scoring, different RunnerOutput shape.

Lenient scoring (top1_correct / topk_correct) reproduces v1 semantics — any
expected target overlapping any returned entity is a hit. Strict scoring
(policy_satisfied) honors the question's ambiguity_policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from benchmark.schema import AmbiguityPolicy, Question


AbstentionOutcome = Literal[
    "correct_abstain",   # abstained on a true-negative question
    "wrong_abstain",     # abstained when expected had targets
    "false_answer",      # answered when expected was empty
    "true_answer",       # answered, top-k contained an expected target
    "miss",              # answered, no expected target hit
]

FailureAttribution = Literal[
    "none",
    "parser",
    "scorer",
    "scene_graph",
    "vlm_format",
    "vlm_hallucination",
    "abstention",
]


@dataclass
class Evidence:
    """What the runner cites as basis for its answer.

    Graph-only runners populate entity_ids + relation_path + distance_m.
    VLM/hybrid runners additionally populate source_frame_idx + crop_bbox.
    Empty-default fields are explicitly None so absence is distinguishable
    from zero.
    """
    entity_ids: list[str] = field(default_factory=list)
    relation_path: list[dict[str, str]] = field(default_factory=list)
    distance_m: float | None = None
    source_frame_idx: int | None = None
    crop_bbox: list[float] | None = None  # [x, y, w, h] normalized


@dataclass
class RunnerOutput:
    """What a single runner produced for a single question."""
    question_id: str
    abstained: bool = False
    answer_entity_ids: list[str] = field(default_factory=list)  # ordered top-k
    answer_text: str = ""
    answer_count: int | None = None
    answer_yes_no: bool | None = None
    confidence: float | None = None
    evidence: Evidence = field(default_factory=Evidence)
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class ScoredResult:
    question_id: str
    scene_id: str
    runner_name: str
    runner_config: dict[str, Any]
    output: RunnerOutput
    top1_correct: bool
    topk_correct: bool
    policy_satisfied: bool
    abstention_outcome: AbstentionOutcome
    false_positives: int
    expected_covered: int
    expected_total: int
    failure_attribution: FailureAttribution


def _entity_matches_target(
    entity_id: str,
    target_canonical: str,
    target_kind: str,
    scene_objects: dict[str, dict[str, Any]],
) -> bool:
    """Decide if a returned canonical_id covers an expected target.
    For target_kind=zone, look up the entity's zone string in the scene."""
    if target_kind == "entity":
        return entity_id == target_canonical
    obj = scene_objects.get(entity_id)
    if obj is None:
        return False
    return str(obj.get("zone", "")) == target_canonical


def _policy_satisfied(
    policy: AmbiguityPolicy,
    expected_covered: set[str],
    expected_canonical_ids: list[str],
    abstained: bool,
    answer_count: int,
) -> bool:
    if policy == "none":
        return abstained or answer_count == 0
    if policy == "abstain_ok":
        return abstained or len(expected_covered) >= 1
    if policy == "one_of":
        return len(expected_covered) >= 1
    if policy == "any_of_subset":
        return len(expected_covered) >= 1
    if policy == "all_of":
        return expected_covered == set(expected_canonical_ids)
    return False


def score_output(
    question: Question,
    output: RunnerOutput,
    scene_objects: dict[str, dict[str, Any]],
    *,
    runner_name: str,
    runner_config: dict[str, Any],
) -> ScoredResult:
    answers = list(output.answer_entity_ids)
    targets = question.expected_targets
    expected_count = len(targets)
    expected_canonical = [t.canonical_id for t in targets]

    covered: set[str] = set()
    for ans in answers:
        for t in targets:
            if _entity_matches_target(ans, t.canonical_id, t.target_kind, scene_objects):
                covered.add(t.canonical_id)

    if expected_count == 0:
        # True-negative: success means returning nothing (or abstaining).
        top1_ok = output.abstained or len(answers) == 0
        topk_ok = top1_ok
        false_positives = len(answers)
    elif output.abstained:
        top1_ok = False
        topk_ok = False
        false_positives = 0
    else:
        top1_ok = bool(answers) and any(
            _entity_matches_target(answers[0], t.canonical_id, t.target_kind, scene_objects)
            for t in targets
        )
        topk_ok = len(covered) > 0
        false_positives = sum(
            1 for a in answers
            if not any(
                _entity_matches_target(a, t.canonical_id, t.target_kind, scene_objects)
                for t in targets
            )
        )

    policy_ok = _policy_satisfied(
        question.ambiguity_policy,
        covered,
        expected_canonical,
        output.abstained,
        len(answers),
    )

    if output.abstained:
        abst: AbstentionOutcome = "correct_abstain" if expected_count == 0 else "wrong_abstain"
    elif expected_count == 0:
        abst = "correct_abstain" if not answers else "false_answer"
    else:
        abst = "true_answer" if topk_ok else "miss"

    fail: FailureAttribution = "none"
    if output.error == "unparseable":
        fail = "parser"
    elif output.error and output.error.lower().startswith("vlm_format"):
        fail = "vlm_format"
    elif output.error and output.error.lower().startswith("vlm_hallucination"):
        fail = "vlm_hallucination"
    elif expected_count == 0 and answers:
        fail = "scorer"
    elif expected_count > 0 and not topk_ok:
        if output.abstained:
            fail = "abstention"
        elif not answers:
            fail = "scene_graph"
        else:
            fail = "scorer"
    elif expected_count > 0 and topk_ok and not top1_ok:
        fail = "scorer"

    return ScoredResult(
        question_id=question.question_id,
        scene_id=question.scene_id,
        runner_name=runner_name,
        runner_config=runner_config,
        output=output,
        top1_correct=top1_ok,
        topk_correct=topk_ok,
        policy_satisfied=policy_ok,
        abstention_outcome=abst,
        false_positives=false_positives,
        expected_covered=len(covered),
        expected_total=expected_count,
        failure_attribution=fail,
    )


def rollup(
    results: list[ScoredResult],
    questions: list[Question],
) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"n_questions": 0}

    by_qid = {q.question_id: q for q in questions}
    cat_buckets: dict[str, list[ScoredResult]] = {}
    for r in results:
        q = by_qid.get(r.question_id)
        cat = q.category if q is not None else "unknown"
        cat_buckets.setdefault(cat, []).append(r)

    def _acc(rs: list[ScoredResult], attr: str) -> float:
        return sum(1 for r in rs if getattr(r, attr)) / len(rs) if rs else 0.0

    per_category = {
        cat: {
            "n": len(rs),
            "top1": _acc(rs, "top1_correct"),
            "topk": _acc(rs, "topk_correct"),
            "policy_satisfied": _acc(rs, "policy_satisfied"),
            "mean_fp": sum(r.false_positives for r in rs) / len(rs),
        }
        for cat, rs in cat_buckets.items()
    }

    abst_counts: dict[str, int] = {}
    for r in results:
        abst_counts[r.abstention_outcome] = abst_counts.get(r.abstention_outcome, 0) + 1

    fail_counts: dict[str, int] = {}
    for r in results:
        fail_counts[r.failure_attribution] = fail_counts.get(r.failure_attribution, 0) + 1

    latencies = [r.output.latency_ms for r in results]

    return {
        "n_questions": n,
        "top1_accuracy": _acc(results, "top1_correct"),
        "topk_recall": _acc(results, "topk_correct"),
        "policy_satisfied_rate": _acc(results, "policy_satisfied"),
        "avg_false_positives_per_query": sum(r.false_positives for r in results) / n,
        "abstention_outcome_counts": abst_counts,
        "failure_attribution_counts": fail_counts,
        "per_category": per_category,
        "mean_latency_ms": sum(latencies) / n,
        "max_latency_ms": max(latencies) if latencies else 0.0,
    }


def evidence_to_dict(e: Evidence) -> dict[str, Any]:
    return {
        "entity_ids": list(e.entity_ids),
        "relation_path": [dict(r) for r in e.relation_path],
        "distance_m": e.distance_m,
        "source_frame_idx": e.source_frame_idx,
        "crop_bbox": list(e.crop_bbox) if e.crop_bbox is not None else None,
    }


def runner_output_to_dict(o: RunnerOutput) -> dict[str, Any]:
    return {
        "question_id": o.question_id,
        "abstained": o.abstained,
        "answer_entity_ids": list(o.answer_entity_ids),
        "answer_text": o.answer_text,
        "answer_count": o.answer_count,
        "answer_yes_no": o.answer_yes_no,
        "confidence": o.confidence,
        "evidence": evidence_to_dict(o.evidence),
        "latency_ms": o.latency_ms,
        "error": o.error,
    }


def scored_result_to_dict(r: ScoredResult) -> dict[str, Any]:
    return {
        "question_id": r.question_id,
        "scene_id": r.scene_id,
        "runner_name": r.runner_name,
        "runner_config": dict(r.runner_config),
        "output": runner_output_to_dict(r.output),
        "top1_correct": r.top1_correct,
        "topk_correct": r.topk_correct,
        "policy_satisfied": r.policy_satisfied,
        "abstention_outcome": r.abstention_outcome,
        "false_positives": r.false_positives,
        "expected_covered": r.expected_covered,
        "expected_total": r.expected_total,
        "failure_attribution": r.failure_attribution,
    }

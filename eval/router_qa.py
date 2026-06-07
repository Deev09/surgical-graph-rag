"""Phase 5 P5.04 — reasoner-native QA scorer (new eval track).

Scores a mixed question set by running each question through the real Router
(compiler -> executor -> verbalizer) and classifying the Answer against an
explicit expected outcome. This is a SEPARATE eval track from
benchmark/runner.py (which scores top-k retrieval, a different shape of
system); the two are NOT comparable and this does not touch the v1 benchmark.

Pure library: takes an already-built bundle + a Router + the question list +
an ExecutionContext, returns a scorecard dict. No Replica build, no I/O —
the tools/ runner does graph assembly and artifact I/O. This keeps the
scorer unit-testable on tiny synthetic bundles.

Expected outcomes (per question): answer, empty, defer, unknown,
parser_failure, execution_error.

Score categories:
  true_answer          expected answer; answered; cited covers must_contain
                       and includes none of must_not_contain.
  false_answer         answered when expected defer/empty (fabricated), OR
                       answered including a must_not_contain entity.
  miss                 expected answer; answered but missed a must_contain
                       (or returned empty/unknown).
  correct_defer        expected defer; deferred (out_of_schema + deferred note).
  true_empty           expected empty; returned empty.
  true_unknown / true_parser_failure / true_execution_error
                       schema support for the rarer expected outcomes.
  unexpected           outcome did not match the expected category.

Deferral detection (harness limitation, recorded): the Answer dataclass does
not self-describe deferral -- Answer.outcome == "abstain" is overloaded
across deferred / generic out_of_schema / execution_error. So we read the
COMPILE metadata via the Router's own compiler (out_of_schema + a "deferred:"
note), which is deterministic (no LLM compiler in P5). A future reasoner
enhancement could add a `deferred` flag to Answer.
"""
from __future__ import annotations

from collections import Counter

from graph.schema import SceneGraphBundle
from reasoner.base import ExecutionContext


DEFERRAL_DETECTION = "compile_metadata"

# Categories that count as the expected outcome being met.
SUCCESS_CATEGORIES = frozenset({
    "true_answer", "correct_defer", "true_empty",
    "true_unknown", "true_parser_failure", "true_execution_error",
})


def _is_deferred(router, question: str, bundle: SceneGraphBundle) -> bool:
    """Read the compile-time deferral metadata via the Router's own compiler.
    Deterministic: with no LLM compiler in P5, this equals what
    router.answer() used internally."""
    cr = router.compiler.compile(question, bundle)
    return cr.outcome == "out_of_schema" and (cr.notes or "").startswith("deferred:")


def _classify(question: dict, ans, deferred: bool) -> str:
    exp = question["expected_outcome"]
    got = ans.outcome
    cited = set(ans.cited_uids)
    must_contain = question.get("expected_must_contain", []) or []
    must_not_contain = question.get("expected_must_not_contain", []) or []

    if exp == "answer":
        if got != "bindings":
            return "miss"
        if any(m not in cited for m in must_contain):
            return "miss"
        if any(m in cited for m in must_not_contain):
            return "false_answer"
        return "true_answer"
    if exp == "defer":
        if got == "bindings":
            return "false_answer"
        if deferred:
            return "correct_defer"
        return "unexpected"
    if exp == "empty":
        if got == "bindings":
            return "false_answer"
        if got == "empty":
            return "true_empty"
        return "unexpected"
    if exp == "unknown":
        if got == "bindings":
            return "false_answer"
        return "true_unknown" if got == "unknown" else "unexpected"
    if exp == "parser_failure":
        if got == "bindings":
            return "false_answer"
        return "true_parser_failure" if got == "parser_failure" else "unexpected"
    if exp == "execution_error":
        # execution_error surfaces as Answer.outcome == "abstain" (overloaded);
        # not exercised by the P5 set. Best-effort, flagged as a limitation.
        if got == "bindings":
            return "false_answer"
        return "true_execution_error" if got == "abstain" else "unexpected"
    return "unexpected"


def score_questions(
    questions: list[dict],
    bundle: SceneGraphBundle,
    router,
    ctx: ExecutionContext,
) -> dict:
    """Run every question through the Router and classify. Returns a
    deterministic scorecard dict (rows sorted by question_id; cited lists
    sorted)."""
    rows: list[dict] = []
    for q in sorted(questions, key=lambda x: x["question_id"]):
        deferred = _is_deferred(router, q["question"], bundle)
        ans = router.answer(q["question"], bundle, ctx)
        category = _classify(q, ans, deferred)
        rows.append({
            "question_id": q["question_id"],
            "question": q["question"],
            "expected_outcome": q["expected_outcome"],
            "actual_outcome": ans.outcome,
            "deferred": deferred,
            "category": category,
            "cited_uids": sorted(ans.cited_uids),
            "cited_edges": sorted(ans.cited_edges),
        })

    counts = Counter(r["category"] for r in rows)
    all_met = all(r["category"] in SUCCESS_CATEGORIES for r in rows)
    return {
        "deferral_detection": DEFERRAL_DETECTION,
        "per_question": rows,
        "aggregate": {
            "total_questions": len(rows),
            "category_counts": dict(sorted(counts.items())),
            "false_answer_count": counts.get("false_answer", 0),
            "miss_count": counts.get("miss", 0),
            "unexpected_count": counts.get("unexpected", 0),
            "all_expected_outcomes_met": all_met,
        },
    }

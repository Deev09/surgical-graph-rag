"""P1.11 tests: benchmark schema v0.2 + runner scoring fixes.

Run: python tests/benchmark/test_runner_and_schema.py

Required by the batch instructions:
  - Schema validation (every answer_type × scalar-presence combination).
  - Serde round-trips for v0.2 questions with scalar expected values.
  - v0.1 backward compat: loadable with DeprecationWarning.
  - Scalar scoring: count and yes_no, correct / incorrect / missing /
    abstained — top1_correct, topk_correct, policy_satisfied all agree.
  - any_of_subset: hit + extra → not satisfied; hit + no extra → satisfied.
  - one_of: hit + extra → satisfied (permissive).
  - True-negative abstention behavior preserved.
  - Unchanged v0.1 entity scoring for existing-shape question/output.

P1.11 explicitly is a benchmark-semantics change, not a model improvement.
Old v0.1 metrics are NOT directly comparable to v0.2 metrics.
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
import warnings
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.schema import (
    SCHEMA_VERSION, ExpectedTarget, Question, load_questions, save_questions,
    validate_question,
)
from benchmark.runner import (
    Evidence, RunnerOutput, score_output,
)


def _q(
    *, qid: str = "q1", scene: str = "s1", text: str = "test?", category: str = "relative_position",
    answer_type: str = "entity", ambiguity_policy: str = "one_of", requires_3d: bool = False,
    expected_targets: list[ExpectedTarget] | None = None,
    expected_count: int | None = None, expected_yes_no: bool | None = None,
) -> Question:
    return Question(
        question_id=qid, scene_id=scene, text=text, category=category,
        answer_type=answer_type, ambiguity_policy=ambiguity_policy,
        requires_3d=requires_3d,
        expected_targets=expected_targets or [],
        expected_count=expected_count, expected_yes_no=expected_yes_no,
    )


def _target(canonical_id: str) -> ExpectedTarget:
    return ExpectedTarget(
        canonical_id=canonical_id, display_label=canonical_id,
        aliases=[], target_kind="entity",
    )


def _scene_objects(uids: list[str]) -> dict:
    return {u: {"zone": ""} for u in uids}


# ---------- schema validation ----------

def test_validate_count_requires_expected_count() -> None:
    q = _q(answer_type="count", ambiguity_policy="none")
    try:
        validate_question(q)
    except ValueError as e:
        if "expected_count" not in str(e):
            raise AssertionError(f"unexpected error text: {e}")
        return
    raise AssertionError("expected ValueError when expected_count missing for count type")


def test_validate_count_rejects_expected_yes_no() -> None:
    q = _q(
        answer_type="count", ambiguity_policy="none",
        expected_count=3, expected_yes_no=True,
    )
    try:
        validate_question(q)
    except ValueError as e:
        if "expected_yes_no" not in str(e):
            raise AssertionError(f"unexpected error text: {e}")
        return
    raise AssertionError("expected ValueError when both scalars set on count")


def test_validate_yes_no_requires_expected_yes_no() -> None:
    q = _q(answer_type="yes_no", ambiguity_policy="none")
    try:
        validate_question(q)
    except ValueError as e:
        if "expected_yes_no" not in str(e):
            raise AssertionError(f"unexpected error text: {e}")
        return
    raise AssertionError("expected ValueError when expected_yes_no missing")


def test_validate_yes_no_rejects_expected_count() -> None:
    q = _q(
        answer_type="yes_no", ambiguity_policy="none",
        expected_yes_no=False, expected_count=0,
    )
    try:
        validate_question(q)
    except ValueError as e:
        if "expected_count" not in str(e):
            raise AssertionError(f"unexpected error text: {e}")
        return
    raise AssertionError("expected ValueError when both scalars set on yes_no")


def test_validate_entity_rejects_scalar_fields() -> None:
    q = _q(
        answer_type="entity", ambiguity_policy="one_of",
        expected_targets=[_target("obj_1")], expected_count=1,
    )
    try:
        validate_question(q)
    except ValueError as e:
        if "expected_count" not in str(e):
            raise AssertionError(f"unexpected error text: {e}")
        return
    raise AssertionError("expected ValueError when scalar set on entity")


def test_validate_count_accepts_when_well_formed() -> None:
    q = _q(answer_type="count", ambiguity_policy="none", expected_count=5)
    validate_question(q)  # must not raise


def test_validate_yes_no_accepts_when_well_formed() -> None:
    q = _q(answer_type="yes_no", ambiguity_policy="none", expected_yes_no=True)
    validate_question(q)


def test_validate_count_zero_is_valid() -> None:
    """expected_count=0 is a concrete answer, not a missing field."""
    q = _q(answer_type="count", ambiguity_policy="none", expected_count=0)
    validate_question(q)


# ---------- serde round-trips ----------

def _roundtrip(questions: list[Question], scene_id: str = "scene_x") -> list[Question]:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "qs.json"
        save_questions(scene_id, questions, p)
        return load_questions(p)


def test_serde_roundtrip_count_question() -> None:
    q = _q(
        qid="q_count", answer_type="count", ambiguity_policy="none",
        expected_count=7,
    )
    [loaded] = _roundtrip([q])
    if loaded.expected_count != 7 or loaded.answer_type != "count":
        raise AssertionError(f"count round-trip lost: {loaded}")
    if loaded.expected_yes_no is not None:
        raise AssertionError("expected_yes_no should be None for count")


def test_serde_roundtrip_yes_no_question() -> None:
    q = _q(
        qid="q_yn", answer_type="yes_no", ambiguity_policy="none",
        expected_yes_no=False,
    )
    [loaded] = _roundtrip([q])
    if loaded.expected_yes_no is not False or loaded.answer_type != "yes_no":
        raise AssertionError(f"yes_no round-trip lost: {loaded}")
    if loaded.expected_count is not None:
        raise AssertionError("expected_count should be None for yes_no")


def test_serde_roundtrip_entity_unchanged() -> None:
    q = _q(
        qid="q_ent", answer_type="entity", ambiguity_policy="one_of",
        expected_targets=[_target("obj_1")],
    )
    [loaded] = _roundtrip([q])
    if loaded.expected_targets[0].canonical_id != "obj_1":
        raise AssertionError("entity round-trip lost target")
    if loaded.expected_count is not None or loaded.expected_yes_no is not None:
        raise AssertionError("scalar fields should default to None")


def test_serde_writes_current_version() -> None:
    q = _q(answer_type="entity", expected_targets=[_target("obj_1")])
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "qs.json"
        save_questions("scene_x", [q], p)
        raw = json.loads(p.read_text())
    if raw["schema_version"] != SCHEMA_VERSION:
        raise AssertionError(f"expected schema_version {SCHEMA_VERSION}, got {raw['schema_version']!r}")


# ---------- v0.1 backward compat ----------

def test_v0_1_file_loads_with_deprecation_warning() -> None:
    """Construct a synthetic v0.1 file (entity types only — count/yes_no
    in v0.1 were never scoreable). Loader must accept and warn."""
    v0_1 = {
        "schema_version": "v0.1",
        "scene_id": "test_scene",
        "questions": [
            {
                "question_id": "q1", "scene_id": "test_scene",
                "text": "what is left of x?", "category": "relative_position",
                "answer_type": "entity", "ambiguity_policy": "one_of",
                "requires_3d": False,
                "expected_targets": [
                    {"canonical_id": "obj_1", "display_label": "obj_1",
                     "aliases": [], "target_kind": "entity"},
                ],
                "paraphrases": [], "notes": "",
            },
        ],
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "v01.json"
        p.write_text(json.dumps(v0_1))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            qs = load_questions(p)
        deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        if not deprecation:
            raise AssertionError("expected DeprecationWarning when loading v0.1")
        if "NOT comparable" not in str(deprecation[0].message):
            raise AssertionError(
                f"DeprecationWarning text should warn about metric incomparability; got {deprecation[0].message}"
            )
    if len(qs) != 1 or qs[0].expected_count is not None or qs[0].expected_yes_no is not None:
        raise AssertionError("v0.1 entity question should load with scalars defaulted to None")


def test_v0_1_file_with_count_type_but_no_expected_count_rejected() -> None:
    """v0.1 had no expected_count, so a v0.1 question with answer_type=count
    is unscoreable and must fail validation on load."""
    bad_v0_1 = {
        "schema_version": "v0.1",
        "scene_id": "test_scene",
        "questions": [
            {
                "question_id": "q1", "scene_id": "test_scene",
                "text": "how many chairs?", "category": "counting",
                "answer_type": "count", "ambiguity_policy": "none",
                "requires_3d": False,
                "expected_targets": [], "paraphrases": [], "notes": "",
            },
        ],
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "v01_bad.json"
        p.write_text(json.dumps(bad_v0_1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                load_questions(p)
            except ValueError as e:
                if "expected_count" not in str(e):
                    raise AssertionError(f"unexpected error text: {e}")
                return
    raise AssertionError("expected ValueError for v0.1 count question without expected_count")


def test_unsupported_schema_version_rejected() -> None:
    bad = {"schema_version": "v9.9", "scene_id": "x", "questions": []}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "vbad.json"
        p.write_text(json.dumps(bad))
        try:
            load_questions(p)
        except ValueError as e:
            if "not supported" not in str(e):
                raise AssertionError(f"unexpected error text: {e}")
            return
    raise AssertionError("expected ValueError for unsupported schema_version")


# ---------- existing repo question files still load ----------

def test_existing_replica_question_file_still_loads() -> None:
    p = REPO_ROOT / "benchmark" / "questions" / "replica_room_0.json"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        qs = load_questions(p)
    if not qs:
        raise AssertionError("no questions loaded")
    for q in qs:
        if q.expected_count is not None or q.expected_yes_no is not None:
            raise AssertionError(f"v0.1 question {q.question_id} should have scalars=None")


def test_existing_bathroom_question_file_still_loads() -> None:
    p = REPO_ROOT / "benchmark" / "questions" / "graffiti_bathroom.json"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        qs = load_questions(p)
    if not qs:
        raise AssertionError("no questions loaded")


# ---------- scalar scoring: count ----------

def _runner_output(**kwargs) -> RunnerOutput:
    base = {
        "question_id": "q1",
        "abstained": False,
        "answer_entity_ids": [],
        "answer_text": "",
        "answer_count": None,
        "answer_yes_no": None,
        "confidence": None,
        "evidence": Evidence(),
        "latency_ms": 0.0,
        "error": None,
    }
    base.update(kwargs)
    return RunnerOutput(**base)


def test_count_correct() -> None:
    q = _q(answer_type="count", ambiguity_policy="none", expected_count=3)
    out = _runner_output(answer_count=3)
    r = score_output(q, out, {}, runner_name="t", runner_config={})
    if not (r.top1_correct and r.topk_correct and r.policy_satisfied):
        raise AssertionError("count correct: all three flags should be True")
    if r.false_positives != 0 or r.expected_covered != 1:
        raise AssertionError(f"count correct: bad counts {r}")
    if r.failure_attribution != "none":
        raise AssertionError(f"count correct: failure should be 'none', got {r.failure_attribution}")


def test_count_incorrect() -> None:
    q = _q(answer_type="count", ambiguity_policy="none", expected_count=3)
    out = _runner_output(answer_count=5)
    r = score_output(q, out, {}, runner_name="t", runner_config={})
    if r.top1_correct or r.topk_correct or r.policy_satisfied:
        raise AssertionError("count incorrect: all three flags should be False")
    if r.false_positives != 1:
        raise AssertionError(f"count incorrect: expected fp=1, got {r.false_positives}")
    if r.failure_attribution != "scorer":
        raise AssertionError(f"count incorrect: expected 'scorer', got {r.failure_attribution}")


def test_count_missing() -> None:
    """Runner provided no answer_count (None). Treat as scene_graph failure."""
    q = _q(answer_type="count", ambiguity_policy="none", expected_count=3)
    out = _runner_output()  # answer_count=None
    r = score_output(q, out, {}, runner_name="t", runner_config={})
    if r.top1_correct or r.topk_correct or r.policy_satisfied:
        raise AssertionError("count missing: all three flags should be False")
    if r.failure_attribution != "scene_graph":
        raise AssertionError(f"count missing: expected 'scene_graph', got {r.failure_attribution}")


def test_count_abstained() -> None:
    q = _q(answer_type="count", ambiguity_policy="none", expected_count=3)
    out = _runner_output(abstained=True)
    r = score_output(q, out, {}, runner_name="t", runner_config={})
    if r.abstention_outcome != "wrong_abstain":
        raise AssertionError(f"expected wrong_abstain, got {r.abstention_outcome}")
    if r.failure_attribution != "abstention":
        raise AssertionError(f"expected abstention, got {r.failure_attribution}")


def test_count_zero_correct() -> None:
    q = _q(answer_type="count", ambiguity_policy="none", expected_count=0)
    out = _runner_output(answer_count=0)
    r = score_output(q, out, {}, runner_name="t", runner_config={})
    if not r.top1_correct:
        raise AssertionError("answer_count=0 should match expected_count=0")


# ---------- scalar scoring: yes_no ----------

def test_yes_no_correct() -> None:
    q = _q(answer_type="yes_no", ambiguity_policy="none", expected_yes_no=True)
    out = _runner_output(answer_yes_no=True)
    r = score_output(q, out, {}, runner_name="t", runner_config={})
    if not (r.top1_correct and r.topk_correct and r.policy_satisfied):
        raise AssertionError("yes_no correct: all three flags should be True")


def test_yes_no_incorrect() -> None:
    q = _q(answer_type="yes_no", ambiguity_policy="none", expected_yes_no=True)
    out = _runner_output(answer_yes_no=False)
    r = score_output(q, out, {}, runner_name="t", runner_config={})
    if r.top1_correct or r.policy_satisfied:
        raise AssertionError("yes_no incorrect: flags should be False")
    if r.false_positives != 1:
        raise AssertionError(f"yes_no incorrect: expected fp=1, got {r.false_positives}")


def test_yes_no_missing() -> None:
    q = _q(answer_type="yes_no", ambiguity_policy="none", expected_yes_no=True)
    out = _runner_output()  # answer_yes_no=None
    r = score_output(q, out, {}, runner_name="t", runner_config={})
    if r.top1_correct or r.policy_satisfied:
        raise AssertionError("yes_no missing: flags should be False")
    if r.failure_attribution != "scene_graph":
        raise AssertionError(f"expected scene_graph, got {r.failure_attribution}")


def test_yes_no_abstained() -> None:
    q = _q(answer_type="yes_no", ambiguity_policy="none", expected_yes_no=False)
    out = _runner_output(abstained=True)
    r = score_output(q, out, {}, runner_name="t", runner_config={})
    if r.abstention_outcome != "wrong_abstain":
        raise AssertionError(f"expected wrong_abstain, got {r.abstention_outcome}")


# ---------- any_of_subset fix ----------

def test_any_of_subset_hit_with_no_extras_satisfied() -> None:
    q = _q(
        answer_type="entity_list", ambiguity_policy="any_of_subset",
        expected_targets=[_target("obj_1"), _target("obj_2")],
    )
    out = _runner_output(answer_entity_ids=["obj_1"])
    r = score_output(q, out, _scene_objects(["obj_1", "obj_2"]), runner_name="t", runner_config={})
    if not r.policy_satisfied:
        raise AssertionError("any_of_subset: hit + no extras should satisfy")


def test_any_of_subset_hit_with_extras_rejected() -> None:
    """The P1.11 fix: hit-plus-garbage no longer satisfies any_of_subset."""
    q = _q(
        answer_type="entity_list", ambiguity_policy="any_of_subset",
        expected_targets=[_target("obj_1"), _target("obj_2")],
    )
    out = _runner_output(answer_entity_ids=["obj_1", "obj_99"])
    r = score_output(q, out, _scene_objects(["obj_1", "obj_2", "obj_99"]), runner_name="t", runner_config={})
    if r.policy_satisfied:
        raise AssertionError(
            "any_of_subset: hit + extra should NOT satisfy after P1.11 fix"
        )
    if r.false_positives != 1:
        raise AssertionError(f"expected fp=1, got {r.false_positives}")


def test_any_of_subset_full_set_no_extras_satisfied() -> None:
    q = _q(
        answer_type="entity_list", ambiguity_policy="any_of_subset",
        expected_targets=[_target("obj_1"), _target("obj_2")],
    )
    out = _runner_output(answer_entity_ids=["obj_1", "obj_2"])
    r = score_output(q, out, _scene_objects(["obj_1", "obj_2"]), runner_name="t", runner_config={})
    if not r.policy_satisfied:
        raise AssertionError("any_of_subset: full expected set should satisfy")


# ---------- one_of preserved ----------

def test_one_of_hit_with_extras_still_satisfied() -> None:
    q = _q(
        answer_type="entity_list", ambiguity_policy="one_of",
        expected_targets=[_target("obj_1"), _target("obj_2")],
    )
    out = _runner_output(answer_entity_ids=["obj_1", "obj_99"])
    r = score_output(q, out, _scene_objects(["obj_1", "obj_2", "obj_99"]), runner_name="t", runner_config={})
    if not r.policy_satisfied:
        raise AssertionError("one_of: hit + extra should still satisfy (permissive)")


def test_one_of_no_hit_not_satisfied() -> None:
    q = _q(
        answer_type="entity", ambiguity_policy="one_of",
        expected_targets=[_target("obj_1")],
    )
    out = _runner_output(answer_entity_ids=["obj_99"])
    r = score_output(q, out, _scene_objects(["obj_1", "obj_99"]), runner_name="t", runner_config={})
    if r.policy_satisfied:
        raise AssertionError("one_of: no hit should not satisfy")


# ---------- true-negative abstention preserved ----------

def test_true_negative_empty_answer_correct() -> None:
    q = _q(answer_type="none", ambiguity_policy="none")
    out = _runner_output(answer_entity_ids=[])
    r = score_output(q, out, _scene_objects([]), runner_name="t", runner_config={})
    if not r.top1_correct:
        raise AssertionError("true-negative empty answer should be top1_correct")
    if r.abstention_outcome != "correct_abstain":
        raise AssertionError(f"expected correct_abstain, got {r.abstention_outcome}")


def test_true_negative_abstain_correct() -> None:
    q = _q(answer_type="none", ambiguity_policy="none")
    out = _runner_output(abstained=True)
    r = score_output(q, out, _scene_objects([]), runner_name="t", runner_config={})
    if r.abstention_outcome != "correct_abstain":
        raise AssertionError(f"expected correct_abstain, got {r.abstention_outcome}")


def test_true_negative_false_answer_fails() -> None:
    q = _q(answer_type="none", ambiguity_policy="none")
    out = _runner_output(answer_entity_ids=["obj_99"])
    r = score_output(q, out, _scene_objects(["obj_99"]), runner_name="t", runner_config={})
    if r.top1_correct:
        raise AssertionError("true-negative with answer should NOT be correct")
    if r.abstention_outcome != "false_answer":
        raise AssertionError(f"expected false_answer, got {r.abstention_outcome}")


# ---------- unchanged entity scoring (regression) ----------

def test_entity_single_hit_top1_and_topk() -> None:
    q = _q(
        answer_type="entity", ambiguity_policy="one_of",
        expected_targets=[_target("obj_42")],
    )
    out = _runner_output(answer_entity_ids=["obj_42"])
    r = score_output(q, out, _scene_objects(["obj_42"]), runner_name="t", runner_config={})
    if not (r.top1_correct and r.topk_correct and r.policy_satisfied):
        raise AssertionError("entity hit: all three flags should be True")
    if r.false_positives != 0:
        raise AssertionError(f"expected fp=0, got {r.false_positives}")


def test_entity_top1_wrong_topk_right() -> None:
    q = _q(
        answer_type="entity_list", ambiguity_policy="one_of",
        expected_targets=[_target("obj_1")],
    )
    out = _runner_output(answer_entity_ids=["obj_99", "obj_1"])
    r = score_output(q, out, _scene_objects(["obj_1", "obj_99"]), runner_name="t", runner_config={})
    if r.top1_correct:
        raise AssertionError("top1 should be False (first answer wrong)")
    if not r.topk_correct:
        raise AssertionError("topk should be True (later answer right)")
    if not r.policy_satisfied:
        raise AssertionError("one_of policy: hit + extra still satisfies")


def test_entity_failure_attribution_scene_graph_when_no_answers() -> None:
    q = _q(
        answer_type="entity", ambiguity_policy="one_of",
        expected_targets=[_target("obj_1")],
    )
    out = _runner_output(answer_entity_ids=[])
    r = score_output(q, out, _scene_objects(["obj_1"]), runner_name="t", runner_config={})
    if r.failure_attribution != "scene_graph":
        raise AssertionError(f"expected scene_graph, got {r.failure_attribution}")


def test_entity_failure_attribution_scorer_when_wrong_answers() -> None:
    q = _q(
        answer_type="entity", ambiguity_policy="one_of",
        expected_targets=[_target("obj_1")],
    )
    out = _runner_output(answer_entity_ids=["obj_99"])
    r = score_output(q, out, _scene_objects(["obj_1", "obj_99"]), runner_name="t", runner_config={})
    if r.failure_attribution != "scorer":
        raise AssertionError(f"expected scorer, got {r.failure_attribution}")


TESTS = [
    # schema validation
    test_validate_count_requires_expected_count,
    test_validate_count_rejects_expected_yes_no,
    test_validate_yes_no_requires_expected_yes_no,
    test_validate_yes_no_rejects_expected_count,
    test_validate_entity_rejects_scalar_fields,
    test_validate_count_accepts_when_well_formed,
    test_validate_yes_no_accepts_when_well_formed,
    test_validate_count_zero_is_valid,
    # serde
    test_serde_roundtrip_count_question,
    test_serde_roundtrip_yes_no_question,
    test_serde_roundtrip_entity_unchanged,
    test_serde_writes_current_version,
    # backward compat
    test_v0_1_file_loads_with_deprecation_warning,
    test_v0_1_file_with_count_type_but_no_expected_count_rejected,
    test_unsupported_schema_version_rejected,
    test_existing_replica_question_file_still_loads,
    test_existing_bathroom_question_file_still_loads,
    # scalar scoring: count
    test_count_correct,
    test_count_incorrect,
    test_count_missing,
    test_count_abstained,
    test_count_zero_correct,
    # scalar scoring: yes_no
    test_yes_no_correct,
    test_yes_no_incorrect,
    test_yes_no_missing,
    test_yes_no_abstained,
    # any_of_subset fix
    test_any_of_subset_hit_with_no_extras_satisfied,
    test_any_of_subset_hit_with_extras_rejected,
    test_any_of_subset_full_set_no_extras_satisfied,
    # one_of preserved
    test_one_of_hit_with_extras_still_satisfied,
    test_one_of_no_hit_not_satisfied,
    # true-negative
    test_true_negative_empty_answer_correct,
    test_true_negative_abstain_correct,
    test_true_negative_false_answer_fails,
    # entity regression
    test_entity_single_hit_top1_and_topk,
    test_entity_top1_wrong_topk_right,
    test_entity_failure_attribution_scene_graph_when_no_answers,
    test_entity_failure_attribution_scorer_when_wrong_answers,
]


def main() -> int:
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
            print()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

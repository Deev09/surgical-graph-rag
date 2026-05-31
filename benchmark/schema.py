"""Benchmark schema for spatial-QA evaluation.

P1.11 bump: v0.1 → v0.2.

  v0.1: entity / entity_list / location / none questions. count and
        yes_no answer_types existed in the schema but the runner never
        scored them (bug — see phase0_design.md §7.1).
  v0.2: count and yes_no are first-class. Adds optional
        expected_count: int | None and expected_yes_no: bool | None
        fields to Question. Validation requires the scalar field that
        matches the answer_type and forbids the mismatched scalar.

Backward compatibility:
  - v0.1 files load with a DeprecationWarning. Old questions had no
    scalar fields; defaulting them to None means entity-type questions
    still load cleanly. Old v0.1 files with count or yes_no answer_type
    were never scoreable and will fail validation on load — that is the
    correct breaking change; their historical "metrics" were 0.0 by bug,
    not by design.
  - Save always writes the current SCHEMA_VERSION (v0.2). Loading an old
    file and saving promotes it.
  - Old frozen JSON eval artifacts in baselines/ and scenes/*/eval/ are
    NOT re-scored automatically. Their historical metrics remain in
    place as artifacts of v0.1 scoring; new runs produce v0.2 metrics
    that are NOT directly comparable.

The cross-runner evaluator stores results-side fields in
benchmark/runner.py. Their semantics changed in P1.11 (any_of_subset
fix; count/yes_no scoring); see runner module for details.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from benchmark.categories import CATEGORIES


SCHEMA_VERSION = "v0.2"
SUPPORTED_VERSIONS = ("v0.1", "v0.2")

AnswerType = Literal["entity", "entity_list", "count", "yes_no", "location", "none"]
AmbiguityPolicy = Literal["one_of", "all_of", "any_of_subset", "none", "abstain_ok"]
TargetKind = Literal["entity", "zone"]

_VALID_ANSWER_TYPES: frozenset[str] = frozenset({
    "entity", "entity_list", "count", "yes_no", "location", "none",
})
_VALID_AMBIGUITY_POLICIES: frozenset[str] = frozenset({
    "one_of", "all_of", "any_of_subset", "none", "abstain_ok",
})
_VALID_TARGET_KINDS: frozenset[str] = frozenset({"entity", "zone"})


@dataclass
class ExpectedTarget:
    canonical_id: str
    display_label: str
    aliases: list[str] = field(default_factory=list)
    target_kind: TargetKind = "entity"


@dataclass
class Question:
    question_id: str
    scene_id: str
    text: str
    category: str
    answer_type: AnswerType
    ambiguity_policy: AmbiguityPolicy
    requires_3d: bool
    expected_targets: list[ExpectedTarget] = field(default_factory=list)
    paraphrases: list[str] = field(default_factory=list)
    notes: str = ""
    # P1.11 v0.2 additions:
    expected_count: int | None = None
    expected_yes_no: bool | None = None


def validate_question(q: Question) -> None:
    if q.category not in CATEGORIES:
        raise ValueError(f"{q.question_id}: unknown category {q.category!r}")
    if q.answer_type not in _VALID_ANSWER_TYPES:
        raise ValueError(f"{q.question_id}: unknown answer_type {q.answer_type!r}")
    if q.ambiguity_policy not in _VALID_AMBIGUITY_POLICIES:
        raise ValueError(f"{q.question_id}: unknown ambiguity_policy {q.ambiguity_policy!r}")
    if q.answer_type == "none" and q.expected_targets:
        raise ValueError(f"{q.question_id}: answer_type=none must have empty expected_targets")
    if (
        q.answer_type not in ("none", "count", "yes_no")
        and q.ambiguity_policy != "none"
        and not q.expected_targets
    ):
        raise ValueError(f"{q.question_id}: missing expected_targets")
    for t in q.expected_targets:
        if t.target_kind not in _VALID_TARGET_KINDS:
            raise ValueError(f"{q.question_id}: bad target_kind {t.target_kind!r}")

    # P1.11 v0.2: scalar expected values are present only for their
    # matching answer_type. Mismatches are rejected upfront.
    if q.answer_type == "count":
        if q.expected_count is None:
            raise ValueError(
                f"{q.question_id}: answer_type=count requires expected_count to be set"
            )
        if q.expected_yes_no is not None:
            raise ValueError(
                f"{q.question_id}: answer_type=count must not set expected_yes_no"
            )
    elif q.answer_type == "yes_no":
        if q.expected_yes_no is None:
            raise ValueError(
                f"{q.question_id}: answer_type=yes_no requires expected_yes_no to be set"
            )
        if q.expected_count is not None:
            raise ValueError(
                f"{q.question_id}: answer_type=yes_no must not set expected_count"
            )
    else:
        if q.expected_count is not None:
            raise ValueError(
                f"{q.question_id}: answer_type={q.answer_type} must not set expected_count"
            )
        if q.expected_yes_no is not None:
            raise ValueError(
                f"{q.question_id}: answer_type={q.answer_type} must not set expected_yes_no"
            )


def _target_to_dict(t: ExpectedTarget) -> dict[str, Any]:
    return {
        "canonical_id": t.canonical_id,
        "display_label": t.display_label,
        "aliases": list(t.aliases),
        "target_kind": t.target_kind,
    }


def _question_to_dict(q: Question) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question_id": q.question_id,
        "scene_id": q.scene_id,
        "text": q.text,
        "category": q.category,
        "answer_type": q.answer_type,
        "ambiguity_policy": q.ambiguity_policy,
        "requires_3d": q.requires_3d,
        "expected_targets": [_target_to_dict(t) for t in q.expected_targets],
        "paraphrases": list(q.paraphrases),
        "notes": q.notes,
        "expected_count": q.expected_count,
        "expected_yes_no": q.expected_yes_no,
    }
    return payload


def _question_from_dict(d: dict[str, Any], source_version: str) -> Question:
    targets = [
        ExpectedTarget(
            canonical_id=t["canonical_id"],
            display_label=t["display_label"],
            aliases=list(t.get("aliases", [])),
            target_kind=t.get("target_kind", "entity"),
        )
        for t in d.get("expected_targets", [])
    ]
    q = Question(
        question_id=d["question_id"],
        scene_id=d["scene_id"],
        text=d["text"],
        category=d["category"],
        answer_type=d["answer_type"],
        ambiguity_policy=d["ambiguity_policy"],
        requires_3d=bool(d["requires_3d"]),
        expected_targets=targets,
        paraphrases=list(d.get("paraphrases", [])),
        notes=d.get("notes", ""),
        expected_count=d.get("expected_count"),
        expected_yes_no=d.get("expected_yes_no"),
    )
    validate_question(q)
    return q


def load_questions(path: Path | str) -> list[Question]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    version = raw.get("schema_version")
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"schema_version {version!r} not supported; "
            f"expected one of {SUPPORTED_VERSIONS}"
        )
    if version != SCHEMA_VERSION:
        warnings.warn(
            f"Loaded questions from {path} at schema_version {version!r}; "
            f"current is {SCHEMA_VERSION!r}. v0.1 metrics from prior runs are "
            "NOT comparable to v0.2 metrics (any_of_subset semantics fixed; "
            "count and yes_no now scored). Save to promote to v0.2.",
            DeprecationWarning,
            stacklevel=2,
        )
    return [_question_from_dict(entry, version) for entry in raw["questions"]]


def save_questions(scene_id: str, questions: list[Question], path: Path | str) -> None:
    for q in questions:
        validate_question(q)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": scene_id,
        "questions": [_question_to_dict(q) for q in questions],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def to_legacy_dict(questions: list[Question]) -> dict[str, list[str]]:
    """Reduce a Question list to the v1 expected_answers.json shape:
    {query: [canonical_id, ...]}. Plug into eval_scene.py to reproduce v1
    numbers. Only entity-typed questions contribute targets; count and
    yes_no questions produce an empty list (the legacy shape can't
    represent them)."""
    return {q.text: [t.canonical_id for t in q.expected_targets] for q in questions}

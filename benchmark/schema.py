"""Benchmark schema for spatial-QA evaluation.

This module defines the question/expected-target schema used by the cross-runner
evaluator (graph / vlm / hybrid). The schema is intentionally richer than the
v1 expected_answers.json (a flat dict[query, list[label]]):

  - question_id      stable id, lets us track per-question regressions
  - paraphrases      alt phrasings of the same question
  - category         which kind of spatial reasoning the question tests
  - answer_type      what shape the answer should take
  - expected_targets structured (canonical_id + display_label + aliases + kind)
  - ambiguity_policy how to score a partial-match answer
  - requires_3d      hypothesis: can a VLM-from-photos answer this at all
  - notes            free text for the author's reasoning

A v1-compat reducer (`to_legacy_dict`) flattens a list of Questions back to the
original dict[query, list[str]] form so the existing eval_scene.py, run_benchmark,
and node_matches_expectation continue to work without modification.

Benchmark-definition note: this is a richer schema than v1's expected_answers.json.
Old evaluation_table results are not directly comparable. Re-run with
`to_legacy_dict(load_questions(...))` fed to eval_scene.py to reproduce v1
numbers bit-identically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from benchmark.categories import CATEGORIES


SCHEMA_VERSION = "v0.1"

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


def validate_question(q: Question) -> None:
    if q.category not in CATEGORIES:
        raise ValueError(f"{q.question_id}: unknown category {q.category!r}")
    if q.answer_type not in _VALID_ANSWER_TYPES:
        raise ValueError(f"{q.question_id}: unknown answer_type {q.answer_type!r}")
    if q.ambiguity_policy not in _VALID_AMBIGUITY_POLICIES:
        raise ValueError(f"{q.question_id}: unknown ambiguity_policy {q.ambiguity_policy!r}")
    if q.answer_type == "none" and q.expected_targets:
        raise ValueError(f"{q.question_id}: answer_type=none must have empty expected_targets")
    if q.answer_type != "none" and q.ambiguity_policy != "none" and not q.expected_targets:
        raise ValueError(f"{q.question_id}: missing expected_targets")
    for t in q.expected_targets:
        if t.target_kind not in _VALID_TARGET_KINDS:
            raise ValueError(f"{q.question_id}: bad target_kind {t.target_kind!r}")


def _target_to_dict(t: ExpectedTarget) -> dict[str, Any]:
    return {
        "canonical_id": t.canonical_id,
        "display_label": t.display_label,
        "aliases": list(t.aliases),
        "target_kind": t.target_kind,
    }


def _question_to_dict(q: Question) -> dict[str, Any]:
    return {
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
    }


def _question_from_dict(d: dict[str, Any]) -> Question:
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
    )
    validate_question(q)
    return q


def load_questions(path: Path | str) -> list[Question]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version {raw.get('schema_version')!r} != expected {SCHEMA_VERSION!r}"
        )
    return [_question_from_dict(entry) for entry in raw["questions"]]


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
    {query: [canonical_id, ...]}. Plug into eval_scene.py to reproduce v1 numbers."""
    return {q.text: [t.canonical_id for t in q.expected_targets] for q in questions}

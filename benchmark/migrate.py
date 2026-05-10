"""One-shot migration of legacy expected_answers.json to the new schema.

Reads:
  - <scene>/expected_answers.json  (legacy: dict[query, list[label]])
  - <scene>/scene_graph.json       (for canonical_id verification)

Writes:
  - benchmark/questions/<scene_id>.json  (rich schema)

The defaults this tool fills in (category / answer_type / ambiguity_policy /
requires_3d) are GUESSES based on parsing the question text. The output JSON
is meant to be hand-edited afterward — that's why every migrated question gets
a notes string flagging it as auto-migrated.

Idempotent: re-running with --overwrite regenerates from legacy. Without
--overwrite, refuses to clobber an existing file.

Usage:
    python -m benchmark.migrate \\
        --scene-id graffiti_bathroom \\
        --expected baselines/v1/expected_answers.json \\
        --scene-graph baselines/v1/scene_graph.json \\
        --out benchmark/questions/graffiti_bathroom.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from benchmark.schema import (
    ExpectedTarget,
    Question,
    save_questions,
)


_PHRASE_TO_CATEGORY: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^how many\b", re.I), "counting"),
    (re.compile(r"\bnear\b", re.I), "proximity"),
    (re.compile(r"\b(left|right|above|below)\b", re.I), "relative_position"),
    (re.compile(r"\b(in front of|behind)\b", re.I), "relative_position"),
    (re.compile(r"\bon the .*\bwall\b", re.I), "zone"),
    (re.compile(r"^where (is|are)\b", re.I), "zone"),
]


def _guess_category(query: str, expected: list[str]) -> str:
    if not expected:
        return "negative"
    for pattern, cat in _PHRASE_TO_CATEGORY:
        if pattern.search(query):
            return cat
    return "relative_position"


def _looks_like_zone(canonical: str) -> bool:
    return any(s in canonical for s in ("_wall", "_floor", "_zone", "_ceiling"))


def _guess_answer_type(query: str, expected: list[str]) -> str:
    if not expected:
        return "none"
    if any(_looks_like_zone(e) for e in expected):
        return "location"
    return "entity_list" if len(expected) > 1 else "entity"


def _guess_ambiguity(expected: list[str], answer_type: str) -> str:
    if not expected:
        return "none"
    if answer_type in ("entity", "location") and len(expected) == 1:
        return "one_of"
    return "all_of"


def _guess_requires_3d(category: str, expected_count: int) -> bool:
    if category in ("zone", "multi_instance", "out_of_view", "counting", "negative"):
        return True
    return expected_count > 1


def _build_target(canonical: str, scene_objects: dict[str, dict]) -> ExpectedTarget:
    if _looks_like_zone(canonical):
        return ExpectedTarget(
            canonical_id=canonical,
            display_label=canonical.replace("_", " "),
            aliases=[],
            target_kind="zone",
        )
    display = canonical.replace("_", " ")
    m = re.match(r"^(.*)_\d+$", canonical)
    if m and canonical in scene_objects:
        display = m.group(1).replace("_", " ")
    return ExpectedTarget(
        canonical_id=canonical,
        display_label=display,
        aliases=[],
        target_kind="entity",
    )


def _qid(scene_id: str, idx: int) -> str:
    short = "".join(part[:3] for part in scene_id.split("_"))[:8]
    return f"{short}_q{idx:02d}"


def migrate(
    scene_id: str,
    expected_path: Path,
    scene_graph_path: Path,
    out_path: Path,
) -> int:
    legacy = json.loads(expected_path.read_text(encoding="utf-8"))
    sg = json.loads(scene_graph_path.read_text(encoding="utf-8"))
    obj_lookup: dict[str, dict] = {str(o["label"]): o for o in sg.get("objects", [])}

    questions: list[Question] = []
    for i, (query, expected) in enumerate(legacy.items(), start=1):
        category = _guess_category(query, expected)
        answer_type = _guess_answer_type(query, expected)
        ambiguity = _guess_ambiguity(expected, answer_type)
        targets = [_build_target(e, obj_lookup) for e in expected]
        q = Question(
            question_id=_qid(scene_id, i),
            scene_id=scene_id,
            text=query,
            category=category,
            answer_type=answer_type,
            ambiguity_policy=ambiguity,
            requires_3d=_guess_requires_3d(category, len(expected)),
            expected_targets=targets,
            paraphrases=[],
            notes="auto-migrated from legacy expected_answers.json; review fields by hand",
        )
        questions.append(q)

    save_questions(scene_id, questions, out_path)
    return len(questions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--expected", required=True, type=Path)
    ap.add_argument("--scene-graph", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to clobber {args.out} (pass --overwrite)")
    n = migrate(args.scene_id, args.expected, args.scene_graph, args.out)
    print(f"wrote {n} questions to {args.out}")


if __name__ == "__main__":
    main()

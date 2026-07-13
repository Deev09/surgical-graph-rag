"""Phase 8 E2 — draft answer-key generator + human review workflow.

The "not much test data" fix: for any Replica room, run the full question
battery through the real Router and emit a DRAFT answer key a human can
review into ground truth.

A fresh draft records what the system currently says, so scoring the system
against its own draft is CIRCULAR and always passes. The draft is therefore
labeled `answer_key_type: "plausibility_labels_not_ground_truth"` and
`circular_until_reviewed: true`. It becomes evidence only after review.

Review loop (per scene, ~15-30 min with the review PNG + label table):
  1. open demo/<scene_id>_questions.png (rendered by this tool) and read the
     printed uid -> label -> centroid table;
  2. in the draft JSON: remove wrong UIDs from expected_must_contain (move
     genuinely-wrong system answers to expected_must_not_contain), flip
     expected_outcome where the system itself is wrong;
  3. set "exhaustive": true per question ONLY after checking every scene
     object that could answer it (this is what unlocks recall/P-R in E3;
     without it must_contain stays a lower bound and only membership is
     scored);
  4. fill review.status="reviewed", reviewer, date (YYYY-MM-DD), notes;
  5. set answer_key_type: "human_verified", delete circular_until_reviewed,
     and save as eval/questions/phase8/<scene_id>_qa.json (the drafts/ copy
     stays for provenance).

The key shape is a superset of eval/questions/phase7_mixed_qa.json questions,
so eval/router_qa.py scores it UNCHANGED.

Usage:
  python3 tools/draft_answer_key.py <room_dir> <scene_id> [--out DIR] [--no-png]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.question_battery import STRUCTURAL_Q, SUPPORT_Q, _runs
from graph.builder import build_graph
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer

DRAFTS_DIR = REPO_ROOT / "eval" / "questions" / "phase8" / "drafts"
MANIFEST_PATH = REPO_ROOT / "eval" / "questions" / "phase8" / "scene_manifest.json"
PLAUSIBILITY = "plausibility_labels_not_ground_truth"

_STRUCTURAL_RELATION = {
    "what is on the floor?": "SUPPORTS_FLOOR",
    "what is against the wall?": "CONTACTS_SURFACE",
    "what is near the wall?": "NEAR_SURFACE",
    "what is attached to the wall?": "ATTACHED_TO",
}
_OUTCOME_MAP = {"bindings": "answer", "empty": "empty", "abstain": "defer"}


def battery_questions() -> list[dict]:
    """The full battery with stable ids + the relation each question tests."""
    rows = []
    for i, q in enumerate(STRUCTURAL_Q + SUPPORT_Q, start=1):
        rows.append({
            "question_id": f"Q{i:02d}",
            "question": q,
            "relation": _STRUCTURAL_RELATION.get(q, "ON_ENTITY_SURFACE"),
        })
    return rows


def draft_from_bundle(bundle, labels: dict[str, str], scene_id: str,
                      compiler_name: str = "rules_v1") -> dict:
    """Pure core (unit-testable on synthetic bundles): battery -> draft key."""
    router = Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                    verbalizer=StandardVerbalizer())
    ctx = ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))
    questions = []
    for row in battery_questions():
        ans = router.answer(row["question"], bundle, ctx)
        cited = sorted(ans.cited_uids)
        questions.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "relation": row["relation"],
            "expected_outcome": _OUTCOME_MAP.get(ans.outcome, "unknown"),
            "expected_must_contain": cited,
            "expected_must_not_contain": [],
            "exhaustive": False,
            "candidate_labels": {u: labels.get(u, "?") for u in cited},
            "review": {"status": "unreviewed", "reviewer": None,
                       "date": None, "notes": ""},
        })
    return {
        "schema": "phase8_scene_qa",
        "schema_version": 1,
        "fixture_id": f"{scene_id}_phase8_qa_draft",
        "scene_id": scene_id,
        "answer_key_type": PLAUSIBILITY,
        "circular_until_reviewed": True,
        "generator": {
            "tool": "tools/draft_answer_key.py",
            "bundle_hash": bundle.bundle_hash,
            "compiler": compiler_name,
        },
        "review_instructions": "see tools/draft_answer_key.py docstring",
        "questions": questions,
    }


def _update_manifest(scene_id: str, room_dir: Path, draft_path: Path) -> None:
    manifest = {"schema": "phase8_scene_manifest", "schema_version": 1, "scenes": []}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    reviewed = f"eval/questions/phase8/{scene_id}_qa.json"
    entry = {
        "scene_id": scene_id,
        "room_dir": str(room_dir),
        "draft_path": str(draft_path.relative_to(REPO_ROOT)),
        "key_path": reviewed,
        "note": "key_path is used when it exists (human_verified); "
                "otherwise the draft scores as plausibility only",
    }
    scenes = [s for s in manifest["scenes"] if s["scene_id"] != scene_id]
    scenes.append(entry)
    manifest["scenes"] = sorted(scenes, key=lambda s: s["scene_id"])
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("room_dir", type=Path)
    parser.add_argument("scene_id")
    parser.add_argument("--out", type=Path, default=DRAFTS_DIR)
    parser.add_argument("--no-png", action="store_true",
                        help="Skip rendering the review PNG.")
    args = parser.parse_args(argv)

    if not (args.room_dir / "habitat" / "info_semantic.json").exists():
        print(f"Refusing: {args.room_dir}/habitat/info_semantic.json not found.")
        return 1

    from demo.replica_habitat_import import import_habitat_room
    arts = import_habitat_room(args.room_dir, args.scene_id)
    labels = {e.identity.object_uid: e.identity.display_label for e in arts.entities}
    bundle, _ = build_graph(arts, _runs(), density_policy="phase2_telemetry_only")

    draft = draft_from_bundle(bundle, labels, args.scene_id)
    args.out.mkdir(parents=True, exist_ok=True)
    draft_path = args.out / f"{args.scene_id}_qa_draft.json"
    draft_path.write_text(json.dumps(draft, indent=2) + "\n", encoding="utf-8")
    _update_manifest(args.scene_id, args.room_dir, draft_path)

    png_note = "(skipped)"
    if not args.no_png:
        rc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "demo" / "visualize_questions.py"),
             str(args.room_dir), args.scene_id],
            capture_output=True, text=True).returncode
        png = REPO_ROOT / "demo" / f"{args.scene_id}_questions.png"
        png_note = str(png) if rc == 0 and png.exists() else f"(render failed rc={rc})"

    answered = [q for q in draft["questions"] if q["expected_outcome"] == "answer"]
    print("=" * 78)
    print(f"DRAFT KEY: {args.scene_id}  ({len(draft['questions'])} questions, "
          f"{len(answered)} answered)")
    print(f"  draft   -> {draft_path.relative_to(REPO_ROOT)}")
    print(f"  png     -> {png_note}")
    print(f"  manifest-> {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    print("=" * 78)
    print("uid -> label -> centroid (cited objects, for review):")
    seen: set[str] = set()
    by_uid = {e.identity.object_uid: e for e in arts.entities}
    for q in answered:
        for uid in q["expected_must_contain"]:
            if uid in seen:
                continue
            seen.add(uid)
            ent = by_uid.get(uid)
            c = tuple(round(v, 2) for v in ent.centroid) if ent else "?"
            print(f"  {uid:10} {labels.get(uid, '?'):20} {c}   [{q['question_id']}]")
    print()
    print("WARNING: this draft records what the system currently answers.")
    print("Scoring against it is CIRCULAR (always passes) until a human reviews")
    print("it per the docstring and promotes it to answer_key_type=human_verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

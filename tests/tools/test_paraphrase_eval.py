"""Phase 8 E4 tests: paraphrase-eval parse-target extraction + group scoring.

Run: python tests/tools/test_paraphrase_eval.py

No dataset dependency: everything here exercises the compiler-only half
(bundle=None), which is exactly the tool's dataset-absent mode.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reasoner.compiler_rules import RulesCompiler
from tools.paraphrase_eval import parse_target, run_group


COMPILER = RulesCompiler()


def _target_of(question: str) -> dict | None:
    return parse_target(COMPILER.compile(question, None))


def test_parse_target_entity_anchor():
    t = _target_of("what is left of the table?")
    if t != {"edge_type": "LEFT_OF", "anchor_kind": "entity", "anchor": "table"}:
        raise AssertionError(f"unexpected target: {t}")


def test_parse_target_surface_anchor():
    t = _target_of("what is attached to the wall?")
    if t != {"edge_type": "ATTACHED_TO", "anchor_kind": "surface", "anchor": "wall"}:
        raise AssertionError(f"unexpected target: {t}")
    t = _target_of("what is on the floor?")
    if t != {"edge_type": "SUPPORTS", "anchor_kind": "surface", "anchor": "floor"}:
        raise AssertionError(f"unexpected floor target (SUPPORTS anchors on source): {t}")


def test_parse_target_entity_class_anchor():
    t = _target_of("what is on the table?")
    if t != {"edge_type": "SUPPORTS", "anchor_kind": "entity_class", "anchor": "table"}:
        raise AssertionError(f"unexpected target: {t}")


def test_parse_target_none_on_failures():
    if _target_of("completely unparseable gibberish") is not None:
        raise AssertionError("parser_failure should yield no parse target")
    if _target_of("what is on the ceiling fan?") is not None:
        raise AssertionError("out_of_schema should yield no parse target")


def test_run_group_counts_outcomes():
    group = {
        "group_id": "mini",
        "canonical": "what is against the wall?",
        "expected_parse": {"edge_type": "CONTACTS_SURFACE",
                           "anchor_kind": "surface", "anchor": "wall"},
        "paraphrases": [
            "which objects are against the wall?",   # expect parser_failure
            "what is against the wall",              # expect compiled (no '?')
            "what is on the wall?",                  # expect out_of_schema
        ],
    }
    out = run_group(group, COMPILER, None, None, None)
    if not out["canonical_ok"]:
        raise AssertionError("canonical 'against the wall' must compile correctly")
    s = out["summary"]
    if s["n"] != 3 or s["compiled"] != 1 or s["parser_failure"] != 1 or s["out_of_schema"] != 1:
        raise AssertionError(f"unexpected summary: {s}")
    if s["parse_target_match"] != 1:
        raise AssertionError(f"the compiled paraphrase should match the target: {s}")


def test_run_group_flags_wrong_parse():
    # "close to the wall" compiles via the generic NEAR-entity template ->
    # compiled but the WRONG parse (entity anchor, not NEAR_SURFACE).
    group = {
        "group_id": "near_wall_mini",
        "canonical": "what is near the wall?",
        "expected_parse": {"edge_type": "NEAR_SURFACE",
                           "anchor_kind": "surface", "anchor": "wall"},
        "paraphrases": ["what's close to the wall?"],
    }
    out = run_group(group, COMPILER, None, None, None)
    row = out["paraphrases"][0]
    if row["compile_outcome"] != "compiled":
        raise AssertionError(f"expected compiled, got {row['compile_outcome']}")
    if row["parse_target_match"] is not False:
        raise AssertionError(
            f"NEAR-entity fallback must be flagged as a parse mismatch: {row}")


TESTS = [
    test_parse_target_entity_anchor,
    test_parse_target_surface_anchor,
    test_parse_target_entity_class_anchor,
    test_parse_target_none_on_failures,
    test_run_group_counts_outcomes,
    test_run_group_flags_wrong_parse,
]


def main() -> int:
    failed = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
            print()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

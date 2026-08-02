"""semantics_v2 S1 guards: every v1 path must be byte-identical.

Run: python tests/tools/test_semantics_v2_guards.py

The golden bundle hash below was computed at HEAD BEFORE any S1 code
landed (commit fc643d4 state). If it drifts, an S-stage change leaked
into the frozen v1 track — that is a protocol violation, not a value to
update casually.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GOLDEN_V1_SYNTHETIC_BUNDLE_HASH = "graph_dd30b1f3cecfabd5"


def _synthetic_graph():
    from demo.question_battery import _runs
    from demo.replica_habitat_import import import_habitat_room
    from graph.builder import build_graph
    from tests.segmenter.test_c1_pipeline import PERFECT, _scene
    with tempfile.TemporaryDirectory() as td:
        room, _ = _scene(Path(td), PERFECT)
        arts = import_habitat_room(room, "synthetic_scene")
        bundle, _ = build_graph(arts, _runs(),
                                density_policy="phase2_telemetry_only")
        return bundle


def test_v1_battery_bundle_hash_unchanged():
    bundle = _synthetic_graph()
    if bundle.bundle_hash != GOLDEN_V1_SYNTHETIC_BUNDLE_HASH:
        raise AssertionError(
            f"v1 battery output drifted: {bundle.bundle_hash} != "
            f"{GOLDEN_V1_SYNTHETIC_BUNDLE_HASH} — an S-stage change leaked "
            "into the frozen track")


def test_default_compiler_vocabulary_unchanged():
    from reasoner.compiler_rules import RulesCompiler
    bundle = _synthetic_graph()
    v1 = RulesCompiler()
    for noun in ("cabinet", "nightstand", "bed"):
        cr = v1.compile(f"what is on the {noun}?", bundle)
        if cr.outcome == "compiled":
            raise AssertionError(
                f"default compiler must NOT know {noun!r} (frozen v1 "
                f"vocabulary): {cr.outcome}")
    cr = v1.compile("what is on the table?", bundle)
    if cr.outcome != "compiled":
        raise AssertionError(f"default compiler broke on 'table': {cr.outcome}")


def test_v2_compiler_accepts_d3_anchors_opt_in_only():
    from demo.semantics_v2 import make_v2_compiler
    bundle = _synthetic_graph()
    v2 = make_v2_compiler()
    for noun in ("cabinet", "nightstand", "bed", "table"):
        cr = v2.compile(f"what is on the {noun}?", bundle)
        if cr.outcome != "compiled":
            raise AssertionError(f"v2 compiler must accept {noun!r}: "
                                 f"{cr.outcome}")


def test_runs_v2_swaps_only_the_two_relations():
    from demo.question_battery import _runs
    from demo.semantics_v2 import runs_v2
    v1 = [(type(r.extractor).__name__, type(r.config).__name__)
          for r in _runs()]
    v2 = [(type(r.extractor).__name__, type(r.config).__name__)
          for r in runs_v2()]
    if len(v1) != len(v2):
        raise AssertionError(f"stack length drifted: {len(v1)} vs {len(v2)}")
    diffs = [(a, b) for a, b in zip(v1, v2) if a != b]
    expected = [
        (("OnEntitySurfaceExtractor", "OnEntitySurfaceConfig"),
         ("OnEntitySurfaceV2Extractor", "OnEntitySurfaceV2Config")),
        (("AttachedToExtractor", "AttachedToConfig"),
         ("AttachedToV2Extractor", "AttachedToV2Config")),
    ]
    if diffs != expected:
        raise AssertionError(f"runs_v2 must swap exactly the two relation "
                             f"extractors: {diffs}")


TESTS = [
    test_v1_battery_bundle_hash_unchanged,
    test_default_compiler_vocabulary_unchanged,
    test_v2_compiler_accepts_d3_anchors_opt_in_only,
    test_runs_v2_swaps_only_the_two_relations,
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

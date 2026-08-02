"""Tests for tools/c3_stage0m_measure.py (read-only measurement logic).

Run: python tests/tools/test_c3_stage0m.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.c3_stage0m_measure import BANDS, _subset_components, decide


def test_subset_components_respects_subset():
    # 5 faces in a chain 0-1-2-3-4; subset {0,1,3,4} -> two components
    offsets = np.array([0, 1, 3, 5, 7, 8])
    neighbors = np.array([1, 0, 2, 1, 3, 2, 4, 3])
    comps = _subset_components(np.array([0, 1, 3, 4]), offsets, neighbors)
    sizes = sorted(len(c) for c in comps)
    if sizes != [2, 2]:
        raise AssertionError(f"expected two 2-face components: {sizes}")
    if set(map(int, comps[0])) not in ({0, 1}, {3, 4}):
        raise AssertionError(f"wrong membership: {comps}")


def _mk_m2(cov_ok: bool, loop_ok: bool):
    return [{"bands": [{"band_m": b, "n_faces": 10, "coverage_ok": cov_ok,
                        "boundary_loop_ok": loop_ok} for b in BANDS]}]


def test_decision_rule_matrix():
    empty = {str(b): [] for b in BANDS}
    one = {str(b): [{"x": 1}] for b in BANDS}
    if decide(_mk_m2(True, True), empty)["verdict"] != "CLEAN":
        raise AssertionError("viable+loops+no impostors must be CLEAN")
    if decide(_mk_m2(True, True), one)["verdict"] != "MIXED":
        raise AssertionError("viable with impostors must be MIXED")
    if decide(_mk_m2(False, True), empty)["verdict"] != "OVERLAP":
        raise AssertionError("no viable band must be OVERLAP")
    if decide(_mk_m2(True, False), empty)["verdict"] != "OVERLAP":
        raise AssertionError("boundary failure blocks viability")


TESTS = [
    test_subset_components_respects_subset,
    test_decision_rule_matrix,
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

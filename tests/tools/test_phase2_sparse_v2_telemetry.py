"""P2.08 telemetry tests: family-only and combined sparse-v2 density.

Run: python tests/tools/test_phase2_sparse_v2_telemetry.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph.builder import SPARSE_DENSITY_LIMIT
from tools.phase2_sparse_v2_telemetry import ARTIFACT_PATH, main


def _load_payload() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def test_telemetry_artifact_has_no_timestamp_churn() -> None:
    payload = _load_payload()
    if "exported_at" in payload:
        raise AssertionError("canonical telemetry artifact should not contain exported_at")


def test_combined_density_includes_directional_and_proximity_edges() -> None:
    payload = _load_payload()
    directional = payload["directional_sparse"]["logical_edges"]
    v1 = payload["v1"]["logical_edges"]
    v2 = payload["v2"]["logical_edges"]
    combined = payload["combined_sparse_graph_before_near_surface"]
    if combined["v1_logical_edges"] != directional + v1:
        raise AssertionError("combined v1 density omitted an edge family")
    if combined["v2_logical_edges"] != directional + v2:
        raise AssertionError("combined v2 density omitted an edge family")


def test_combined_v2_reports_current_builder_guardrail_conflict() -> None:
    payload = _load_payload()
    combined = payload["combined_sparse_graph_before_near_surface"]
    if combined["v2_density_ratio_per_entity"] <= SPARSE_DENSITY_LIMIT:
        raise AssertionError("Replica fixture no longer exercises the guardrail conflict")
    if combined["v2_exceeds_phase1_sparse_density_limit"] is not True:
        raise AssertionError("combined sparse-v2 guardrail conflict was not reported")


def test_telemetry_regeneration_is_deterministic() -> None:
    before = ARTIFACT_PATH.read_text(encoding="utf-8")
    exit_code = main()
    after = ARTIFACT_PATH.read_text(encoding="utf-8")
    if exit_code != 0:
        raise AssertionError(f"telemetry regeneration failed with {exit_code}")
    if before != after:
        raise AssertionError("telemetry regeneration changed canonical artifact content")


TESTS = [
    test_telemetry_artifact_has_no_timestamp_churn,
    test_combined_density_includes_directional_and_proximity_edges,
    test_combined_v2_reports_current_builder_guardrail_conflict,
    test_telemetry_regeneration_is_deterministic,
]


def main_test() -> int:
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
    sys.exit(main_test())

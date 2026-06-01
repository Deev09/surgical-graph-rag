"""P2.09 telemetry tests: NEAR_SURFACE density accounting.

Run: python tests/tools/test_phase2_near_surface_telemetry.py
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
from tools.phase2_near_surface_telemetry import ARTIFACT_PATH, main


def _load_payload() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def test_telemetry_artifact_has_no_timestamp_churn() -> None:
    payload = _load_payload()
    if "exported_at" in payload:
        raise AssertionError("canonical telemetry artifact should not contain exported_at")


def test_near_surface_type_counts_sum_to_total() -> None:
    payload = _load_payload()
    near_surface = payload["near_surface"]
    if sum(near_surface["edges_per_surface_type"].values()) != near_surface["logical_edges"]:
        raise AssertionError("NEAR_SURFACE type counts do not sum to total")


def test_combined_density_includes_all_three_relation_families() -> None:
    payload = _load_payload()
    density = payload["combined_density_versus_phase1_guardrail"]
    directional = density["directional_sparse_logical"]
    proximity_v1 = density["proximity_v1_logical"]
    proximity_v2 = density["proximity_v2_logical"]
    near_surface = density["near_surface_logical"]
    if density["combined_v1_plus_near_surface"]["logical_edges"] != (
        directional + proximity_v1 + near_surface
    ):
        raise AssertionError("combined v1 density omitted a relation family")
    if density["combined_v2_plus_near_surface"]["logical_edges"] != (
        directional + proximity_v2 + near_surface
    ):
        raise AssertionError("combined v2 density omitted a relation family")


def test_combined_v2_reports_current_builder_guardrail_conflict() -> None:
    payload = _load_payload()
    combined = payload["combined_density_versus_phase1_guardrail"][
        "combined_v2_plus_near_surface"
    ]
    if combined["density_ratio_per_entity"] <= SPARSE_DENSITY_LIMIT:
        raise AssertionError("Replica fixture no longer exercises the guardrail conflict")
    if combined["exceeds_phase1_guardrail"] is not True:
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
    test_near_surface_type_counts_sum_to_total,
    test_combined_density_includes_all_three_relation_families,
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

"""P3.07 exit gate tests.

Reads the committed report at
scenes/replica_room_0/eval/phase3_exit_gate_report.json and asserts
all blocking gates pass + structural invariants. Also reruns the gate
to confirm:

  - the report is byte-stable across reruns (the determinism claim
    that makes "committed artifact = signal, not noise" honest);
  - the gate does NOT rewrite the committed P3.05 telemetry artifact
    (the canonical-artifacts-stay-calm rule);
  - the gate does NOT rewrite the Phase 2 exit gate report
    (Phase 3 only writes its own report; Phase 2 artifacts untouched);
  - no timestamp keys leaked into the report.

Run: python tests/tools/test_phase3_exit_gate.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph.relations.surface import (
    PLANE_MODE_VERSION,
    POLYGON_CLIPPED_VERSION,
)
from tools.phase3_exit_gate import ARTIFACT_PATH, main


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
REPLICA_V2_DIR = REPLICA_SCENE_DIR / "enriched" / "v2"
LEGACY_RELATIONS_PATH = (
    REPLICA_SCENE_DIR / "computed_relations" / "scene_graph.json"
)
TELEMETRY_ARTIFACT_PATH = (
    REPLICA_SCENE_DIR / "eval" / "phase3_polygon_clip_telemetry.json"
)
PHASE2_GATE_ARTIFACT_PATH = (
    REPLICA_SCENE_DIR / "eval" / "phase2_exit_gate_report.json"
)


def _load_payload() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def _has_inputs() -> bool:
    return (
        (REPLICA_V2_DIR / "scene_graph.json").exists()
        and LEGACY_RELATIONS_PATH.exists()
        and TELEMETRY_ARTIFACT_PATH.exists()
    )


# --- structural / shape claims on the committed artifact -----------------


def test_artifact_exists_and_is_loadable() -> None:
    if not ARTIFACT_PATH.exists():
        raise AssertionError(
            f"exit gate report missing at {ARTIFACT_PATH}; "
            "run `python tools/phase3_exit_gate.py` first"
        )
    payload = _load_payload()
    if payload.get("gate") != "phase3_exit_gate":
        raise AssertionError(f"unexpected gate value: {payload.get('gate')!r}")


def test_artifact_has_no_timestamp_keys() -> None:
    """Per CLAUDE.md: frozen artifacts must be deterministic / timestamp-free."""
    payload = _load_payload()
    forbidden = {"exported_at", "generated_at", "timestamp", "run_time", "time"}
    leaked = forbidden & set(payload.keys())
    if leaked:
        raise AssertionError(f"timestamp key(s) leaked into artifact: {leaked!r}")


def test_artifact_records_both_mode_extractor_versions() -> None:
    """The version strings recorded in the artifact must match the
    extractor's source-of-truth constants."""
    payload = _load_payload()
    versions = payload["extractor_versions"]
    if versions["plane_mode"] != PLANE_MODE_VERSION:
        raise AssertionError(
            f"plane_mode drift: {versions['plane_mode']!r} vs "
            f"{PLANE_MODE_VERSION!r}"
        )
    if versions["polygon_mode"] != POLYGON_CLIPPED_VERSION:
        raise AssertionError(
            f"polygon_mode drift: {versions['polygon_mode']!r} vs "
            f"{POLYGON_CLIPPED_VERSION!r}"
        )


# --- overall + per-gate pass claims --------------------------------------


def test_overall_blocking_pass_is_true() -> None:
    payload = _load_payload()
    if not payload["overall_blocking_pass"]:
        failed = [
            name for name, g in payload["blocking_gates"].items() if not g["pass"]
        ]
        raise AssertionError(
            f"overall_blocking_pass is False; failed gates: {failed}"
        )


def test_each_blocking_gate_passes_individually() -> None:
    payload = _load_payload()
    expected = {
        "G1_structural_surfaces",
        "G2_world_frame_obbs",
        "G3_phase1_compat_reproduction",
        "G4_deterministic_and_replayable_both_candidates",
        "G5a_phase2_smoke_under_default_config",
        "G5b_phase3_polygon_smoke_under_opt_in_config",
        "G7_builder_structural_completeness",
        "G8_polygon_clip_invariants_and_phase2_byte_equivalence",
    }
    actual = set(payload["blocking_gates"].keys())
    if actual != expected:
        raise AssertionError(
            f"blocking gate set drift: missing={expected - actual} "
            f"unexpected={actual - expected}"
        )
    for name, gate in payload["blocking_gates"].items():
        if not gate["pass"]:
            raise AssertionError(f"{name} did not pass: {gate!r}")


def test_g6_telemetry_records_both_candidates() -> None:
    """G6 is recorded, not blocking. But the artifact must record density
    for BOTH candidates so the P3.06 promotion gate (and the Phase 2
    P2.10 density discussion) has data."""
    payload = _load_payload()
    g6 = payload["telemetry_gates"]["G6_combined_density_both_candidates"]
    for label in ("plane_mode", "polygon_mode"):
        if label not in g6:
            raise AssertionError(f"G6 missing {label} block")
        if "density_ratio_per_entity" not in g6[label]:
            raise AssertionError(f"G6.{label} missing density_ratio_per_entity")


# --- G4: determinism + round-trip on both candidates ---------------------


def test_g4_both_candidates_deterministic_and_round_trippable() -> None:
    payload = _load_payload()
    g4 = payload["blocking_gates"]["G4_deterministic_and_replayable_both_candidates"]
    for label in ("plane_mode", "polygon_mode"):
        if not g4[f"{label}_two_run_hash_match"]:
            raise AssertionError(f"G4 {label} two-run hash mismatch")
        if not g4[f"{label}_round_trip_equal"]:
            raise AssertionError(f"G4 {label} round-trip not equal")


# --- G5a / G5b: smoke claims ---------------------------------------------


def test_g5a_phase2_smoke_under_default_config() -> None:
    payload = _load_payload()
    g5a = payload["blocking_gates"]["G5a_phase2_smoke_under_default_config"]
    if g5a["config_used"] != "SurfaceProximityConfig()":
        raise AssertionError(
            f"G5a config drift: {g5a['config_used']!r} (must be the default)"
        )
    if g5a["failures"]:
        raise AssertionError(f"G5a failures: {g5a['failures']!r}")


def test_g5b_phase3_smoke_under_opt_in_config() -> None:
    payload = _load_payload()
    g5b = payload["blocking_gates"]["G5b_phase3_polygon_smoke_under_opt_in_config"]
    if g5b["config_used"] != "SurfaceProximityConfig(use_polygon_clip=True)":
        raise AssertionError(
            f"G5b config drift: {g5b['config_used']!r} (must opt in)"
        )
    if g5b["failures"]:
        raise AssertionError(f"G5b failures: {g5b['failures']!r}")
    if g5b["cases_checked"]["synthetic"] < 8:
        raise AssertionError(
            f"G5b checked only {g5b['cases_checked']['synthetic']} synthetic "
            "cases; fixture should have ≥8"
        )


# --- G8: sub-claim invariants --------------------------------------------


def test_g8_all_sub_claims_present_and_true() -> None:
    payload = _load_payload()
    g8 = payload["blocking_gates"]["G8_polygon_clip_invariants_and_phase2_byte_equivalence"]
    required_true_keys = [
        "a_polygon_mode_two_runs_byte_equal",
        "b_default_equals_explicit_false_bundle_hash",
        "c_polygon_mode_bundle_hash_differs",
        "d_subset_holds_on_polygoned_surfaces",
        "e_no_not_near_to_near_flips_globally",
        "f_telemetry_artifact_present",
        "f_telemetry_artifact_agrees_with_gate",
        "g_phase3_policy_ok",
    ]
    for key in required_true_keys:
        if key not in g8:
            raise AssertionError(f"G8 missing sub-claim {key!r}")
        if not g8[key]:
            raise AssertionError(f"G8 sub-claim {key!r} is False: {g8!r}")
    if g8["f_artifact_mismatches"]:
        raise AssertionError(
            f"G8(f) telemetry mismatches: {g8['f_artifact_mismatches']!r}"
        )
    if g8["d_subset_violations"]:
        raise AssertionError(
            f"G8(d) subset violations: {g8['d_subset_violations']!r}"
        )
    if g8["e_not_near_to_near_pairs"]:
        raise AssertionError(
            f"G8(e) not_near_to_near pairs: {g8['e_not_near_to_near_pairs']!r}"
        )


def test_g8_phase3_policy_is_keep_opt_in() -> None:
    """The exit gate must encode the P3.06 closeout decision verbatim;
    a future commit that silently promotes polygon-clip to default would
    have to flip these strings, making it visible in the gate diff."""
    payload = _load_payload()
    policy = payload["blocking_gates"][
        "G8_polygon_clip_invariants_and_phase2_byte_equivalence"
    ]["g_phase3_policy"]
    if policy["default_behavior"] != "phase2_plane_mode":
        raise AssertionError(
            f"phase3_policy.default_behavior drift: {policy['default_behavior']!r}"
        )
    if policy["polygon_clip_status"] != "opt_in_candidate":
        raise AssertionError(
            f"phase3_policy.polygon_clip_status drift: "
            f"{policy['polygon_clip_status']!r}"
        )


# --- summary blocks reflect underlying claims ----------------------------


def test_phase2_byte_equivalence_summary_consistent() -> None:
    payload = _load_payload()
    summary = payload["phase2_byte_equivalence_summary"]
    g8 = payload["blocking_gates"][
        "G8_polygon_clip_invariants_and_phase2_byte_equivalence"
    ]
    if (
        summary["default_bundle_hash_equals_explicit_false"]
        != g8["b_default_equals_explicit_false_bundle_hash"]
    ):
        raise AssertionError(
            "byte_equivalence summary drifts from G8(b)"
        )
    g5a = payload["blocking_gates"]["G5a_phase2_smoke_under_default_config"]
    if summary["phase2_smoke_passes_under_default_config"] != g5a["pass"]:
        raise AssertionError(
            "byte_equivalence summary drifts from G5a"
        )


def test_polygon_candidate_readiness_summary_consistent() -> None:
    payload = _load_payload()
    summary = payload["polygon_candidate_readiness_summary"]
    g5b = payload["blocking_gates"][
        "G5b_phase3_polygon_smoke_under_opt_in_config"
    ]
    g8 = payload["blocking_gates"][
        "G8_polygon_clip_invariants_and_phase2_byte_equivalence"
    ]
    if summary["phase3_smoke_passes_under_opt_in_config"] != g5b["pass"]:
        raise AssertionError("readiness summary drifts from G5b")
    if (
        summary["polygon_mode_two_runs_byte_equal"]
        != g8["a_polygon_mode_two_runs_byte_equal"]
    ):
        raise AssertionError("readiness summary drifts from G8(a)")
    if (
        summary["subset_holds_on_polygoned_surfaces"]
        != g8["d_subset_holds_on_polygoned_surfaces"]
    ):
        raise AssertionError("readiness summary drifts from G8(d)")


# --- determinism + canonical-artifacts-stay-calm at the byte level -------


def test_telemetry_artifact_untouched_recorded_true() -> None:
    payload = _load_payload()
    if not payload["telemetry_artifact_untouched_by_gate_run"]:
        raise AssertionError(
            "telemetry_artifact_untouched_by_gate_run=False — the gate "
            "rewrote the P3.05 telemetry artifact"
        )


def test_gate_rerun_does_not_rewrite_telemetry_or_phase2_artifacts() -> None:
    """Defense-in-depth on canonical-artifacts-stay-calm. We capture the
    Phase 2 gate report bytes and P3.05 telemetry bytes BEFORE rerunning
    the Phase 3 gate, then re-read AFTER. Any drift is an immediate fail."""
    if not _has_inputs():
        print("  SKIP (enriched-v2 or legacy relations missing)")
        return
    telemetry_before = TELEMETRY_ARTIFACT_PATH.read_bytes()
    phase2_gate_before = (
        PHASE2_GATE_ARTIFACT_PATH.read_bytes()
        if PHASE2_GATE_ARTIFACT_PATH.exists()
        else None
    )
    rc = main()
    if rc != 0:
        raise AssertionError(f"phase3_exit_gate.main exited non-zero: {rc}")
    if TELEMETRY_ARTIFACT_PATH.read_bytes() != telemetry_before:
        raise AssertionError(
            "P3.05 telemetry artifact bytes drifted across Phase 3 gate run"
        )
    if phase2_gate_before is not None:
        if PHASE2_GATE_ARTIFACT_PATH.read_bytes() != phase2_gate_before:
            raise AssertionError(
                "Phase 2 exit gate report bytes drifted across Phase 3 gate run "
                "— Phase 3 gate must not touch Phase 1/2 artifacts"
            )


def test_gate_rerun_produces_byte_identical_report() -> None:
    """The Phase 3 report itself must be byte-stable on rerun. Catches
    silent non-determinism (set ordering, dict insertion order, etc.)
    before it lands in the committed artifact."""
    if not _has_inputs():
        print("  SKIP (enriched-v2 or legacy relations missing)")
        return
    before = ARTIFACT_PATH.read_bytes()
    rc = main()
    if rc != 0:
        raise AssertionError(f"phase3_exit_gate.main exited non-zero: {rc}")
    after = ARTIFACT_PATH.read_bytes()
    if before != after:
        import difflib
        before_lines = before.decode("utf-8").splitlines()
        after_lines = after.decode("utf-8").splitlines()
        diff = "\n".join(
            difflib.unified_diff(before_lines, after_lines, lineterm="", n=2)
        )
        raise AssertionError(
            f"exit gate report drifted on rerun "
            f"({len(before)} vs {len(after)} bytes); diff:\n{diff[:2000]}"
        )


TESTS = [
    test_artifact_exists_and_is_loadable,
    test_artifact_has_no_timestamp_keys,
    test_artifact_records_both_mode_extractor_versions,
    test_overall_blocking_pass_is_true,
    test_each_blocking_gate_passes_individually,
    test_g6_telemetry_records_both_candidates,
    test_g4_both_candidates_deterministic_and_round_trippable,
    test_g5a_phase2_smoke_under_default_config,
    test_g5b_phase3_smoke_under_opt_in_config,
    test_g8_all_sub_claims_present_and_true,
    test_g8_phase3_policy_is_keep_opt_in,
    test_phase2_byte_equivalence_summary_consistent,
    test_polygon_candidate_readiness_summary_consistent,
    test_telemetry_artifact_untouched_recorded_true,
    test_gate_rerun_does_not_rewrite_telemetry_or_phase2_artifacts,
    test_gate_rerun_produces_byte_identical_report,
]


def main_cli() -> int:
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
    sys.exit(main_cli())

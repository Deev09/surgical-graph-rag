"""P3.05 telemetry tests: polygon-clip monotonicity artifact.

Reads the committed artifact at
scenes/replica_room_0/eval/phase3_polygon_clip_telemetry.json and
asserts the six closeout claims:

  1. plane-mode and polygon-mode edge counts present and sane.
  2. flipped-edge counts match the edge-count delta.
  3. A6 subset claim holds: 0 violations on surfaces with polygon present;
     0 not-near-to-near flips overall (polygon mode never adds edges).
  4. A4 byte-equivalence: default vs explicit-False bundle_hash equal;
     polygon-mode bundle_hash differs (proves opt-in IS hashed).
  5. plane-mode determinism flag true; no `exported_at` / timestamp keys.
  6. Re-running the tool produces a byte-identical artifact.

Plus tolerance health-check: observed max polygon-off-plane drift is
strictly below POLYGON_ON_PLANE_TOL_M (so the validator does not
need to be touched again for this scene).

Run: python tests/tools/test_phase3_polygon_clip_telemetry.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry.surface_distance import POLYGON_ON_PLANE_TOL_M
from graph.relations.surface import (
    PLANE_MODE_VERSION,
    POLYGON_CLIPPED_VERSION,
)
from tools.phase3_polygon_clip_telemetry import ARTIFACT_PATH, main


REPLICA_V2_DIR = (
    REPO_ROOT / "scenes" / "replica_room_0" / "enriched" / "v2"
)


def _load_payload() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def _replica_v2_present() -> bool:
    return (REPLICA_V2_DIR / "scene_graph.json").exists()


# --- artifact structural / shape claims (read committed copy) -------------


def test_artifact_exists_and_is_loadable() -> None:
    if not ARTIFACT_PATH.exists():
        raise AssertionError(
            f"telemetry artifact missing at {ARTIFACT_PATH}; "
            "run `python tools/phase3_polygon_clip_telemetry.py` first"
        )
    payload = _load_payload()
    if payload.get("gate") != "phase3_polygon_clip_telemetry":
        raise AssertionError(f"unexpected gate value: {payload.get('gate')!r}")


def test_artifact_has_no_timestamp_keys() -> None:
    """Per CLAUDE.md: frozen artifacts must be deterministic / timestamp-free."""
    payload = _load_payload()
    forbidden = {"exported_at", "generated_at", "timestamp", "run_time", "time"}
    leaked = forbidden & set(payload.keys())
    if leaked:
        raise AssertionError(f"timestamp key(s) leaked into artifact: {leaked!r}")


def test_artifact_records_both_mode_extractor_versions() -> None:
    """Sanity: the version strings recorded in the artifact must match the
    extractor's source-of-truth constants."""
    payload = _load_payload()
    plane = payload["config"]["plane_mode"]
    polygon = payload["config"]["polygon_mode"]
    if plane["extractor_version"] != PLANE_MODE_VERSION:
        raise AssertionError(
            f"plane_mode.extractor_version drift: "
            f"{plane['extractor_version']!r} vs {PLANE_MODE_VERSION!r}"
        )
    if polygon["extractor_version"] != POLYGON_CLIPPED_VERSION:
        raise AssertionError(
            f"polygon_mode.extractor_version drift: "
            f"{polygon['extractor_version']!r} vs {POLYGON_CLIPPED_VERSION!r}"
        )


# --- Q1 / Q2: edge counts sane and self-consistent ------------------------


def test_edge_counts_present_and_by_type_sums_match_totals() -> None:
    payload = _load_payload()
    counts = payload["edge_counts"]
    plane_total = counts["plane_mode_total"]
    polygon_total = counts["polygon_mode_total"]
    if plane_total < 0 or polygon_total < 0:
        raise AssertionError(
            f"negative edge counts: plane={plane_total} polygon={polygon_total}"
        )
    by_type = counts["by_surface_type"]
    plane_sum = sum(v["plane"] for v in by_type.values())
    polygon_sum = sum(v["polygon"] for v in by_type.values())
    if plane_sum != plane_total:
        raise AssertionError(
            f"by_surface_type plane sum {plane_sum} != total {plane_total}"
        )
    if polygon_sum != polygon_total:
        raise AssertionError(
            f"by_surface_type polygon sum {polygon_sum} != total {polygon_total}"
        )
    if counts["delta_polygon_minus_plane"] != polygon_total - plane_total:
        raise AssertionError("delta_polygon_minus_plane is inconsistent")


# --- Q3: flipped edges accounting ---------------------------------------


def test_flipped_counts_match_edge_count_delta() -> None:
    """plane_total - polygon_total == near_to_not_near - not_near_to_near.
    Restates monotonicity in plain arithmetic: any polygon-mode edge that
    plane mode does NOT also have is a not_near_to_near flip, and any
    plane-only edge is a near_to_not_near flip."""
    payload = _load_payload()
    counts = payload["edge_counts"]
    flips = payload["flipped_edges"]
    plane_minus_polygon = counts["plane_mode_total"] - counts["polygon_mode_total"]
    flips_net = flips["near_to_not_near_count"] - flips["not_near_to_near_count"]
    if plane_minus_polygon != flips_net:
        raise AssertionError(
            f"flip accounting inconsistent: "
            f"plane-polygon={plane_minus_polygon} vs flips_net={flips_net}"
        )


# --- Q4: A6 subset claim --------------------------------------------------


def test_subset_claim_holds_on_polygoned_surfaces() -> None:
    """A6: polygon-mode edges ⊆ plane-mode edges for surfaces with polygon
    present. Violation count must be 0; if it isn't, the geometry is wrong
    and Phase 3 cannot close."""
    payload = _load_payload()
    subset = payload["subset_claim_for_surfaces_with_polygons"]
    if not subset["subset_holds"]:
        raise AssertionError(
            f"A6 subset claim violated: {subset['violation_count']} "
            f"polygon-mode edges are not in plane-mode edge set; "
            f"violators: {subset['violations']!r}"
        )
    if subset["violation_count"] != 0:
        raise AssertionError(
            f"subset_holds=True but violation_count={subset['violation_count']}"
        )
    if subset["polygon_mode_pair_count_on_polygoned_surfaces"] > (
        subset["plane_mode_pair_count_on_polygoned_surfaces"]
    ):
        raise AssertionError(
            "polygon-mode pair count on polygoned surfaces exceeds plane-mode — "
            "monotonicity violated independently of subset accounting"
        )


def test_polygon_mode_never_adds_edges() -> None:
    """Stronger restatement of the A6 dispatcher monotonicity (distance
    only grows under polygon clipping → edges can only be dropped, never
    added). Applies globally, not just to polygoned surfaces, because the
    dispatcher falls back to plane for polygon=None surfaces."""
    payload = _load_payload()
    if payload["flipped_edges"]["not_near_to_near_count"] != 0:
        raise AssertionError(
            "polygon mode emitted an edge that plane mode rejected — "
            "A6 dispatcher monotonicity is violated; this is a geometry bug"
        )


# --- Q5: A4 byte-equivalence at GraphBuilder bundle_hash level -----------


def test_default_and_explicit_false_bundle_hashes_equal() -> None:
    payload = _load_payload()
    bx = payload["phase2_byte_equivalence"]
    if not bx["default_equals_explicit_false"]:
        raise AssertionError(
            f"A4 bundle_hash equivalence violated: "
            f"default={bx['default_bundle_hash']!r} "
            f"explicit_false={bx['explicit_false_bundle_hash']!r}"
        )
    if bx["default_bundle_hash"] != bx["explicit_false_bundle_hash"]:
        raise AssertionError(
            "default_equals_explicit_false=True but hashes literally differ — "
            "artifact is internally inconsistent"
        )


def test_polygon_mode_bundle_hash_differs() -> None:
    """If polygon-mode hash matched plane-mode hash, use_polygon_clip would
    not be reaching the hash payload — defeating the whole purpose of the
    distinct extractor version."""
    payload = _load_payload()
    bx = payload["phase2_byte_equivalence"]
    if not bx["polygon_mode_differs"]:
        raise AssertionError(
            "polygon_mode_bundle_hash equals default — use_polygon_clip=True "
            "is being omitted from the hash payload"
        )


# --- Q6: determinism flag + tool-rerun byte identity --------------------


def test_determinism_flags_are_true() -> None:
    payload = _load_payload()
    det = payload["determinism"]
    if not det["plane_mode_two_runs_byte_equal"]:
        raise AssertionError("plane-mode two-run determinism failed")
    if not det["polygon_mode_two_runs_byte_equal"]:
        raise AssertionError("polygon-mode two-run determinism failed")


def test_tool_rerun_produces_byte_identical_artifact() -> None:
    """The real Q6 test: re-run the tool and assert the file is byte-equal
    to what is already on disk. Catches accidental introduction of any
    non-deterministic value (timestamps, dict-iteration order, set
    iteration, etc.) before it lands in the committed artifact."""
    if not _replica_v2_present():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    before = ARTIFACT_PATH.read_bytes()
    rc = main()
    if rc != 0:
        raise AssertionError(f"tool exited non-zero on rerun: {rc}")
    after = ARTIFACT_PATH.read_bytes()
    if before != after:
        # Surface a small diff hint to make local debugging less painful.
        import difflib
        before_lines = before.decode("utf-8").splitlines()
        after_lines = after.decode("utf-8").splitlines()
        diff = "\n".join(
            difflib.unified_diff(before_lines, after_lines, lineterm="", n=2)
        )
        raise AssertionError(
            f"artifact bytes drifted on rerun ({len(before)} vs {len(after)} "
            f"bytes); diff:\n{diff[:2000]}"
        )


# --- bonus: tolerance health check on the actual data --------------------


def test_observed_polygon_drift_within_validation_tolerance() -> None:
    """Sanity guard on the loosened tolerance from P3.05. If a future
    importer change pushes real polygon drift above the validator's
    tolerance, this test trips long before the extractor crashes in
    production telemetry."""
    payload = _load_payload()
    drift = payload["polygon_on_plane_drift"]
    observed = drift["max_polygon_off_plane_drift_m"]
    tol = drift["polygon_on_plane_tolerance_m"]
    if tol != POLYGON_ON_PLANE_TOL_M:
        raise AssertionError(
            f"artifact tolerance {tol} drifted from source-of-truth "
            f"POLYGON_ON_PLANE_TOL_M={POLYGON_ON_PLANE_TOL_M}"
        )
    if observed > tol:
        raise AssertionError(
            f"observed polygon-off-plane drift {observed} exceeds "
            f"tolerance {tol} — extractor would refuse to run on this "
            f"data in polygon mode; loosen POLYGON_ON_PLANE_TOL_M or "
            f"fix importer drift"
        )
    if not drift["within_tolerance"]:
        raise AssertionError(
            "within_tolerance=False but observed <= tol; artifact is "
            "internally inconsistent"
        )


TESTS = [
    test_artifact_exists_and_is_loadable,
    test_artifact_has_no_timestamp_keys,
    test_artifact_records_both_mode_extractor_versions,
    test_edge_counts_present_and_by_type_sums_match_totals,
    test_flipped_counts_match_edge_count_delta,
    test_subset_claim_holds_on_polygoned_surfaces,
    test_polygon_mode_never_adds_edges,
    test_default_and_explicit_false_bundle_hashes_equal,
    test_polygon_mode_bundle_hash_differs,
    test_determinism_flags_are_true,
    test_tool_rerun_produces_byte_identical_artifact,
    test_observed_polygon_drift_within_validation_tolerance,
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

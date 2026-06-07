"""P4.05 telemetry tests: ON_SURFACE coverage artifact.

Reads the committed artifact and asserts the honest-coverage invariants;
reruns the tool to prove byte identity (determinism). Guards against the
two ways "just telemetry" turns into a performance claim:
  - support_facts_total == on_surface_edges_total (clean inverse, not
    independent corroboration);
  - materialized_supports_edges_total == 0 (P4.03 invariant visible);
  - wall/ceiling 0 is flagged not-support-capable, not measured absence;
  - deferred QA items carry deferred_not_zero semantics (not empty).

Run: python tests/tools/test_phase4_on_surface_telemetry.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph.relations.on_surface import ON_SURFACE_VERSION
from tools.phase4_on_surface_telemetry import ARTIFACT_PATH, REPLICA_V2_DIR, main


def _load() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def _replica_v2_present() -> bool:
    return (REPLICA_V2_DIR / "scene_graph.json").exists()


def test_artifact_exists_and_kind() -> None:
    if not ARTIFACT_PATH.exists():
        raise AssertionError(
            f"artifact missing at {ARTIFACT_PATH}; run the tool first"
        )
    p = _load()
    if p.get("artifact_kind") != "on_surface_coverage_telemetry":
        raise AssertionError(f"artifact_kind wrong: {p.get('artifact_kind')!r}")
    if p.get("phase") != "P4.05":
        raise AssertionError(f"phase wrong: {p.get('phase')!r}")


def test_no_timestamp_keys() -> None:
    p = _load()
    forbidden = {"exported_at", "generated_at", "timestamp", "run_time", "time"}
    leaked = forbidden & set(p.keys())
    if leaked:
        raise AssertionError(f"timestamp key(s) leaked: {leaked!r}")
    if not p["determinism"]["timestamp_free"]:
        raise AssertionError("determinism.timestamp_free must be True")


def test_inputs_record_extractor_version() -> None:
    p = _load()
    if p["inputs"]["extractor_version"] != ON_SURFACE_VERSION:
        raise AssertionError(
            f"extractor_version drift: {p['inputs']['extractor_version']!r}"
        )
    if p["inputs"]["extractor"] != "on_surface":
        raise AssertionError("inputs.extractor must be 'on_surface'")
    for k in ("entity_bundle_hash", "graph_bundle_hash", "config"):
        if k not in p["inputs"]:
            raise AssertionError(f"inputs missing {k}")


def test_clean_inverse_and_no_materialized_supports() -> None:
    cov = _load()["coverage_summary"]
    if cov["support_facts_total"] != cov["on_surface_edges_total"]:
        raise AssertionError(
            "support_facts_total must equal on_surface_edges_total (clean inverse)"
        )
    if cov["materialized_supports_edges_total"] != 0:
        raise AssertionError(
            "materialized_supports_edges_total must be 0 (P4.03 invariant)"
        )


def test_by_surface_type_sums_and_floor_only() -> None:
    cov = _load()["coverage_summary"]
    bst = cov["by_surface_type"]
    total = sum(v["on_surface_edges"] for v in bst.values())
    if total != cov["on_surface_edges_total"]:
        raise AssertionError("by_surface_type edges do not sum to total")
    # floor is support-capable; wall/ceiling are not (by-design 0)
    if not bst["floor"]["support_capable"]:
        raise AssertionError("floor must be support_capable")
    for stype in ("wall", "ceiling"):
        if bst[stype]["support_capable"]:
            raise AssertionError(f"{stype} must NOT be support_capable under Design B")
        if bst[stype]["on_surface_edges"] != 0:
            raise AssertionError(
                f"{stype} on_surface_edges must be 0 (not support-capable)"
            )


def test_floor_has_coverage_on_real_data() -> None:
    if not _replica_v2_present():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    bst = _load()["coverage_summary"]["by_surface_type"]
    if bst["floor"]["on_surface_edges"] <= 0:
        raise AssertionError(
            "expected nonzero floor ON_SURFACE coverage on real Replica"
        )
    if bst["floor"]["unique_entities"] <= 0:
        raise AssertionError("expected at least one entity resting on the floor")


def test_qa_readiness_and_deferred_not_zero() -> None:
    p = _load()
    qa = p["qa_readiness"]
    if qa["what_is_on_the_floor"] != "answerable":
        raise AssertionError("floor must be answerable")
    for k in ("what_is_on_the_table", "what_is_on_the_chair"):
        if not qa[k].startswith("deferred"):
            raise AssertionError(f"{k} must be deferred; got {qa[k]!r}")
    if not qa["what_is_against_the_wall"].startswith("deferred"):
        raise AssertionError("wall must be deferred")
    ds = p["deferred_semantics"]
    if ds["deferred_not_zero"] is not True:
        raise AssertionError("deferred_not_zero must be True")
    if "not" not in ds["note"].lower() or "empty" not in ds["note"].lower():
        raise AssertionError("deferred note must state deferred != empty")


def test_interpretation_limits_disclaim_benchmark() -> None:
    limits = " ".join(_load()["interpretation_limits"]).lower()
    if "not a v1 benchmark" not in limits:
        raise AssertionError("interpretation_limits must disclaim a benchmark claim")
    if "deferred does not mean empty" not in limits:
        raise AssertionError("interpretation_limits must state deferred != empty")


def test_tool_rerun_byte_identical() -> None:
    if not _replica_v2_present():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    before = ARTIFACT_PATH.read_bytes()
    rc = main()
    if rc != 0:
        raise AssertionError(f"tool exited non-zero: {rc}")
    after = ARTIFACT_PATH.read_bytes()
    if before != after:
        import difflib
        diff = "\n".join(difflib.unified_diff(
            before.decode().splitlines(), after.decode().splitlines(),
            lineterm="", n=2,
        ))
        raise AssertionError(f"artifact drifted on rerun:\n{diff[:1500]}")


TESTS = [
    test_artifact_exists_and_kind,
    test_no_timestamp_keys,
    test_inputs_record_extractor_version,
    test_clean_inverse_and_no_materialized_supports,
    test_by_surface_type_sums_and_floor_only,
    test_floor_has_coverage_on_real_data,
    test_qa_readiness_and_deferred_not_zero,
    test_interpretation_limits_disclaim_benchmark,
    test_tool_rerun_byte_identical,
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

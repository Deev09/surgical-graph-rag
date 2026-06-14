"""P6.02 tests: OnEntitySurfaceExtractor (entity-top support).

The synthetic cases pin the rest-contact predicate against the frozen P6
fixture; the real Replica cases prove the extractor emits exactly the
frozen furniture-top positives without UID-specific logic.

Run: python tests/relations/test_on_entity_surface.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.base import ReconstructionConfig
from adapters.oracle_replica import OracleReplicaAdapter, build_replica_capture_bundle
from common.types import Plane, SceneFrame
from extractors.base import InstanceExtractorConfig
from extractors.entity_surfaces import (
    DEFAULT_SUPPORT_CLASS_ALLOWLIST,
    derive_entity_top_surfaces,
    normalize_entity_class,
)
from extractors.oracle_replica import OracleReplicaExtractor
from geometry.rest_contact import RestContactConfig, rest_contact
from graph.relations.on_entity_surface import (
    ON_ENTITY_SURFACE_VERSION,
    OnEntitySurfaceConfig,
    OnEntitySurfaceExtractor,
)
from representations.mesh import MeshRepresentation


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
REPLICA_V2_DIR = REPLICA_SCENE_DIR / "enriched" / "v2"
P6_FIXTURE_PATH = (
    REPO_ROOT / "eval" / "questions" / "phase6_entity_surface_smoke.json"
)


def _replica_v2_present() -> bool:
    return (REPLICA_V2_DIR / "scene_graph.json").exists()


def _load_fixture() -> dict:
    with P6_FIXTURE_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _rest_config_from_fixture(fixture: dict) -> RestContactConfig:
    d = fixture["config_defaults"]
    return RestContactConfig(
        contact_threshold_m=d["contact_threshold_m"],
        penetration_tolerance_m=d["penetration_tolerance_m"],
        max_tilt_deg=d["max_tilt_deg"],
        footprint_tolerance_m=d["footprint_tolerance_m"],
    )


def _relation_config_from_fixture(fixture: dict) -> OnEntitySurfaceConfig:
    d = fixture["config_defaults"]
    return OnEntitySurfaceConfig(
        contact_threshold_m=d["contact_threshold_m"],
        penetration_tolerance_m=d["penetration_tolerance_m"],
        max_tilt_deg=d["max_tilt_deg"],
        footprint_tolerance_m=d["footprint_tolerance_m"],
        support_class_allowlist=tuple(fixture["support_class_allowlist"]),
    )


def _plane(d: dict) -> Plane:
    return Plane(a=d["a"], b=d["b"], c=d["c"], d=d["d"])


def _aabb(case: dict):
    mn = case["entity_aabb"]["min"]
    mx = case["entity_aabb"]["max"]
    return ((mn[0], mn[1], mn[2]), (mx[0], mx[1], mx[2]))


def _build_replica_artifacts():
    capture = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    representation = MeshRepresentation(
        bundle=OracleReplicaAdapter().reconstruct(
            capture,
            ReconstructionConfig(name="oracle_replica", version="0.1", params={}),
        ),
    )
    return OracleReplicaExtractor(enriched_v2_path=REPLICA_V2_DIR).extract(
        representation,
        InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )


def test_config_defaults_match_fixture() -> None:
    fixture = _load_fixture()
    d = fixture["config_defaults"]
    cfg = OnEntitySurfaceConfig()
    if (
        cfg.contact_threshold_m != d["contact_threshold_m"]
        or cfg.penetration_tolerance_m != d["penetration_tolerance_m"]
        or cfg.max_tilt_deg != d["max_tilt_deg"]
        or cfg.footprint_tolerance_m != d["footprint_tolerance_m"]
        or tuple(fixture["support_class_allowlist"]) != DEFAULT_SUPPORT_CLASS_ALLOWLIST
    ):
        raise AssertionError("OnEntitySurfaceConfig defaults drifted from fixture")


def test_label_normalization_keeps_hyphenated_classes() -> None:
    cases = {
        "table_5": "table",
        "plant-stand_1": "plant-stand",
        "plant stand 1": "plant-stand",
        "indoor-plant_1": "indoor-plant",
    }
    for raw, expected in cases.items():
        got = normalize_entity_class(raw)
        if got != expected:
            raise AssertionError(f"{raw!r} -> {got!r}, expected {expected!r}")


def test_synthetic_rest_contact_fixture_cases() -> None:
    fixture = _load_fixture()
    surfaces = fixture["synthetic_surfaces"]
    cfg = _rest_config_from_fixture(fixture)
    gravity = (0.0, 0.0, -1.0)
    failures: list[str] = []
    checked = 0
    for case in fixture["cases"]:
        if not case["synthetic"]:
            continue
        checked += 1
        surface = surfaces[case["surface_ref"]]
        result = rest_contact(
            _aabb(case),
            tuple(case["entity_centroid"]),
            _plane(surface["plane"]),
            [(v[0], v[1], v[2]) for v in surface["polygon"]],
            gravity,
            cfg,
        )
        if result.on_surface != case["expected_on_entity_surface"]:
            failures.append(
                f"{case['id']}: on_surface={result.on_surface} "
                f"expected {case['expected_on_entity_surface']}"
            )
        if result.failed_clauses != case["expected_failed_clauses"]:
            failures.append(
                f"{case['id']}: failed_clauses={result.failed_clauses} "
                f"expected {case['expected_failed_clauses']}"
            )
    if checked < 7:
        raise AssertionError(f"expected >=7 synthetic cases, checked {checked}")
    if failures:
        raise AssertionError("\n".join(failures))


def test_real_replica_edges_match_frozen_fixture() -> None:
    if not _replica_v2_present():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    fixture = _load_fixture()
    artifacts = _build_replica_artifacts()
    edges, diag = OnEntitySurfaceExtractor().extract(
        artifacts, _relation_config_from_fixture(fixture)
    )
    pairs = {(e.source.uid, e.target.uid) for e in edges}
    # Exact frozen P6 positive set: five table-supported objects plus one
    # plant-on-plant-stand. The smoke fixture samples three of these; the
    # full table answer set is also pinned in phase6_mixed_qa.json.
    expected_positive = {
        ("obj_92", "obj_10"),
        ("obj_90", "obj_93"),
        ("obj_12", "obj_11"),
        ("obj_59", "obj_11"),
        ("obj_87", "obj_93"),
        ("obj_35", "obj_55"),
    }
    if pairs != expected_positive:
        raise AssertionError(f"real ON_ENTITY_SURFACE pairs drifted: {sorted(pairs)}")
    if diag.physical_edges_per_type.get("ON_ENTITY_SURFACE") != len(expected_positive):
        raise AssertionError("diagnostic edge count drifted")
    for edge in edges:
        if edge.type != "ON_ENTITY_SURFACE":
            raise AssertionError(f"wrong edge type {edge.type!r}")
        if edge.source.kind != "entity" or edge.target.kind != "entity":
            raise AssertionError(f"wrong endpoint kinds {edge.source}->{edge.target}")
        if edge.extractor_version != ON_ENTITY_SURFACE_VERSION:
            raise AssertionError("extractor version not recorded")
        if edge.target.uid != edge.evidence.get("owner_entity_uid"):
            raise AssertionError("target/evidence owner invariant violated")
        if "entity_surface_uid" not in edge.evidence:
            raise AssertionError("missing entity_surface_uid evidence")


def test_real_replica_negatives_and_supported_class_filter() -> None:
    if not _replica_v2_present():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    artifacts = _build_replica_artifacts()
    edges, diag = OnEntitySurfaceExtractor().extract(
        artifacts, OnEntitySurfaceConfig()
    )
    pairs = {(e.source.uid, e.target.uid) for e in edges}
    if ("obj_43", "obj_11") in pairs:
        raise AssertionError("pot obj_43 must remain band-excluded")
    if ("obj_55", "obj_11") in pairs:
        raise AssertionError("plant-stand obj_55 must be excluded as supported furniture")
    if diag.rejections_per_type.get("ON_ENTITY_SURFACE", 0) <= len(edges):
        raise AssertionError("expected rejected entity/supporter pairs in diagnostics")


def test_derivation_does_not_persist_structural_surfaces() -> None:
    if not _replica_v2_present():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    artifacts = _build_replica_artifacts()
    before = [s.surface_uid for s in artifacts.structural_surfaces]
    derived = derive_entity_top_surfaces(
        artifacts.entities,
        frame=artifacts.frame,
        support_class_allowlist=DEFAULT_SUPPORT_CLASS_ALLOWLIST,
    )
    after = [s.surface_uid for s in artifacts.structural_surfaces]
    if before != after:
        raise AssertionError("entity surface derivation mutated structural_surfaces")
    if not any(s.owner_class == "table" for s in derived):
        raise AssertionError("expected derived table surfaces")
    if not any(s.owner_class == "plant-stand" for s in derived):
        raise AssertionError("expected derived plant-stand surface")


TESTS = [
    test_config_defaults_match_fixture,
    test_label_normalization_keeps_hyphenated_classes,
    test_synthetic_rest_contact_fixture_cases,
    test_real_replica_edges_match_frozen_fixture,
    test_real_replica_negatives_and_supported_class_filter,
    test_derivation_does_not_persist_structural_surfaces,
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

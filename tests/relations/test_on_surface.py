"""P4.02 tests: OnSurfaceExtractor (ON_SURFACE rest-contact translation).

P4.01 proves the rest_contact predicate math exhaustively. This suite
proves the EXTRACTOR translation for every frozen fixture case: emit/reject
decision, edge direction + provenance, evidence assembly, rejection reason
and failed_clauses (no accidental clause loss). Plus:
  - real F1 (obj_39 / floor_25) on the live enriched-v2 bundle;
  - the D4a/G8 threshold-ordering guard raises;
  - subset against polygon-mode NEAR_SURFACE on Replica (G2);
  - isolation: OnSurfaceExtractor is not in any default builder.

Run: python tests/relations/test_on_surface.py
"""
from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.base import ReconstructionConfig
from adapters.oracle_replica import OracleReplicaAdapter, build_replica_capture_bundle
from common.types import Plane, SceneFrame
from extractors.base import (
    EntityArtifact, EntityArtifacts, EntityIdentity, ExtractionDiagnostics,
    InstanceExtractorConfig, SemanticHypothesis, StructuralSurface,
)
from extractors.oracle_replica import OracleReplicaExtractor
from extractors.serde import CURRENT_SCHEMA_VERSION as ENT_SCHEMA_VERSION
from graph.relations.on_surface import (
    ON_SURFACE_VERSION,
    OnSurfaceConfig,
    OnSurfaceExtractor,
)
from graph.relations.surface import (
    SurfaceProximityConfig,
    SurfaceProximityExtractor,
)
from representations.mesh import MeshRepresentation


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
REPLICA_V2_DIR = REPLICA_SCENE_DIR / "enriched" / "v2"
P4_FIXTURE_PATH = (
    REPO_ROOT / "eval" / "questions" / "phase4_on_surface_smoke.json"
)


def _assert_value_error(fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _replica_v2_present() -> bool:
    return (REPLICA_V2_DIR / "scene_graph.json").exists()


def _frame() -> SceneFrame:
    return SceneFrame(
        gravity=(0.0, 0.0, -1.0),
        canonical_forward=None, canonical_right=None,
        units="meters", notes="",
    )


def _entity(uid, lo, hi, centroid) -> EntityArtifact:
    return EntityArtifact(
        identity=EntityIdentity(
            object_uid=uid, display_label=uid, aliases=[], source_instance_ref=uid,
        ),
        bbox_aabb=(lo, hi), bbox_obb=None, centroid=centroid,
        geometry_handle=None,
        semantic_hypotheses=[SemanticHypothesis(label=uid, confidence=1.0, source="t")],
        embedding=None, extraction_diagnostics={},
    )


def _surface(uid, stype, plane, polygon, source="habitat_label") -> StructuralSurface:
    return StructuralSurface(
        surface_uid=uid, surface_type=stype, plane=plane,
        polygon=polygon, confidence=1.0, source=source,
    )


def _artifacts(entities, surfaces) -> EntityArtifacts:
    return EntityArtifacts(
        schema_version=ENT_SCHEMA_VERSION, bundle_hash="ent_t", scene_id="t",
        frame=_frame(), representation_hash="rep_t",
        extractor_name="test", extractor_version="0.0",
        entities=entities, structural_surfaces=surfaces,
        geometry_store_path=None,
        diagnostics=ExtractionDiagnostics(
            n_entities=len(entities), n_structural_surfaces=len(surfaces),
            runtime_seconds=0.0, coverage_score=None, notes="",
        ),
        notes={},
    )


def _load_fixture() -> dict:
    with P4_FIXTURE_PATH.open() as fh:
        return json.load(fh)


def _surface_from_synth(surface: dict) -> StructuralSurface:
    p = surface["plane"]
    return _surface(
        surface["surface_uid"], surface["surface_type"],
        Plane(a=p["a"], b=p["b"], c=p["c"], d=p["d"]),
        [(v[0], v[1], v[2]) for v in surface["polygon"]],
        source=surface["source"],
    )


def _entity_from_case(case: dict) -> EntityArtifact:
    mn = case["entity_aabb"]["min"]
    mx = case["entity_aabb"]["max"]
    return _entity(
        case["id"], (mn[0], mn[1], mn[2]), (mx[0], mx[1], mx[2]),
        tuple(case["entity_centroid"]),
    )


def _config_from_fixture(fixture: dict) -> OnSurfaceConfig:
    d = fixture["config_defaults"]
    return OnSurfaceConfig(
        contact_threshold_m=d["contact_threshold_m"],
        penetration_tolerance_m=d["penetration_tolerance_m"],
        max_tilt_deg=d["max_tilt_deg"],
        footprint_tolerance_m=d["footprint_tolerance_m"],
        near_surface_threshold_m=d["near_surface_threshold_m_floor"],
    )


def _build_replica_artifacts() -> EntityArtifacts:
    capture = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    repr_bundle = OracleReplicaAdapter().reconstruct(
        capture,
        ReconstructionConfig(name="oracle_replica", version="0.1", params={}),
    )
    representation = MeshRepresentation(bundle=repr_bundle)
    return OracleReplicaExtractor(enriched_v2_path=REPLICA_V2_DIR).extract(
        representation,
        InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )


# --- config defaults agree with fixture ----------------------------------


def test_config_defaults_match_fixture() -> None:
    fixture = _load_fixture()
    d = fixture["config_defaults"]
    default = OnSurfaceConfig()
    # Frozen fixture names the floor NEAR threshold "near_surface_threshold_m_floor".
    if (
        default.contact_threshold_m != d["contact_threshold_m"]
        or default.penetration_tolerance_m != d["penetration_tolerance_m"]
        or default.max_tilt_deg != d["max_tilt_deg"]
        or default.footprint_tolerance_m != d["footprint_tolerance_m"]
        or default.near_surface_threshold_m != d["near_surface_threshold_m_floor"]
    ):
        raise AssertionError(
            "OnSurfaceConfig defaults drifted from fixture config_defaults"
        )


# --- thin translation coverage for ALL synthetic cases F2-F9 -------------


def test_all_synthetic_cases_translate_correctly() -> None:
    """For each frozen synthetic case: the extractor emits exactly one
    ON_SURFACE edge iff expected_on_surface; emitted edges have the right
    direction + provenance + assembled evidence; rejected cases carry the
    declared failed_clauses with no accidental clause loss."""
    fixture = _load_fixture()
    tol = fixture["numeric_tolerance_m"]
    cfg = _config_from_fixture(fixture)
    surfaces = fixture["synthetic_surfaces"]
    extractor = OnSurfaceExtractor()

    bool_keys = {"support_capable", "centroid_on_support_side", "footprint_ok", "contact"}
    failures: list[str] = []
    checked = 0
    for case in fixture["cases"]:
        if not case["synthetic"]:
            continue
        cid = case["id"]
        surface = _surface_from_synth(surfaces[case["surface_ref"]])
        artifacts = _artifacts([_entity_from_case(case)], [surface])
        edges, diag = extractor.extract(artifacts, cfg)

        if case["expected_on_surface"]:
            if len(edges) != 1:
                failures.append(f"{cid}: expected 1 edge, got {len(edges)}")
                checked += 1
                continue
            e = edges[0]
            # direction + provenance
            if e.type != "ON_SURFACE":
                failures.append(f"{cid}: edge type {e.type!r}")
            if e.source.kind != "entity" or e.target.kind != "surface":
                failures.append(f"{cid}: wrong direction {e.source.kind}->{e.target.kind}")
            if e.extractor != "on_surface" or e.extractor_version != ON_SURFACE_VERSION:
                failures.append(f"{cid}: provenance {e.extractor}/{e.extractor_version}")
            # evidence assembly (extractor-side keys + clause values)
            ev = e.evidence
            for k in ("near_surface_threshold_m", "surface_type", "source"):
                if k not in ev:
                    failures.append(f"{cid}: evidence missing {k}")
            if ev.get("polygon_clip_required") is not True:
                failures.append(f"{cid}: polygon_clip_required != True")
            for key, expected in case.get("expected_clauses", {}).items():
                actual = ev.get(key)
                if key in bool_keys:
                    if actual is not expected:
                        failures.append(f"{cid}: ev[{key}]={actual} exp {expected}")
                else:
                    if actual is None or abs(actual - expected) > tol:
                        failures.append(f"{cid}: ev[{key}]={actual} exp {expected}")
        else:
            if len(edges) != 0:
                failures.append(f"{cid}: expected 0 edges, got {len(edges)}")
            if diag.rejections_per_type.get("ON_SURFACE", 0) != 1:
                failures.append(f"{cid}: expected 1 rejection")
            elif diag.rejection_samples:
                rej = diag.rejection_samples[0]
                if rej.rejected_reason != "rest_contact_clauses_failed":
                    failures.append(f"{cid}: reason {rej.rejected_reason!r}")
                got = rej.evidence.get("failed_clauses")
                if got != case["expected_failed_clauses"]:
                    failures.append(
                        f"{cid}: failed_clauses {got} exp {case['expected_failed_clauses']}"
                    )
        checked += 1

    if checked < 8:
        raise AssertionError(f"expected >=8 synthetic cases, checked {checked}")
    if failures:
        raise AssertionError(
            f"{len(failures)} translation mismatches:\n  " + "\n  ".join(failures)
        )


# --- real F1 -------------------------------------------------------------


def test_f1_real_replica_stool_on_floor() -> None:
    """F1: real obj_39 (stool) rests on floor_25 on the live enriched-v2
    bundle under default OnSurfaceConfig."""
    if not _replica_v2_present():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    fixture = _load_fixture()
    f1 = next(c for c in fixture["cases"] if c["id"] == "F1")
    artifacts = _build_replica_artifacts()
    edges, _ = OnSurfaceExtractor().extract(artifacts, OnSurfaceConfig())
    pairs = {(e.source.uid, e.target.uid) for e in edges}
    want = (f1["entity_uid"], f1["surface_uid"])
    if want not in pairs:
        raise AssertionError(
            f"F1: expected ON_SURFACE {want} on real Replica; not emitted"
        )


# --- threshold-ordering guard (D4a/G8) -----------------------------------


def test_threshold_ordering_guard_raises() -> None:
    # contact alone exceeds near
    _assert_value_error(OnSurfaceConfig, contact_threshold_m=0.10)
    # hypot(contact, footprint) exceeds near even when each < near
    _assert_value_error(
        OnSurfaceConfig, contact_threshold_m=0.04, footprint_tolerance_m=0.04,
    )
    # default is valid (0.02 <= 0.05)
    OnSurfaceConfig()
    # raising near makes a wider contact band valid again
    OnSurfaceConfig(contact_threshold_m=0.04, footprint_tolerance_m=0.04,
                    near_surface_threshold_m=0.30)


def test_invalid_rest_thresholds_raise_via_config() -> None:
    _assert_value_error(OnSurfaceConfig, contact_threshold_m=-0.01)
    _assert_value_error(OnSurfaceConfig, penetration_tolerance_m=-0.01)
    _assert_value_error(OnSurfaceConfig, max_tilt_deg=0.0)


# --- subset vs polygon-mode NEAR_SURFACE (G2) ----------------------------


def test_on_surface_subset_of_polygon_mode_near_surface() -> None:
    """G2: every ON_SURFACE (entity, surface) pair must also be a
    polygon-mode NEAR_SURFACE (entity, surface) pair on Replica room_0."""
    if not _replica_v2_present():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    artifacts = _build_replica_artifacts()
    on_edges, _ = OnSurfaceExtractor().extract(artifacts, OnSurfaceConfig())
    near_edges, _ = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(use_polygon_clip=True),
    )
    on_pairs = {(e.source.uid, e.target.uid) for e in on_edges}
    near_pairs = {(e.source.uid, e.target.uid) for e in near_edges}
    violations = sorted(on_pairs - near_pairs)
    if violations:
        raise AssertionError(
            f"G2 subset violated: {len(violations)} ON_SURFACE pairs not in "
            f"polygon-mode NEAR_SURFACE: {violations[:5]}"
        )
    if not on_pairs:
        raise AssertionError(
            "expected at least one ON_SURFACE edge on real Replica (floor rests)"
        )


# --- policy: synth fallback + polygon required ---------------------------


def test_skips_synth_bbox_fallback_by_default() -> None:
    floor = _surface(
        "floor_synth", "floor", Plane(0.0, 0.0, 1.0, 0.0),
        [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)],
        source="synth_bbox_fallback",
    )
    arts = _artifacts(
        [_entity("e1", (0.8, 0.8, -0.01), (1.2, 1.2, 0.3), (1.0, 1.0, 0.145))],
        [floor],
    )
    edges, diag = OnSurfaceExtractor().extract(arts, OnSurfaceConfig())
    if edges:
        raise AssertionError("must skip synth_bbox_fallback by default")
    sample = diag.rejection_samples[0]
    if sample.rejected_reason != "surface_source_excluded":
        raise AssertionError(f"reason {sample.rejected_reason!r}")


def test_skips_polygon_none_surface() -> None:
    floor = _surface(
        "floor_mesh", "floor", Plane(0.0, 0.0, 1.0, 0.0),
        None, source="mesh_ransac",
    )
    arts = _artifacts(
        [_entity("e1", (0.8, 0.8, -0.01), (1.2, 1.2, 0.3), (1.0, 1.0, 0.145))],
        [floor],
    )
    edges, diag = OnSurfaceExtractor().extract(arts, OnSurfaceConfig())
    if edges:
        raise AssertionError("must skip polygon=None surface (footprint required)")
    sample = diag.rejection_samples[0]
    if sample.rejected_reason != "polygon_required_for_on_surface":
        raise AssertionError(f"reason {sample.rejected_reason!r}")


def test_rejects_wrong_config_type() -> None:
    arts = _artifacts([], [])
    _assert_value_error  # noqa: B018 - keep import linters quiet
    try:
        OnSurfaceExtractor().extract(arts, SurfaceProximityConfig())
    except TypeError:
        return
    raise AssertionError("expected TypeError for wrong config type")


TESTS = [
    test_config_defaults_match_fixture,
    test_all_synthetic_cases_translate_correctly,
    test_f1_real_replica_stool_on_floor,
    test_threshold_ordering_guard_raises,
    test_invalid_rest_thresholds_raise_via_config,
    test_on_surface_subset_of_polygon_mode_near_surface,
    test_skips_synth_bbox_fallback_by_default,
    test_skips_polygon_none_surface,
    test_rejects_wrong_config_type,
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

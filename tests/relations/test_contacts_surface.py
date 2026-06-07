"""P5.02 tests: ContactsSurfaceExtractor (wall-contact translation).

P5.01 proves the wall_contact predicate math. This suite proves the
EXTRACTOR translation for the frozen fixture cases (emit/reject, edge
direction + provenance, evidence assembly incl. up->list, rejection reasons),
the real W1 positive + WN negatives, the G2-analog subset vs polygon-mode
NEAR_SURFACE, the wall-only / synth-fallback / polygon-none policies, and
that ON_SURFACE-style up evidence survives dump/load.

Run: python tests/relations/test_contacts_surface.py
"""
from __future__ import annotations

import json
import sys
import tempfile
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
from graph.builder import ExtractorRun, build_graph
from graph.relations.contacts_surface import (
    CONTACTS_SURFACE_VERSION,
    ContactsSurfaceConfig,
    ContactsSurfaceExtractor,
)
from graph.relations.surface import (
    SurfaceProximityConfig, SurfaceProximityExtractor,
)
from graph.serde import dump_scene_graph_bundle, load_scene_graph_bundle
from representations.mesh import MeshRepresentation


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
REPLICA_V2_DIR = REPLICA_SCENE_DIR / "enriched" / "v2"
P5_FIXTURE_PATH = (
    REPO_ROOT / "eval" / "questions" / "phase5_wall_contact_smoke.json"
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
        gravity=(0.0, 0.0, -1.0), canonical_forward=None,
        canonical_right=None, units="meters", notes="",
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
    with P5_FIXTURE_PATH.open() as fh:
        return json.load(fh)


def _synth_surface(fixture, ref) -> StructuralSurface:
    s = fixture["synthetic_surfaces"][ref]
    p = s["plane"]
    return _surface(
        s["surface_uid"], s["surface_type"],
        Plane(a=p["a"], b=p["b"], c=p["c"], d=p["d"]),
        [(v[0], v[1], v[2]) for v in s["polygon"]],
        source=s["source"],
    )


def _entity_from_case(case) -> EntityArtifact:
    mn = case["entity_aabb"]["min"]
    mx = case["entity_aabb"]["max"]
    return _entity(
        case["id"], (mn[0], mn[1], mn[2]), (mx[0], mx[1], mx[2]),
        tuple(case["entity_centroid"]),
    )


def _config_from_fixture(fixture) -> ContactsSurfaceConfig:
    d = fixture["config_defaults"]
    return ContactsSurfaceConfig(
        contact_threshold_m=d["contact_threshold_m"],
        penetration_tolerance_m=d["penetration_tolerance_m"],
        max_wall_tilt_deg=d["max_wall_tilt_deg"],
        footprint_tolerance_m=d["footprint_tolerance_m"],
        near_surface_threshold_m=d["near_surface_threshold_m_wall"],
    )


def _build_replica_artifacts() -> EntityArtifacts:
    capture = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    repr_bundle = OracleReplicaAdapter().reconstruct(
        capture, ReconstructionConfig(name="oracle_replica", version="0.1", params={}),
    )
    representation = MeshRepresentation(bundle=repr_bundle)
    return OracleReplicaExtractor(enriched_v2_path=REPLICA_V2_DIR).extract(
        representation,
        InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )


# --- config defaults agree with fixture ----------------------------------


def test_config_defaults_match_fixture() -> None:
    d = _load_fixture()["config_defaults"]
    c = ContactsSurfaceConfig()
    if (
        c.contact_threshold_m != d["contact_threshold_m"]
        or c.penetration_tolerance_m != d["penetration_tolerance_m"]
        or c.max_wall_tilt_deg != d["max_wall_tilt_deg"]
        or c.footprint_tolerance_m != d["footprint_tolerance_m"]
        or c.near_surface_threshold_m != d["near_surface_threshold_m_wall"]
    ):
        raise AssertionError("ContactsSurfaceConfig defaults drifted from fixture")


# --- synthetic WS1-WS6 translation (WS5 = non-wall policy) ----------------


def test_synthetic_cases_translate() -> None:
    fixture = _load_fixture()
    tol = fixture["numeric_tolerance_m"]
    cfg = _config_from_fixture(fixture)
    extractor = ContactsSurfaceExtractor()
    bool_keys = {"wall_capable", "on_interior_side", "footprint_ok", "contact"}
    failures: list[str] = []
    checked = 0
    for case in fixture["cases"]:
        if not case["synthetic"]:
            continue
        cid = case["id"]
        surface = _synth_surface(fixture, case["surface_ref"])
        arts = _artifacts([_entity_from_case(case)], [surface])
        edges, diag = extractor.extract(arts, cfg)

        if surface.surface_type != "wall":
            # WS5: floor surface -> wall-only policy skip (not evaluated)
            if edges:
                failures.append(f"{cid}: non-wall surface must yield 0 edges")
            sample = diag.rejection_samples[0] if diag.rejection_samples else None
            if sample is None or sample.rejected_reason != "surface_type_not_wall":
                failures.append(f"{cid}: expected surface_type_not_wall skip")
            checked += 1
            continue

        if case["expected_contacts_surface"]:
            if len(edges) != 1:
                failures.append(f"{cid}: expected 1 edge, got {len(edges)}")
            else:
                e = edges[0]
                if e.type != "CONTACTS_SURFACE":
                    failures.append(f"{cid}: type {e.type!r}")
                if e.source.kind != "entity" or e.target.kind != "surface":
                    failures.append(f"{cid}: wrong direction")
                if e.extractor != "contacts_surface" or e.extractor_version != CONTACTS_SURFACE_VERSION:
                    failures.append(f"{cid}: provenance")
                if not isinstance(e.evidence.get("up"), list):
                    failures.append(f"{cid}: up must be a list (serde-safe)")
                if e.evidence.get("polygon_clip_required") is not True:
                    failures.append(f"{cid}: polygon_clip_required")
                for k, exp in case.get("expected_clauses", {}).items():
                    act = e.evidence.get(k)
                    if k in bool_keys:
                        if act is not exp:
                            failures.append(f"{cid}: ev[{k}]={act} exp {exp}")
                    else:
                        if act is None or abs(act - exp) > tol:
                            failures.append(f"{cid}: ev[{k}]={act} exp {exp}")
        else:
            if edges:
                failures.append(f"{cid}: expected 0 edges, got {len(edges)}")
            sample = diag.rejection_samples[0] if diag.rejection_samples else None
            if sample is None or sample.rejected_reason != "wall_contact_clauses_failed":
                failures.append(f"{cid}: expected wall_contact_clauses_failed")
            elif sample.evidence.get("failed_clauses") != case["expected_failed_clauses"]:
                failures.append(
                    f"{cid}: failed_clauses {sample.evidence.get('failed_clauses')} "
                    f"exp {case['expected_failed_clauses']}"
                )
        checked += 1
    if checked < 6:
        raise AssertionError(f"expected >=6 synthetic cases, checked {checked}")
    if failures:
        raise AssertionError(
            f"{len(failures)} mismatches:\n  " + "\n  ".join(failures)
        )


# --- real W1 positive + WN negatives -------------------------------------


def test_real_w1_positive_and_wn_negatives() -> None:
    if not _replica_v2_present():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    fixture = _load_fixture()
    artifacts = _build_replica_artifacts()
    edges, _ = ContactsSurfaceExtractor().extract(artifacts, ContactsSurfaceConfig())
    pairs = {(e.source.uid, e.target.uid) for e in edges}
    emitted_entities = {e.source.uid for e in edges}

    w1 = next(c for c in fixture["cases"] if c["id"] == "W1")
    if (w1["entity_uid"], w1["surface_uid"]) not in pairs:
        raise AssertionError(f"W1 {w1['entity_uid']}/{w1['surface_uid']} not emitted")

    # WN1/WN2/WN3 negatives must NOT be in the wall-contact set
    for nid in ("WN1", "WN2", "WN3"):
        c = next(x for x in fixture["cases"] if x["id"] == nid)
        if c["entity_uid"] in emitted_entities:
            # only fail if it is emitted against the SAME surface
            if (c["entity_uid"], c["surface_uid"]) in pairs:
                raise AssertionError(
                    f"{nid} {c['entity_uid']} wrongly emitted as wall contact"
                )


def test_only_lamp_is_a_wall_contact_on_replica() -> None:
    """Under the frozen 0.02/0.02 band, the only real wall contact is obj_6
    (lamp). Pins the precision-over-recall result."""
    if not _replica_v2_present():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    edges, _ = ContactsSurfaceExtractor().extract(
        _build_replica_artifacts(), ContactsSurfaceConfig(),
    )
    entities = {e.source.uid for e in edges}
    if entities != {"obj_6"}:
        raise AssertionError(
            f"expected exactly {{obj_6}} as wall contacts; got {sorted(entities)}"
        )


# --- subset vs polygon-mode NEAR_SURFACE ---------------------------------


def test_subset_of_polygon_near_surface() -> None:
    if not _replica_v2_present():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    artifacts = _build_replica_artifacts()
    cs_edges, _ = ContactsSurfaceExtractor().extract(artifacts, ContactsSurfaceConfig())
    near_edges, _ = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(use_polygon_clip=True),
    )
    cs_pairs = {(e.source.uid, e.target.uid) for e in cs_edges}
    near_pairs = {(e.source.uid, e.target.uid) for e in near_edges}
    violations = sorted(cs_pairs - near_pairs)
    if violations:
        raise AssertionError(
            f"subset violated: {len(violations)} CONTACTS_SURFACE pairs not in "
            f"polygon-mode NEAR_SURFACE: {violations[:5]}"
        )


# --- policy skips --------------------------------------------------------


def test_skips_non_wall_surface() -> None:
    floor = _surface("floor_1", "floor", Plane(0.0, 0.0, 1.0, 0.0),
                     [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)])
    arts = _artifacts([_entity("e1", (0.8, 0.8, -0.01), (1.2, 1.2, 0.3), (1.0, 1.0, 0.145))], [floor])
    edges, diag = ContactsSurfaceExtractor().extract(arts, ContactsSurfaceConfig())
    if edges:
        raise AssertionError("floor surface must be skipped (wall-only)")
    if diag.rejection_samples[0].rejected_reason != "surface_type_not_wall":
        raise AssertionError("expected surface_type_not_wall")


def test_skips_synth_bbox_fallback() -> None:
    wall = _surface("wall_synth", "wall", Plane(0.0, -1.0, 0.0, 2.0),
                    [(0, 2, 0), (2, 2, 0), (2, 2, 2), (0, 2, 2)],
                    source="synth_bbox_fallback")
    arts = _artifacts([_entity("e1", (0.8, 1.85, 0.5), (1.2, 1.99, 0.9), (1.0, 1.92, 0.7))], [wall])
    edges, diag = ContactsSurfaceExtractor().extract(arts, ContactsSurfaceConfig())
    if edges:
        raise AssertionError("synth_bbox_fallback wall must be skipped by default")
    if diag.rejection_samples[0].rejected_reason != "surface_source_excluded":
        raise AssertionError("expected surface_source_excluded")


def test_skips_polygon_none_wall() -> None:
    wall = _surface("wall_mesh", "wall", Plane(0.0, -1.0, 0.0, 2.0), None, source="mesh_ransac")
    arts = _artifacts([_entity("e1", (0.8, 1.85, 0.5), (1.2, 1.99, 0.9), (1.0, 1.92, 0.7))], [wall])
    edges, diag = ContactsSurfaceExtractor().extract(arts, ContactsSurfaceConfig())
    if edges:
        raise AssertionError("polygon=None wall must be skipped (footprint required)")
    if diag.rejection_samples[0].rejected_reason != "polygon_required_for_contacts_surface":
        raise AssertionError("expected polygon_required_for_contacts_surface")


def test_threshold_ordering_guard_raises() -> None:
    _assert_value_error(ContactsSurfaceConfig, contact_threshold_m=0.5)
    _assert_value_error(ContactsSurfaceConfig, contact_threshold_m=0.25, footprint_tolerance_m=0.25)
    ContactsSurfaceConfig()  # default valid


def test_rejects_wrong_config_type() -> None:
    arts = _artifacts([], [])
    try:
        ContactsSurfaceExtractor().extract(arts, SurfaceProximityConfig())
    except TypeError:
        return
    raise AssertionError("expected TypeError for wrong config type")


# --- evidence up survives dump/load (the P4.06 serde lesson) -------------


def test_contacts_surface_edge_survives_roundtrip() -> None:
    wall = _surface("synth_wall_north", "wall", Plane(0.0, -1.0, 0.0, 2.0),
                    [(0, 2, 0), (2, 2, 0), (2, 2, 2), (0, 2, 2)])
    arts = _artifacts([_entity("e1", (0.8, 1.85, 0.5), (1.2, 1.99, 0.9), (1.0, 1.92, 0.7))], [wall])
    bundle, _ = build_graph(
        arts, [ExtractorRun(ContactsSurfaceExtractor(), ContactsSurfaceConfig())],
        density_policy="phase2_telemetry_only",
    )
    if not any(e.type == "CONTACTS_SURFACE" for e in bundle.edges):
        raise AssertionError("expected a CONTACTS_SURFACE edge to round-trip")
    from common.equality import array_aware_equal
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "b"
        dump_scene_graph_bundle(bundle, out)
        loaded = load_scene_graph_bundle(out)
    if not array_aware_equal(bundle, loaded):
        raise AssertionError(
            "CONTACTS_SURFACE bundle did not round-trip (likely a tuple in evidence)"
        )


TESTS = [
    test_config_defaults_match_fixture,
    test_synthetic_cases_translate,
    test_real_w1_positive_and_wn_negatives,
    test_only_lamp_is_a_wall_contact_on_replica,
    test_subset_of_polygon_near_surface,
    test_skips_non_wall_surface,
    test_skips_synth_bbox_fallback,
    test_skips_polygon_none_wall,
    test_threshold_ordering_guard_raises,
    test_rejects_wrong_config_type,
    test_contacts_surface_edge_survives_roundtrip,
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

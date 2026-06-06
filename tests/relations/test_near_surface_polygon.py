"""P3.03 tests: polygon-clipped NEAR_SURFACE wiring + A4 byte-equality.

Three contracts:

  1. A4 byte-equality at the config_hash_payload level:
       _config_hash_payload(SurfaceProximityConfig()) ==
       _config_hash_payload(SurfaceProximityConfig(use_polygon_clip=False))
     and `use_polygon_clip` is absent from both payloads. A non-default
     value (use_polygon_clip=True) must appear in the payload.

  2. A4 byte-equality at the GraphBuilder bundle_hash level:
       build_graph(...) with default config and explicit
       SurfaceProximityConfig(use_polygon_clip=False) produce the same
       SceneGraphBundle.bundle_hash on identical inputs. Direct extractor
       output (edges + diagnostics) also matches across both configs.

  3. Opt-in path (use_polygon_clip=True):
       - emits extractor_version="0.2-near_surface_polygon_clipped" on
         ALL opt-in edges including polygon=None fallbacks (mode is what
         changed, not per-surface polygon availability);
       - polygon present: evidence carries distance_metric="polygon_clipped",
         polygon_clipping_applied=True, normal_gap_m, in_plane_gap_m;
       - polygon None : evidence carries distance_metric="bbox_to_plane",
         polygon_clipping_applied=False, fallback_reason="polygon_none";
       - flips fixture S2/S3/S4 (Phase 2 false positives) from near to
         not_near; preserves S1/S5/S7 as near and S8 as near via fallback.

Run: python tests/relations/test_near_surface_polygon.py
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
from extractors.base import (
    EntityArtifact, EntityArtifacts, EntityIdentity, ExtractionDiagnostics,
    InstanceExtractorConfig, SemanticHypothesis, StructuralSurface,
)
from extractors.oracle_replica import OracleReplicaExtractor
from extractors.serde import CURRENT_SCHEMA_VERSION as ENT_SCHEMA_VERSION
from graph.builder import ExtractorRun, _config_hash_payload, build_graph
from graph.relations.surface import (
    PLANE_MODE_VERSION,
    POLYGON_CLIPPED_VERSION,
    SurfaceProximityConfig,
    SurfaceProximityExtractor,
)
from representations.mesh import MeshRepresentation


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
P3_FIXTURE_PATH = (
    REPO_ROOT / "eval" / "questions" / "phase3_near_surface_polygon_smoke.json"
)


def _frame() -> SceneFrame:
    return SceneFrame(
        gravity=(0.0, 0.0, -1.0),
        canonical_forward=None, canonical_right=None,
        units="meters", notes="",
    )


def _entity(uid: str, lo: tuple, hi: tuple) -> EntityArtifact:
    cx = (lo[0] + hi[0]) / 2
    cy = (lo[1] + hi[1]) / 2
    cz = (lo[2] + hi[2]) / 2
    return EntityArtifact(
        identity=EntityIdentity(
            object_uid=uid, display_label=uid, aliases=[], source_instance_ref=uid,
        ),
        bbox_aabb=(lo, hi), bbox_obb=None, centroid=(cx, cy, cz),
        geometry_handle=None,
        semantic_hypotheses=[SemanticHypothesis(label=uid, confidence=1.0, source="t")],
        embedding=None, extraction_diagnostics={},
    )


def _surface(
    uid: str,
    stype: str,
    plane: Plane,
    polygon: list[tuple[float, float, float]] | None = None,
    source: str = "habitat_label",
) -> StructuralSurface:
    return StructuralSurface(
        surface_uid=uid, surface_type=stype, plane=plane,
        polygon=polygon, confidence=1.0, source=source,
    )


def _artifacts(
    entities: list[EntityArtifact],
    surfaces: list[StructuralSurface],
    *, scene_id: str = "t",
    bundle_hash: str = "ent_t",
) -> EntityArtifacts:
    return EntityArtifacts(
        schema_version=ENT_SCHEMA_VERSION, bundle_hash=bundle_hash,
        scene_id=scene_id, frame=_frame(), representation_hash="rep_t",
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
    with P3_FIXTURE_PATH.open() as fh:
        return json.load(fh)


def _fixture_case(fid: str) -> dict:
    fixture = _load_fixture()
    return next(c for c in fixture["cases"] if c["id"] == fid)


def _entity_from_fixture(case: dict) -> EntityArtifact:
    mn = case["entity_aabb"]["min"]
    mx = case["entity_aabb"]["max"]
    return _entity(
        case["id"],
        (mn[0], mn[1], mn[2]),
        (mx[0], mx[1], mx[2]),
    )


def _surface_from_fixture(case: dict) -> StructuralSurface:
    sr = case["surface_record"]
    plane = Plane(a=sr["plane"]["a"], b=sr["plane"]["b"],
                  c=sr["plane"]["c"], d=sr["plane"]["d"])
    if sr["polygon"] is None:
        polygon = None
    else:
        polygon = [(v[0], v[1], v[2]) for v in sr["polygon"]]
    return _surface(
        sr["uid"], sr["surface_type"], plane,
        polygon=polygon, source=sr["source"],
    )


REPLICA_V2_DIR = REPLICA_SCENE_DIR / "enriched" / "v2"


def _build_replica_artifacts() -> EntityArtifacts:
    """Mirror tests/relations/test_near_surface.py:_real_replica_artifacts —
    enriched-v2 path is required so the bundle carries structural surfaces;
    without it the extractor receives an empty surface list."""
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


def _replica_v2_present() -> bool:
    return (REPLICA_V2_DIR / "scene_graph.json").exists()


# --- A4: _config_hash_payload byte-equality ------------------------------


def test_default_config_payload_omits_use_polygon_clip() -> None:
    payload = _config_hash_payload(SurfaceProximityConfig())
    if "use_polygon_clip" in payload:
        raise AssertionError(
            "default SurfaceProximityConfig() payload must omit "
            "use_polygon_clip (hash_omit_if_default mechanism); "
            f"got payload keys {sorted(payload.keys())}"
        )


def test_default_and_explicit_false_payloads_are_identical() -> None:
    """The A4 invariant: SurfaceProximityConfig() and
    SurfaceProximityConfig(use_polygon_clip=False) must serialize to the
    SAME bytes through _config_hash_payload, so bundle hashes stay
    byte-equal across Phase 2 -> Phase 3 transition for unchanged callers."""
    default_payload = _config_hash_payload(SurfaceProximityConfig())
    explicit_payload = _config_hash_payload(
        SurfaceProximityConfig(use_polygon_clip=False)
    )
    if default_payload != explicit_payload:
        raise AssertionError(
            "default and explicit-False payloads differ:\n"
            f"  default:  {default_payload!r}\n"
            f"  explicit: {explicit_payload!r}"
        )
    if json.dumps(default_payload, sort_keys=True) != json.dumps(
        explicit_payload, sort_keys=True
    ):
        raise AssertionError("JSON-serialized payloads differ")


def test_polygon_clip_true_payload_includes_use_polygon_clip() -> None:
    """Non-default value must appear in the payload; otherwise the
    GraphBuilder bundle_hash would not distinguish polygon-mode from
    plane-mode runs, which would be a worse bug than the byte-equality
    risk this metadata is guarding against."""
    payload = _config_hash_payload(
        SurfaceProximityConfig(use_polygon_clip=True)
    )
    if payload.get("use_polygon_clip") is not True:
        raise AssertionError(
            "use_polygon_clip=True must appear in the hash payload; "
            f"got {payload!r}"
        )


# --- A4: direct extractor byte-equality on Replica room_0 ----------------


def test_direct_extractor_byte_equal_on_replica_room_0() -> None:
    """SurfaceProximityConfig() and SurfaceProximityConfig(use_polygon_clip=False)
    must produce identical edges + diagnostics on the real Replica bundle."""
    if not _replica_v2_present():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    artifacts = _build_replica_artifacts()
    edges_default, diag_default = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(),
    )
    edges_explicit, diag_explicit = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(use_polygon_clip=False),
    )
    if len(edges_default) != len(edges_explicit):
        raise AssertionError(
            f"edge counts differ: default={len(edges_default)} "
            f"explicit={len(edges_explicit)}"
        )
    for ed, ex in zip(edges_default, edges_explicit):
        if ed.edge_id != ex.edge_id:
            raise AssertionError(f"edge_id drift: {ed.edge_id} vs {ex.edge_id}")
        if ed.extractor_version != ex.extractor_version:
            raise AssertionError(
                f"version drift: {ed.extractor_version} vs {ex.extractor_version}"
            )
        if ed.evidence != ex.evidence:
            raise AssertionError(
                f"evidence drift on {ed.edge_id}:\n  {ed.evidence!r}\n  {ex.evidence!r}"
            )
    if diag_default.version != diag_explicit.version:
        raise AssertionError("diagnostics.version differs across default configs")
    if diag_default.version != PLANE_MODE_VERSION:
        raise AssertionError(
            f"default diagnostics.version must be {PLANE_MODE_VERSION!r}, "
            f"got {diag_default.version!r}"
        )
    if diag_default.physical_edges_per_type != diag_explicit.physical_edges_per_type:
        raise AssertionError("physical_edges_per_type drifted")
    if diag_default.rejections_per_type != diag_explicit.rejections_per_type:
        raise AssertionError("rejections_per_type drifted")


def test_direct_extractor_default_evidence_has_no_phase3_keys() -> None:
    """Phase 2 evidence schema must remain unchanged on the default path —
    no normal_gap_m, no polygon_clipping_applied, no in_plane_gap_m, no
    fallback_reason leakage. Even on real Replica edges."""
    if not _replica_v2_present():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    artifacts = _build_replica_artifacts()
    edges, _ = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(),
    )
    if not edges:
        raise AssertionError("expected NEAR_SURFACE edges on Replica room_0")
    forbidden_keys = {
        "normal_gap_m",
        "in_plane_gap_m",
        "polygon_clipping_applied",
        "fallback_reason",
    }
    expected_keys = {
        "distance_m", "distance_metric", "threshold_m", "surface_type", "source",
    }
    for e in edges:
        leaked = forbidden_keys & set(e.evidence.keys())
        if leaked:
            raise AssertionError(
                f"Phase 3 evidence key(s) leaked into default-path edge "
                f"{e.edge_id}: {leaked!r}"
            )
        missing = expected_keys - set(e.evidence.keys())
        if missing:
            raise AssertionError(
                f"default-path edge {e.edge_id} missing Phase 2 evidence "
                f"key(s): {missing!r}"
            )


# --- A4: GraphBuilder bundle_hash equality -------------------------------


def _build_minimal_graph(config: SurfaceProximityConfig):
    """Build a SceneGraphBundle using NEAR_SURFACE only. Sparse mode,
    phase2_telemetry_only policy (density check is informational here —
    we are testing hash equality, not density)."""
    # Tiny synthetic scene: 2 entities + 1 floor surface (no polygon).
    # Both modes evaluate it identically when polygon is None on the floor,
    # except the version on the opt-in run, which is the whole point.
    artifacts = _artifacts(
        [
            _entity("e1", (-0.1, -0.1, 0.01), (0.1, 0.1, 0.2)),
            _entity("e2", (1.0, 1.0, 0.50), (1.1, 1.1, 0.7)),
        ],
        [_surface("floor_1", "floor", Plane(a=0.0, b=0.0, c=1.0, d=0.0))],
    )
    runs = [ExtractorRun(SurfaceProximityExtractor(), config)]
    bundle, _diag = build_graph(
        artifacts, runs, density_policy="phase2_telemetry_only",
    )
    return bundle


def test_builder_bundle_hash_equal_default_vs_explicit_false() -> None:
    """The A4 invariant at the builder level: the SceneGraphBundle.bundle_hash
    must be the same when the run config is SurfaceProximityConfig() vs
    SurfaceProximityConfig(use_polygon_clip=False). This is the test that
    actually catches a missing or wrong hash_omit_if_default metadata —
    a direct extractor comparison would pass even if the metadata were
    missing, because the extractor output is identical; the builder hash
    is where the metadata is consulted."""
    bundle_default = _build_minimal_graph(SurfaceProximityConfig())
    bundle_explicit = _build_minimal_graph(
        SurfaceProximityConfig(use_polygon_clip=False)
    )
    if bundle_default.bundle_hash != bundle_explicit.bundle_hash:
        raise AssertionError(
            "GraphBuilder bundle_hash drifted between default and "
            f"explicit-False configs: {bundle_default.bundle_hash} vs "
            f"{bundle_explicit.bundle_hash}"
        )


def test_builder_bundle_hash_differs_for_polygon_mode() -> None:
    """Opt-in polygon mode is a genuinely different extractor mode and
    MUST produce a different bundle_hash. If this test passes spuriously
    (same hash), it means use_polygon_clip is not being hashed — which
    would defeat the whole point of giving the opt-in path a distinct
    extractor version."""
    bundle_default = _build_minimal_graph(SurfaceProximityConfig())
    bundle_polygon = _build_minimal_graph(
        SurfaceProximityConfig(use_polygon_clip=True)
    )
    if bundle_default.bundle_hash == bundle_polygon.bundle_hash:
        raise AssertionError(
            "polygon-mode bundle_hash must differ from default; "
            "use_polygon_clip=True is not reaching the hash payload"
        )


# --- Opt-in: version + evidence schema -----------------------------------


def test_opt_in_emits_polygon_clipped_version_on_polygon_present_edge() -> None:
    """S1 (entity inside wall polygon footprint): both modes emit NEAR,
    but opt-in mode tags the edge with extractor_version=
    "0.2-near_surface_polygon_clipped" and the polygon-clipped evidence
    keys."""
    case = _fixture_case("S1")
    artifacts = _artifacts(
        [_entity_from_fixture(case)], [_surface_from_fixture(case)],
    )
    edges, diag = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(use_polygon_clip=True),
    )
    if len(edges) != 1:
        raise AssertionError(f"S1 opt-in: expected 1 edge, got {len(edges)}")
    e = edges[0]
    if e.extractor_version != POLYGON_CLIPPED_VERSION:
        raise AssertionError(
            f"S1 opt-in version wrong: {e.extractor_version!r}"
        )
    if e.evidence.get("distance_metric") != "polygon_clipped":
        raise AssertionError(
            f"S1 distance_metric: {e.evidence.get('distance_metric')!r}"
        )
    if e.evidence.get("polygon_clipping_applied") is not True:
        raise AssertionError("S1 polygon_clipping_applied must be True")
    if "normal_gap_m" not in e.evidence or "in_plane_gap_m" not in e.evidence:
        raise AssertionError(
            f"S1 evidence missing normal_gap_m/in_plane_gap_m: {e.evidence!r}"
        )
    if "fallback_reason" in e.evidence:
        raise AssertionError("S1 must not carry fallback_reason")
    # Phase 2 keys still layered on
    for k in ("threshold_m", "surface_type", "source"):
        if k not in e.evidence:
            raise AssertionError(f"S1 missing Phase 2 evidence key {k!r}")
    if diag.version != POLYGON_CLIPPED_VERSION:
        raise AssertionError(
            f"diagnostics.version wrong on opt-in: {diag.version!r}"
        )


def test_opt_in_uses_polygon_clipped_version_for_fallback_edges_too() -> None:
    """A5/sign-off requirement: opt-in MODE is what changed, not per-surface
    polygon availability. Even when the dispatcher falls back to plane
    distance (polygon=None), the edge carries extractor_version=
    "0.2-near_surface_polygon_clipped"."""
    case = _fixture_case("S8")
    artifacts = _artifacts(
        [_entity_from_fixture(case)], [_surface_from_fixture(case)],
    )
    edges, _ = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(use_polygon_clip=True),
    )
    if len(edges) != 1:
        raise AssertionError(
            f"S8 opt-in fallback: expected 1 edge, got {len(edges)}"
        )
    e = edges[0]
    if e.extractor_version != POLYGON_CLIPPED_VERSION:
        raise AssertionError(
            f"S8 fallback version must still be {POLYGON_CLIPPED_VERSION!r}, "
            f"got {e.extractor_version!r}"
        )
    if e.evidence.get("distance_metric") != "bbox_to_plane":
        raise AssertionError(
            f"S8 fallback distance_metric must be 'bbox_to_plane', "
            f"got {e.evidence.get('distance_metric')!r}"
        )
    if e.evidence.get("polygon_clipping_applied") is not False:
        raise AssertionError(
            "S8 fallback polygon_clipping_applied must be False"
        )
    if e.evidence.get("fallback_reason") != "polygon_none":
        raise AssertionError(
            f"S8 fallback_reason wrong: {e.evidence.get('fallback_reason')!r}"
        )
    if "in_plane_gap_m" in e.evidence:
        raise AssertionError(
            "S8 fallback evidence must NOT carry in_plane_gap_m"
        )


def test_default_and_opt_in_edge_ids_differ_for_same_entity_surface() -> None:
    """The edge_id mixes (extractor, version, source, type, target).
    Because the version differs between default and opt-in, the same
    (entity, surface) pair must produce DIFFERENT edge_ids across modes —
    so a downstream merge can never silently treat them as the same edge."""
    case = _fixture_case("S1")
    artifacts = _artifacts(
        [_entity_from_fixture(case)], [_surface_from_fixture(case)],
    )
    default_edges, _ = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(),
    )
    opt_in_edges, _ = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(use_polygon_clip=True),
    )
    if len(default_edges) != 1 or len(opt_in_edges) != 1:
        raise AssertionError(
            f"expected 1 edge per mode on S1; got {len(default_edges)} "
            f"/ {len(opt_in_edges)}"
        )
    if default_edges[0].edge_id == opt_in_edges[0].edge_id:
        raise AssertionError(
            "default and opt-in edges share an edge_id — version must "
            "be embedded in edge_id; check make_edge_id"
        )


# --- Opt-in: fixture S2/S3/S4 plane-vs-polygon flips ---------------------


def _emits_near(case: dict, *, use_polygon_clip: bool) -> tuple[bool, dict]:
    artifacts = _artifacts(
        [_entity_from_fixture(case)], [_surface_from_fixture(case)],
    )
    edges, _ = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(use_polygon_clip=use_polygon_clip),
    )
    return (len(edges) == 1), (edges[0].evidence if edges else {})


def test_opt_in_flips_s2_s3_s4_from_near_to_not_near() -> None:
    """The plane-only false positives this phase exists to fix. Under the
    default Phase 2 config they emit a NEAR edge (plane distance ≤ threshold);
    under polygon-clip they are correctly rejected because in-plane gap pushes
    the dispatcher distance past the threshold."""
    for fid in ("S2", "S3", "S4"):
        case = _fixture_case(fid)
        near_default, _ = _emits_near(case, use_polygon_clip=False)
        near_opt_in, _ = _emits_near(case, use_polygon_clip=True)
        if not near_default:
            raise AssertionError(
                f"{fid}: default (plane mode) should emit NEAR — "
                "fixture documents this as a Phase 2 false positive"
            )
        if near_opt_in:
            raise AssertionError(
                f"{fid}: opt-in (polygon mode) must reject — "
                "this is the case the phase exists to fix"
            )


def test_opt_in_preserves_s1_s5_s7_as_near() -> None:
    """True-near cases must continue to emit edges under opt-in mode."""
    for fid in ("S1", "S5", "S7"):
        case = _fixture_case(fid)
        near_opt_in, _ = _emits_near(case, use_polygon_clip=True)
        if not near_opt_in:
            raise AssertionError(
                f"{fid}: opt-in must keep emitting NEAR (true positive)"
            )


def test_opt_in_preserves_s6_as_not_near() -> None:
    """S6: far-from-everything sanity case. Both modes reject."""
    case = _fixture_case("S6")
    near_default, _ = _emits_near(case, use_polygon_clip=False)
    near_opt_in, _ = _emits_near(case, use_polygon_clip=True)
    if near_default or near_opt_in:
        raise AssertionError(
            f"S6: neither mode should emit NEAR; "
            f"default={near_default}, opt_in={near_opt_in}"
        )


def test_opt_in_s7_matches_fixture_distance_evidence() -> None:
    """S7 is the hypot-arithmetic case (normal=0.03, in_plane=0.039 →
    ~0.049204, comfortably within the 0.05 floor threshold per the
    post-freeze amendment). Edge must be emitted and the evidence must
    carry the dispatcher's numbers within fixture tol."""
    case = _fixture_case("S7")
    fixture = _load_fixture()
    tol = fixture["numeric_tolerance_m"]
    near_opt_in, evidence = _emits_near(case, use_polygon_clip=True)
    if not near_opt_in:
        raise AssertionError("S7 must emit NEAR within threshold")
    if abs(evidence["distance_m"] - case["expected_distance_m"]) > tol:
        raise AssertionError(
            f"S7 distance_m: expected {case['expected_distance_m']} "
            f"got {evidence['distance_m']}"
        )
    if abs(evidence["normal_gap_m"] - case["expected_normal_gap_m"]) > tol:
        raise AssertionError(
            f"S7 normal_gap_m: expected {case['expected_normal_gap_m']} "
            f"got {evidence['normal_gap_m']}"
        )
    if abs(evidence["in_plane_gap_m"] - case["expected_in_plane_gap_m"]) > tol:
        raise AssertionError(
            f"S7 in_plane_gap_m: expected {case['expected_in_plane_gap_m']} "
            f"got {evidence['in_plane_gap_m']}"
        )


# --- A3 / A7 policy still holds under opt-in -----------------------------


def test_opt_in_still_skips_synth_bbox_fallback_by_default() -> None:
    """Phase 2 A3 policy: synth_bbox_fallback surfaces are excluded unless
    include_synth_fallback=True. Switching to polygon-clip mode does NOT
    re-enable them. Rejection is recorded as surface_source_excluded."""
    artifacts = _artifacts(
        [_entity("e1", (-0.5, -0.5, -0.02), (0.5, 0.5, 0.5))],
        [_surface(
            "floor_synth", "floor",
            Plane(a=0.0, b=0.0, c=1.0, d=0.0),
            polygon=None,
            source="synth_bbox_fallback",
        )],
    )
    edges, diag = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(use_polygon_clip=True),
    )
    if edges:
        raise AssertionError(
            "opt-in mode must still skip synth_bbox_fallback when "
            f"include_synth_fallback=False; got {len(edges)} edges"
        )
    if diag.rejections_per_type.get("NEAR_SURFACE", 0) != 1:
        raise AssertionError(
            f"expected 1 surface_source_excluded rejection, got "
            f"{diag.rejections_per_type!r}"
        )
    sample = diag.rejection_samples[0]
    if sample.rejected_reason != "surface_source_excluded":
        raise AssertionError(
            f"rejection reason wrong: {sample.rejected_reason!r}"
        )


TESTS = [
    test_default_config_payload_omits_use_polygon_clip,
    test_default_and_explicit_false_payloads_are_identical,
    test_polygon_clip_true_payload_includes_use_polygon_clip,
    test_direct_extractor_byte_equal_on_replica_room_0,
    test_direct_extractor_default_evidence_has_no_phase3_keys,
    test_builder_bundle_hash_equal_default_vs_explicit_false,
    test_builder_bundle_hash_differs_for_polygon_mode,
    test_opt_in_emits_polygon_clipped_version_on_polygon_present_edge,
    test_opt_in_uses_polygon_clipped_version_for_fallback_edges_too,
    test_default_and_opt_in_edge_ids_differ_for_same_entity_surface,
    test_opt_in_flips_s2_s3_s4_from_near_to_not_near,
    test_opt_in_preserves_s1_s5_s7_as_near,
    test_opt_in_preserves_s6_as_not_near,
    test_opt_in_s7_matches_fixture_distance_evidence,
    test_opt_in_still_skips_synth_bbox_fallback_by_default,
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

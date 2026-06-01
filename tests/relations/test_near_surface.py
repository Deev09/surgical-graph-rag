"""P2.09 tests: NEAR_SURFACE extractor.

Three contracts:
  1. Math: bbox_to_plane (P2.07) with ≤ threshold per surface_type.
  2. Policy: canonical extractor skips synth_bbox_fallback surfaces
     unless include_synth_fallback=True. Skips are recorded as
     EdgeRejection with rejected_reason="surface_source_excluded".
  3. Frozen smoke list (eval/questions/phase2_near_surface_smoke.json)
     passes on the real Replica enriched-v2 bundle, including all
     `near` AND all `not_near` cases.

Isolation: this suite calls SurfaceProximityExtractor directly. The
extractor is not yet wired into any default GraphBuilder run; P2.10
makes the version-aware density-cap decision.

Run: python tests/relations/test_near_surface.py
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
from graph.relations.surface import (
    SurfaceProximityConfig,
    SurfaceProximityExtractor,
)
from representations.mesh import MeshRepresentation


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
REPLICA_V2_DIR = REPLICA_SCENE_DIR / "enriched" / "v2"
SMOKE_LIST_PATH = REPO_ROOT / "eval" / "questions" / "phase2_near_surface_smoke.json"


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
    uid: str, stype: str, plane: Plane, source: str = "habitat_label",
) -> StructuralSurface:
    return StructuralSurface(
        surface_uid=uid, surface_type=stype, plane=plane,
        polygon=None, confidence=1.0, source=source,
    )


def _artifacts(
    entities: list[EntityArtifact], surfaces: list[StructuralSurface],
) -> EntityArtifacts:
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


def _floor_plane() -> Plane:
    # Floor plane z = 0, interior-facing normal +z
    return Plane(a=0.0, b=0.0, c=1.0, d=0.0)


def _ceiling_plane(z: float = 3.0) -> Plane:
    # Ceiling plane z = z, interior-facing normal -z (point inside → +)
    return Plane(a=0.0, b=0.0, c=-1.0, d=z)


# --- math: bbox_to_plane + per-surface threshold --------------------------


def test_entity_intersecting_floor_plane_is_near() -> None:
    artifacts = _artifacts(
        [_entity("e1", (-0.5, -0.5, -0.02), (0.5, 0.5, 0.5))],
        [_surface("floor_1", "floor", _floor_plane())],
    )
    edges, _ = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(),
    )
    if len(edges) != 1:
        raise AssertionError(f"expected 1 NEAR_SURFACE edge, got {len(edges)}")
    e = edges[0]
    if e.target.kind != "surface" or e.target.uid != "floor_1":
        raise AssertionError(f"target wrong: {e.target!r}")
    if abs(e.evidence["distance_m"]) > 1e-12:
        raise AssertionError(f"intersecting distance should be 0: {e.evidence!r}")


def test_entity_within_threshold_emits_edge() -> None:
    """Box at z=(0.02, 0.5); floor plane z=0; bbox_to_plane=0.02; threshold 0.05."""
    artifacts = _artifacts(
        [_entity("e1", (-0.5, -0.5, 0.02), (0.5, 0.5, 0.5))],
        [_surface("floor_1", "floor", _floor_plane())],
    )
    edges, _ = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(),
    )
    if len(edges) != 1:
        raise AssertionError(f"expected 1 edge, got {len(edges)}")
    if abs(edges[0].evidence["distance_m"] - 0.02) > 1e-9:
        raise AssertionError(f"distance wrong: {edges[0].evidence!r}")


def test_entity_beyond_threshold_is_rejected() -> None:
    """Box at z=(0.10, 0.5); floor plane z=0; bbox_to_plane=0.10 > 0.05."""
    artifacts = _artifacts(
        [_entity("e1", (-0.5, -0.5, 0.10), (0.5, 0.5, 0.5))],
        [_surface("floor_1", "floor", _floor_plane())],
    )
    edges, diag = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(),
    )
    if len(edges) != 0:
        raise AssertionError(f"expected 0 edges, got {len(edges)}")
    if diag.rejections_per_type.get("NEAR_SURFACE", 0) != 1:
        raise AssertionError(f"rejection count wrong: {diag.rejections_per_type!r}")


def test_threshold_inequality_is_inclusive() -> None:
    """At exactly distance == threshold, the edge IS emitted (≤ semantics)."""
    artifacts = _artifacts(
        [_entity("e1", (-0.5, -0.5, 0.05), (0.5, 0.5, 0.5))],  # surface dist = 0.05
        [_surface("floor_1", "floor", _floor_plane())],
    )
    edges, _ = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(floor_threshold_m=0.05),
    )
    if len(edges) != 1:
        raise AssertionError(f"≤ semantics violated: got {len(edges)} edges")


def test_thresholds_are_per_surface_type() -> None:
    """Same entity, three surfaces; only the floor (tight 0.05) emits."""
    z_ceiling = 3.0
    artifacts = _artifacts(
        [_entity("e1", (-0.5, -0.5, 0.04), (0.5, 0.5, 0.5))],
        [
            _surface("floor_1", "floor", _floor_plane()),    # dist 0.04 <= 0.05 ✓
            _surface("wall_1", "wall",                       # dist 0.5 > 0.30 ✗
                     Plane(a=1.0, b=0.0, c=0.0, d=-1.0)),    # wall x=1
            _surface("ceiling_1", "ceiling",                 # dist 2.5 > 0.10 ✗
                     _ceiling_plane(z_ceiling)),
        ],
    )
    edges, _ = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(),
    )
    targets = {e.target.uid for e in edges}
    if targets != {"floor_1"}:
        raise AssertionError(f"expected only floor_1, got {targets!r}")


def test_target_kind_is_surface() -> None:
    artifacts = _artifacts(
        [_entity("e1", (-0.5, -0.5, -0.02), (0.5, 0.5, 0.5))],
        [_surface("floor_1", "floor", _floor_plane())],
    )
    edges, _ = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(),
    )
    for e in edges:
        if e.target.kind != "surface":
            raise AssertionError(f"target kind wrong: {e.target.kind!r}")
        if e.source.kind != "entity":
            raise AssertionError(f"source kind wrong: {e.source.kind!r}")


def test_edge_provenance_records_distance_metric_and_threshold() -> None:
    artifacts = _artifacts(
        [_entity("e1", (-0.5, -0.5, -0.02), (0.5, 0.5, 0.5))],
        [_surface("floor_1", "floor", _floor_plane())],
    )
    edges, _ = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(),
    )
    ev = edges[0].evidence
    if ev.get("distance_metric") != "bbox_to_plane":
        raise AssertionError(f"distance_metric wrong: {ev!r}")
    if ev.get("threshold_m") != 0.05:
        raise AssertionError(f"threshold_m missing: {ev!r}")
    if ev.get("surface_type") != "floor":
        raise AssertionError(f"surface_type missing: {ev!r}")
    if ev.get("source") != "habitat_label":
        raise AssertionError(f"source missing: {ev!r}")


def test_extractor_version_is_0_1() -> None:
    artifacts = _artifacts(
        [_entity("e1", (-0.5, -0.5, -0.02), (0.5, 0.5, 0.5))],
        [_surface("floor_1", "floor", _floor_plane())],
    )
    edges, diag = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(),
    )
    if diag.version != "0.1":
        raise AssertionError(f"diag version wrong: {diag.version!r}")
    if edges[0].extractor_version != "0.1":
        raise AssertionError(f"edge version wrong: {edges[0].extractor_version!r}")


def test_deterministic() -> None:
    artifacts = _artifacts(
        [
            _entity("e1", (-0.5, -0.5, -0.02), (0.5, 0.5, 0.5)),
            _entity("e2", (1.0, 1.0, 0.04), (1.5, 1.5, 0.5)),
        ],
        [_surface("floor_1", "floor", _floor_plane())],
    )
    a, _ = SurfaceProximityExtractor().extract(artifacts, SurfaceProximityConfig())
    b, _ = SurfaceProximityExtractor().extract(artifacts, SurfaceProximityConfig())
    if [e.edge_id for e in a] != [e.edge_id for e in b]:
        raise AssertionError("non-deterministic edge_ids")


def test_rejects_non_surface_config_type() -> None:
    artifacts = _artifacts([], [])
    try:
        SurfaceProximityExtractor().extract(artifacts, object())  # type: ignore[arg-type]
    except TypeError:
        return
    raise AssertionError("expected TypeError on wrong config")


def test_rejects_non_sparse_mode() -> None:
    artifacts = _artifacts([], [])
    try:
        SurfaceProximityExtractor().extract(
            artifacts,
            SurfaceProximityConfig(mode="compat"),  # type: ignore[arg-type]
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError on unsupported mode")


def test_rejects_negative_threshold() -> None:
    artifacts = _artifacts([], [])
    try:
        SurfaceProximityExtractor().extract(
            artifacts,
            SurfaceProximityConfig(floor_threshold_m=-0.01),
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError on negative threshold")


def test_rejects_non_finite_threshold() -> None:
    artifacts = _artifacts([], [])
    try:
        SurfaceProximityExtractor().extract(
            artifacts,
            SurfaceProximityConfig(wall_threshold_m=math.inf),
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError on non-finite threshold")


# --- policy: synth_bbox_fallback handling ---------------------------------


def test_synth_fallback_surface_skipped_by_default() -> None:
    artifacts = _artifacts(
        [_entity("e1", (-0.5, -0.5, -0.02), (0.5, 0.5, 0.5))],
        [_surface("floor_synth", "floor", _floor_plane(), source="synth_bbox_fallback")],
    )
    edges, diag = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(),
    )
    if len(edges) != 0:
        raise AssertionError(
            f"canonical path must skip synth_bbox_fallback; got {len(edges)} edges"
        )
    if diag.rejections_per_type.get("NEAR_SURFACE", 0) != 1:
        raise AssertionError(
            f"skip should be recorded as 1 rejection; got {diag.rejections_per_type!r}"
        )
    if not diag.rejection_samples:
        raise AssertionError("skip rejection sample not recorded")
    sample = diag.rejection_samples[0]
    if sample.rejected_reason != "surface_source_excluded":
        raise AssertionError(f"reason wrong: {sample.rejected_reason!r}")
    if sample.evidence.get("source") != "synth_bbox_fallback":
        raise AssertionError(f"evidence missing source: {sample.evidence!r}")


def test_synth_fallback_opt_in_emits_edges() -> None:
    artifacts = _artifacts(
        [_entity("e1", (-0.5, -0.5, -0.02), (0.5, 0.5, 0.5))],
        [_surface("floor_synth", "floor", _floor_plane(), source="synth_bbox_fallback")],
    )
    edges, _ = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(include_synth_fallback=True),
    )
    if len(edges) != 1:
        raise AssertionError(
            f"opt-in path should emit; got {len(edges)} edges"
        )
    if edges[0].evidence.get("source") != "synth_bbox_fallback":
        raise AssertionError(
            f"edge evidence must carry source provenance: {edges[0].evidence!r}"
        )


def test_canonical_mix_skips_only_synth_surfaces() -> None:
    artifacts = _artifacts(
        [_entity("e1", (-0.5, -0.5, -0.02), (0.5, 0.5, 0.5))],
        [
            _surface("floor_real", "floor", _floor_plane(), source="habitat_label"),
            _surface("floor_synth", "floor", _floor_plane(), source="synth_bbox_fallback"),
        ],
    )
    edges, _ = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(),
    )
    targets = {e.target.uid for e in edges}
    if targets != {"floor_real"}:
        raise AssertionError(f"expected only floor_real, got {targets!r}")


def test_synth_fallback_rejections_use_real_entity_refs() -> None:
    artifacts = _artifacts(
        [
            _entity("e1", (-0.5, -0.5, -0.02), (0.5, 0.5, 0.5)),
            _entity("e2", (1.0, 1.0, 0.04), (1.5, 1.5, 0.5)),
        ],
        [_surface("floor_synth", "floor", _floor_plane(), source="synth_bbox_fallback")],
    )
    _edges, diag = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(),
    )
    if diag.rejections_per_type.get("NEAR_SURFACE", 0) != 2:
        raise AssertionError(
            f"expected one rejection per entity/surface pair: {diag.rejections_per_type!r}"
        )
    source_uids = {sample.source.uid for sample in diag.rejection_samples}
    if source_uids != {"e1", "e2"}:
        raise AssertionError(f"policy rejections used fake entity refs: {source_uids!r}")


# --- frozen smoke list against the real Replica enriched-v2 bundle --------


def _real_replica_artifacts():
    capture = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    representation = MeshRepresentation(
        bundle=OracleReplicaAdapter().reconstruct(
            capture, ReconstructionConfig(name="oracle_replica", version="0.1", params={}),
        ),
    )
    return OracleReplicaExtractor(enriched_v2_path=REPLICA_V2_DIR).extract(
        representation,
        InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )


def test_real_replica_smoke_list_all_cases_pass() -> None:
    if not (REPLICA_V2_DIR / "scene_graph.json").exists():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    smoke = json.loads(SMOKE_LIST_PATH.read_text(encoding="utf-8"))
    artifacts = _real_replica_artifacts()
    cfg = SurfaceProximityConfig(
        floor_threshold_m=smoke["thresholds_m"]["floor"],
        wall_threshold_m=smoke["thresholds_m"]["wall"],
        ceiling_threshold_m=smoke["thresholds_m"]["ceiling"],
        include_synth_fallback=smoke["policy"]["include_synth_fallback"],
    )
    edges, _ = SurfaceProximityExtractor().extract(artifacts, cfg)
    emitted_pairs = {
        (e.source.uid, e.target.uid) for e in edges
    }

    failures: list[str] = []
    for case in smoke["cases"]:
        pair = (case["entity_uid"], case["surface_uid"])
        present = pair in emitted_pairs
        if case["expectation"] == "near" and not present:
            failures.append(
                f"NEAR expected but missing: {pair} — {case['rationale']}"
            )
        if case["expectation"] == "not_near" and present:
            failures.append(
                f"NOT NEAR expected but emitted: {pair} — {case['rationale']}"
            )
    if failures:
        raise AssertionError(
            f"smoke list failures ({len(failures)}):\n  " + "\n  ".join(failures)
        )


def test_real_replica_smoke_has_at_least_4_near_and_4_not_near() -> None:
    """Lock the shape of the frozen smoke list — A8 minimum: ≥4 each."""
    smoke = json.loads(SMOKE_LIST_PATH.read_text(encoding="utf-8"))
    near = sum(1 for c in smoke["cases"] if c["expectation"] == "near")
    not_near = sum(1 for c in smoke["cases"] if c["expectation"] == "not_near")
    if near < 4 or not_near < 4:
        raise AssertionError(
            f"smoke list violates A8: near={near} not_near={not_near} (need ≥4 each)"
        )


def test_real_replica_smoke_cases_are_valid_and_cover_walls() -> None:
    if not (REPLICA_V2_DIR / "scene_graph.json").exists():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    smoke = json.loads(SMOKE_LIST_PATH.read_text(encoding="utf-8"))
    artifacts = _real_replica_artifacts()
    entity_uids = {entity.identity.object_uid for entity in artifacts.entities}
    surface_uids = {surface.surface_uid for surface in artifacts.structural_surfaces}
    surface_type_by_uid = {
        surface.surface_uid: surface.surface_type
        for surface in artifacts.structural_surfaces
    }
    expected_counts = {"near": 0, "not_near": 0}
    wall_expectations: set[str] = set()
    for case in smoke["cases"]:
        expectation = case["expectation"]
        if expectation not in expected_counts:
            raise AssertionError(f"unknown smoke expectation: {expectation!r}")
        if case["entity_uid"] not in entity_uids:
            raise AssertionError(f"unknown smoke entity_uid: {case['entity_uid']!r}")
        if case["surface_uid"] not in surface_uids:
            raise AssertionError(f"unknown smoke surface_uid: {case['surface_uid']!r}")
        expected_counts[expectation] += 1
        if surface_type_by_uid[case["surface_uid"]] == "wall":
            wall_expectations.add(expectation)
    if smoke["counts"] != expected_counts:
        raise AssertionError(
            f"smoke counts drifted: declared={smoke['counts']!r} actual={expected_counts!r}"
        )
    if wall_expectations != {"near", "not_near"}:
        raise AssertionError(
            f"smoke fixture must cover wall positives and negatives: {wall_expectations!r}"
        )


def test_real_replica_extractor_runs_against_full_bundle() -> None:
    if not (REPLICA_V2_DIR / "scene_graph.json").exists():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    artifacts = _real_replica_artifacts()
    edges, diag = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(),
    )
    if diag.physical_edges_total != len(edges):
        raise AssertionError("diagnostic count mismatch")
    if diag.physical_edges_total <= 0:
        raise AssertionError(
            "expected nontrivial NEAR_SURFACE on real Replica"
        )
    for e in edges:
        if e.target.kind != "surface":
            raise AssertionError(f"non-surface target on Replica: {e.target!r}")
        if e.evidence.get("distance_metric") != "bbox_to_plane":
            raise AssertionError("Replica edge missing distance_metric")


TESTS = [
    test_entity_intersecting_floor_plane_is_near,
    test_entity_within_threshold_emits_edge,
    test_entity_beyond_threshold_is_rejected,
    test_threshold_inequality_is_inclusive,
    test_thresholds_are_per_surface_type,
    test_target_kind_is_surface,
    test_edge_provenance_records_distance_metric_and_threshold,
    test_extractor_version_is_0_1,
    test_deterministic,
    test_rejects_non_surface_config_type,
    test_rejects_non_sparse_mode,
    test_rejects_negative_threshold,
    test_rejects_non_finite_threshold,
    test_synth_fallback_surface_skipped_by_default,
    test_synth_fallback_opt_in_emits_edges,
    test_canonical_mix_skips_only_synth_surfaces,
    test_synth_fallback_rejections_use_real_entity_refs,
    test_real_replica_smoke_list_all_cases_pass,
    test_real_replica_smoke_has_at_least_4_near_and_4_not_near,
    test_real_replica_smoke_cases_are_valid_and_cover_walls,
    test_real_replica_extractor_runs_against_full_bundle,
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

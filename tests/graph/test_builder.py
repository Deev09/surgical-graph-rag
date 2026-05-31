"""P1.07 tests: GraphBuilder orchestration.

Run: python tests/graph/test_builder.py

Required by the batch instructions:
  - Compat exact reproduction through the builder (5,414 edges, no
    duplicate edge_id or edge-key collisions when directional + proximity
    run together).
  - Sparse determinism (same inputs → same bundle_hash and edges).
  - Density enforcement (sparse mode raises when logical/entity > 14).
  - Empty entity bundle handled without error.
  - Duplicate rejection (both edge_id collisions and edge-key collisions).
  - Mixed-mode rejection (compat + sparse refused in one run).
  - Extractor-order determinism (same order → same hash and edges).
  - Single-mode enforcement, single-extractor-name enforcement.
  - Per-family and overall physical / logical edge totals reported.
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.base import ReconstructionConfig
from adapters.oracle_replica import (
    OracleReplicaAdapter, build_replica_capture_bundle,
)
from common.equality import array_aware_equal
from common.types import SceneFrame
from extractors.base import (
    EntityArtifact, EntityArtifacts, EntityIdentity, ExtractionDiagnostics,
    InstanceExtractorConfig,
)
from extractors.oracle_replica import OracleReplicaExtractor
from graph.builder import (
    ExtractorRun, GraphBuildError, SPARSE_DENSITY_LIMIT, build_graph,
)
from graph.relations.base import (
    RelationExtractorConfig, RelationExtractorDiagnostics, edge_key,
    make_edge_id, make_entity_ref,
)
from graph.relations.directional import (
    DirectionalConfig, DirectionalExtractor,
)
from graph.relations.proximity import ProximityConfig, ProximityExtractor
from graph.schema import Edge, EdgeType
from graph.serde import dump_scene_graph_bundle, load_scene_graph_bundle
from representations.mesh import MeshRepresentation


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
LEGACY_REPLICA_RELATIONS = REPO_ROOT / "scenes" / "replica_room_0" / "computed_relations" / "scene_graph.json"


def _build_oracle_artifacts() -> EntityArtifacts:
    capture = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    repr_bundle = OracleReplicaAdapter().reconstruct(
        capture, ReconstructionConfig(name="oracle_replica", version="0.1", params={}),
    )
    representation = MeshRepresentation(bundle=repr_bundle)
    return OracleReplicaExtractor().extract(
        representation,
        InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )


def _make_synthetic_entities(centroids: list[tuple[str, tuple[float, float, float]]]) -> EntityArtifacts:
    frame = SceneFrame(
        gravity=(0.0, 0.0, -1.0), canonical_forward=None,
        canonical_right=None, units="meters", notes="synthetic",
    )
    entities = [
        EntityArtifact(
            identity=EntityIdentity(
                object_uid=uid, display_label=uid, aliases=[],
                source_instance_ref=uid,
            ),
            bbox_aabb=((c[0]-0.1, c[1]-0.1, c[2]-0.1), (c[0]+0.1, c[1]+0.1, c[2]+0.1)),
            bbox_obb=None,
            centroid=c,
            geometry_handle=None,
            semantic_hypotheses=[],
            embedding=None,
            extraction_diagnostics={},
        )
        for uid, c in centroids
    ]
    return EntityArtifacts(
        schema_version=1,
        bundle_hash="ent_synth",
        scene_id="synth",
        frame=frame,
        representation_hash="repr_synth",
        extractor_name="synth",
        extractor_version="0.1",
        entities=entities,
        structural_surfaces=[],
        geometry_store_path=None,
        diagnostics=ExtractionDiagnostics(
            n_entities=len(entities), n_structural_surfaces=0,
            runtime_seconds=0.0, coverage_score=None, notes="",
        ),
        notes={},
    )


# ---------- stub extractor for duplicate / density scenarios ----------

@dataclass(frozen=True)
class StubConfig:
    """Minimal RelationExtractorConfig for stub-extractor tests."""
    mode: str = "sparse"


class StubExtractor:
    """Returns a fixed list of edges regardless of input. Useful for
    forcing duplicate / density / mode scenarios in tests without
    depending on real geometry."""
    def __init__(
        self, *, name: str, version: str, edges: list[Edge],
        edge_types: frozenset[EdgeType],
    ):
        self.name = name
        self.version = version
        self.edge_types = edge_types
        self._edges = list(edges)

    def extract(
        self, entities: EntityArtifacts, config: RelationExtractorConfig,
    ) -> tuple[list[Edge], RelationExtractorDiagnostics]:
        counts: dict[EdgeType, int] = {}
        for e in self._edges:
            counts[e.type] = counts.get(e.type, 0) + 1
        diag = RelationExtractorDiagnostics(
            extractor=self.name, version=self.version, mode=str(config.mode),
            physical_edges_per_type=counts,
            physical_edges_total=len(self._edges),
            logical_edges_total=len(self._edges),
            rejections_per_type={}, rejection_samples=[], runtime_ms=0,
        )
        return list(self._edges), diag


def _stub_edge(extractor: str, version: str, src_uid: str, type_: EdgeType, tgt_uid: str) -> Edge:
    src = make_entity_ref(src_uid)
    tgt = make_entity_ref(tgt_uid)
    return Edge(
        edge_id=make_edge_id(extractor, version, src, type_, tgt),
        source=src, type=type_, target=tgt,
        frame="world", weight=1.0, confidence=1.0,
        extractor=extractor, extractor_version=version, evidence={},
    )


# ---------- compat exact reproduction through the builder ----------

def test_compat_full_reproduction_through_builder() -> None:
    artifacts = _build_oracle_artifacts()
    runs = [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="compat")),
        ExtractorRun(ProximityExtractor(), ProximityConfig(mode="compat")),
    ]
    bundle, diag = build_graph(artifacts, runs)
    produced_keys = {edge_key(e) for e in bundle.edges}
    legacy = json.loads(LEGACY_REPLICA_RELATIONS.read_text())["relations"]
    legacy_keys = {("entity", r["source"], r["type"], "entity", r["target"]) for r in legacy}
    if produced_keys != legacy_keys:
        raise AssertionError(
            f"compat builder output differs from legacy: "
            f"missing={len(legacy_keys - produced_keys)}, "
            f"extra={len(produced_keys - legacy_keys)}"
        )
    if diag.physical_edges_total != 5414:
        raise AssertionError(f"expected 5414 edges, got {diag.physical_edges_total}")
    if diag.mode != "compat":
        raise AssertionError(f"diagnostics mode mismatch: {diag.mode!r}")


# ---------- sparse determinism ----------

def test_sparse_builder_is_deterministic() -> None:
    artifacts = _build_oracle_artifacts()
    runs = lambda: [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse")),
        ExtractorRun(ProximityExtractor(), ProximityConfig(mode="sparse")),
    ]
    b1, _ = build_graph(artifacts, runs())
    b2, _ = build_graph(artifacts, runs())
    if b1.bundle_hash != b2.bundle_hash:
        raise AssertionError(f"bundle_hash drift: {b1.bundle_hash} vs {b2.bundle_hash}")
    if [e.edge_id for e in b1.edges] != [e.edge_id for e in b2.edges]:
        raise AssertionError("edge_id ordering not deterministic")


def test_sparse_bundle_round_trips() -> None:
    artifacts = _build_oracle_artifacts()
    runs = [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse")),
        ExtractorRun(ProximityExtractor(), ProximityConfig(mode="sparse")),
    ]
    bundle, _ = build_graph(artifacts, runs)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "graph"
        dump_scene_graph_bundle(bundle, out)
        loaded = load_scene_graph_bundle(out)
    if not array_aware_equal(bundle, loaded):
        raise AssertionError("sparse SceneGraphBundle round-trip lost data")


# ---------- density enforcement ----------

def test_sparse_density_guardrail_raises_when_exceeded() -> None:
    """Construct a stub extractor that emits more edges than the limit."""
    entities = _make_synthetic_entities([("a", (0.0, 0.0, 0.0)), ("b", (1.0, 0.0, 0.0))])
    # 2 entities * 14 = 28 max. Emit 30 distinct edge keys to exceed.
    edges = [
        _stub_edge("stub", "0.1", f"a", "LEFT_OF", f"x_{i}")
        for i in range(30)
    ]
    stub = StubExtractor(
        name="stub", version="0.1", edges=edges,
        edge_types=frozenset({"LEFT_OF"}),
    )
    runs = [ExtractorRun(stub, StubConfig(mode="sparse"))]
    try:
        build_graph(entities, runs)
    except GraphBuildError as e:
        if "sparse density guardrail" not in str(e):
            raise AssertionError(f"unexpected GraphBuildError text: {e}")
        return
    raise AssertionError("expected GraphBuildError for density violation")


def test_sparse_density_guardrail_passes_when_within_limit() -> None:
    entities = _make_synthetic_entities([
        (f"e_{i}", (float(i), 0.0, 0.0)) for i in range(10)
    ])
    edges = [
        _stub_edge("stub", "0.1", f"e_0", "LEFT_OF", f"e_{i}")
        for i in range(1, 6)  # 5 edges, ratio 0.5
    ]
    stub = StubExtractor(
        name="stub", version="0.1", edges=edges,
        edge_types=frozenset({"LEFT_OF"}),
    )
    bundle, diag = build_graph(entities, [ExtractorRun(stub, StubConfig(mode="sparse"))])
    if diag.logical_edges_total > SPARSE_DENSITY_LIMIT * 10:
        raise AssertionError("test fixture incorrectly exceeds density")
    if diag.physical_edges_total != 5:
        raise AssertionError(f"expected 5 physical edges, got {diag.physical_edges_total}")


def test_compat_mode_skips_density_guardrail() -> None:
    """Compat mode can far exceed the 14 ratio (it's the legacy graph,
    which produces ~74 ratio on Replica). Builder must NOT enforce the
    guardrail in compat mode."""
    artifacts = _build_oracle_artifacts()
    runs = [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="compat")),
        ExtractorRun(ProximityExtractor(), ProximityConfig(mode="compat")),
    ]
    bundle, diag = build_graph(artifacts, runs)
    # Sanity: this DOES exceed the sparse limit; builder did not raise.
    ratio = diag.logical_edges_total / len(artifacts.entities)
    if ratio <= SPARSE_DENSITY_LIMIT:
        raise AssertionError(
            f"compat ratio {ratio:.2f} did not exceed sparse limit; test premise broken"
        )


# ---------- empty entity bundle ----------

def test_empty_entity_bundle_produces_empty_graph() -> None:
    entities = _make_synthetic_entities([])
    runs = [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse")),
        ExtractorRun(ProximityExtractor(), ProximityConfig(mode="sparse")),
    ]
    bundle, diag = build_graph(entities, runs)
    if bundle.edges:
        raise AssertionError(f"expected no edges, got {len(bundle.edges)}")
    if bundle.nodes:
        raise AssertionError(f"expected no nodes, got {len(bundle.nodes)}")
    if diag.physical_edges_total != 0 or diag.logical_edges_total != 0:
        raise AssertionError("expected zero totals")
    if diag.mode != "sparse":
        raise AssertionError(f"mode mismatch: {diag.mode}")


# ---------- duplicate rejection ----------

def test_duplicate_edge_id_is_rejected() -> None:
    entities = _make_synthetic_entities([("a", (0.0, 0.0, 0.0)), ("b", (1.0, 0.0, 0.0))])
    e = _stub_edge("stub", "0.1", "a", "LEFT_OF", "b")
    # Two identical edges → same edge_id collision.
    stub = StubExtractor(
        name="stub", version="0.1", edges=[e, e],
        edge_types=frozenset({"LEFT_OF"}),
    )
    try:
        build_graph(entities, [ExtractorRun(stub, StubConfig(mode="sparse"))])
    except GraphBuildError as err:
        if "duplicate edge_id" not in str(err):
            raise AssertionError(f"unexpected error text: {err}")
        return
    raise AssertionError("expected GraphBuildError for duplicate edge_id")


def test_duplicate_edge_key_across_extractors_is_rejected() -> None:
    """Two different extractors emit the same (source, type, target).
    Different edge_ids (different extractor names hash differently) but
    same edge key → reject."""
    entities = _make_synthetic_entities([("a", (0.0, 0.0, 0.0)), ("b", (1.0, 0.0, 0.0))])
    e1 = _stub_edge("stub_one", "0.1", "a", "LEFT_OF", "b")
    e2 = _stub_edge("stub_two", "0.1", "a", "LEFT_OF", "b")
    if e1.edge_id == e2.edge_id:
        raise AssertionError("test premise broken: different extractors should hash differently")
    s1 = StubExtractor(name="stub_one", version="0.1", edges=[e1], edge_types=frozenset({"LEFT_OF"}))
    s2 = StubExtractor(name="stub_two", version="0.1", edges=[e2], edge_types=frozenset({"LEFT_OF"}))
    runs = [
        ExtractorRun(s1, StubConfig(mode="sparse")),
        ExtractorRun(s2, StubConfig(mode="sparse")),
    ]
    try:
        build_graph(entities, runs)
    except GraphBuildError as err:
        if "duplicate edge key" not in str(err):
            raise AssertionError(f"unexpected error text: {err}")
        return
    raise AssertionError("expected GraphBuildError for duplicate edge key across extractors")


def test_duplicate_extractor_name_in_runs_is_rejected() -> None:
    entities = _make_synthetic_entities([("a", (0.0, 0.0, 0.0))])
    s1 = StubExtractor(name="stub", version="0.1", edges=[], edge_types=frozenset({"LEFT_OF"}))
    s2 = StubExtractor(name="stub", version="0.2", edges=[], edge_types=frozenset({"LEFT_OF"}))
    runs = [
        ExtractorRun(s1, StubConfig(mode="sparse")),
        ExtractorRun(s2, StubConfig(mode="sparse")),
    ]
    try:
        build_graph(entities, runs)
    except GraphBuildError as err:
        if "appears more than once" not in str(err):
            raise AssertionError(f"unexpected error text: {err}")
        return
    raise AssertionError("expected GraphBuildError for duplicate extractor name")


# ---------- mixed-mode rejection ----------

def test_mixed_mode_runs_rejected() -> None:
    artifacts = _build_oracle_artifacts()
    runs = [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="compat")),
        ExtractorRun(ProximityExtractor(), ProximityConfig(mode="sparse")),
    ]
    try:
        build_graph(artifacts, runs)
    except GraphBuildError as e:
        if "mixed-mode" not in str(e):
            raise AssertionError(f"unexpected error text: {e}")
        return
    raise AssertionError("expected GraphBuildError for mixed-mode runs")


def test_empty_runs_rejected() -> None:
    artifacts = _build_oracle_artifacts()
    try:
        build_graph(artifacts, [])
    except GraphBuildError as e:
        if "at least one" not in str(e):
            raise AssertionError(f"unexpected error text: {e}")
        return
    raise AssertionError("expected GraphBuildError for empty runs")


# ---------- extractor-order determinism ----------

def test_extractor_order_is_honored_and_deterministic() -> None:
    """Running the same set of (extractor, config) pairs in the same
    order must always produce the same bundle_hash. Different orders
    produce different bundle_hashes (the hash truthfully reflects input
    order)."""
    artifacts = _build_oracle_artifacts()
    order_a = [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse")),
        ExtractorRun(ProximityExtractor(), ProximityConfig(mode="sparse")),
    ]
    order_b = [
        ExtractorRun(ProximityExtractor(), ProximityConfig(mode="sparse")),
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse")),
    ]
    b_a1, _ = build_graph(artifacts, order_a)
    b_a2, _ = build_graph(artifacts, order_a)
    b_b, _ = build_graph(artifacts, order_b)

    if b_a1.bundle_hash != b_a2.bundle_hash:
        raise AssertionError("same order should yield same hash")
    if b_a1.bundle_hash == b_b.bundle_hash:
        raise AssertionError("different order should yield different hash")
    # But the set of edges must be the same.
    keys_a = {edge_key(e) for e in b_a1.edges}
    keys_b = {edge_key(e) for e in b_b.edges}
    if keys_a != keys_b:
        raise AssertionError("edge set should not depend on extractor order")


# ---------- physical / logical reporting ----------

def test_per_family_and_overall_totals_reported() -> None:
    artifacts = _build_oracle_artifacts()
    runs = [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="compat")),
        ExtractorRun(ProximityExtractor(), ProximityConfig(mode="compat")),
    ]
    _, diag = build_graph(artifacts, runs)
    if len(diag.per_extractor) != 2:
        raise AssertionError(f"expected 2 per-extractor diagnostics, got {len(diag.per_extractor)}")
    by_name = {d.extractor: d for d in diag.per_extractor}
    if "directional" not in by_name or "proximity" not in by_name:
        raise AssertionError(f"missing per-family diagnostics: {list(by_name)}")
    if by_name["directional"].physical_edges_total != 5188:
        raise AssertionError(
            f"directional physical: expected 5188, got "
            f"{by_name['directional'].physical_edges_total}"
        )
    if by_name["proximity"].physical_edges_total != 226:
        raise AssertionError(
            f"proximity physical: expected 226, got "
            f"{by_name['proximity'].physical_edges_total}"
        )
    if diag.physical_edges_total != 5414:
        raise AssertionError(
            f"overall physical: expected 5414, got {diag.physical_edges_total}"
        )
    # Overall logical = directional logical (2594) + proximity logical (113) = 2707
    if diag.logical_edges_total != 2707:
        raise AssertionError(
            f"overall logical: expected 2707, got {diag.logical_edges_total}"
        )


def test_extractor_versions_and_runtime_reported() -> None:
    artifacts = _build_oracle_artifacts()
    runs = [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse")),
        ExtractorRun(ProximityExtractor(), ProximityConfig(mode="sparse")),
    ]
    _, diag = build_graph(artifacts, runs)
    if set(diag.extractor_versions.keys()) != {"directional", "proximity"}:
        raise AssertionError(f"unexpected extractor versions keys: {diag.extractor_versions}")
    if set(diag.runtime_ms_per_extractor.keys()) != {"directional", "proximity"}:
        raise AssertionError(
            f"unexpected runtime keys: {diag.runtime_ms_per_extractor}"
        )


# ---------- bundle hash inputs ----------

def test_bundle_hash_changes_with_config() -> None:
    artifacts = _build_oracle_artifacts()
    runs_a = [ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse", sparse_min_delta=0.5))]
    runs_b = [ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse", sparse_min_delta=0.7))]
    b_a, _ = build_graph(artifacts, runs_a)
    b_b, _ = build_graph(artifacts, runs_b)
    if b_a.bundle_hash == b_b.bundle_hash:
        raise AssertionError("bundle_hash should change when extractor config changes")


def test_bundle_hash_changes_with_entity_bundle() -> None:
    e1 = _make_synthetic_entities([("a", (0.0, 0.0, 0.0)), ("b", (1.0, 0.0, 0.0))])
    e2 = _make_synthetic_entities([("a", (0.0, 0.0, 0.0)), ("b", (2.0, 0.0, 0.0))])
    # Synth bundles share the same bundle_hash by default; flip one.
    from dataclasses import replace
    e2 = replace(e2, bundle_hash="ent_synth_v2")
    runs = lambda: [ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse"))]
    b1, _ = build_graph(e1, runs())
    b2, _ = build_graph(e2, runs())
    if b1.bundle_hash == b2.bundle_hash:
        raise AssertionError("bundle_hash should change when entity_bundle_hash changes")


TESTS = [
    test_compat_full_reproduction_through_builder,
    test_sparse_builder_is_deterministic,
    test_sparse_bundle_round_trips,
    test_sparse_density_guardrail_raises_when_exceeded,
    test_sparse_density_guardrail_passes_when_within_limit,
    test_compat_mode_skips_density_guardrail,
    test_empty_entity_bundle_produces_empty_graph,
    test_duplicate_edge_id_is_rejected,
    test_duplicate_edge_key_across_extractors_is_rejected,
    test_duplicate_extractor_name_in_runs_is_rejected,
    test_mixed_mode_runs_rejected,
    test_empty_runs_rejected,
    test_extractor_order_is_honored_and_deterministic,
    test_per_family_and_overall_totals_reported,
    test_extractor_versions_and_runtime_reported,
    test_bundle_hash_changes_with_config,
    test_bundle_hash_changes_with_entity_bundle,
]


def main() -> int:
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
            print()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

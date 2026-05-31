"""P1.06 tests: directional + proximity extractors, compat + sparse modes.

Run: python tests/relations/test_directional_proximity.py

Required by the batch instructions:
  - Deterministic output (compat and sparse, both families)
  - Per-family diagnostics (physical vs logical edge counts)
  - Absence of FAR in any emitted edge list
  - Exact edge-key equality vs the 5,414-edge Replica legacy artifact
    when both families run in compat mode
  - Sparse mode: directional emits only canonical types; proximity
    emits NEAR once per pair (not twice)
  - Config type-mismatch rejections (DirectionalExtractor refuses a
    ProximityConfig and vice versa)
  - Logical-edge counter handles inverse pairs and symmetric edges

Phase 1 replay fixture: scenes/replica_room_0/scene_graph.json plus
capture_meta.json (pre-imported via importers/replica.py). Raw Habitat
import remains the responsibility of importers/replica.py; this slice
does not regenerate from raw inputs.
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
from adapters.oracle_replica import (
    OracleReplicaAdapter, build_replica_capture_bundle,
)
from extractors.base import EntityArtifact, EntityArtifacts, EntityIdentity, ExtractionDiagnostics, InstanceExtractorConfig
from extractors.oracle_replica import OracleReplicaExtractor
from common.types import SceneFrame
from graph.relations.base import count_logical_edges, edge_key
from graph.relations.directional import (
    DIRECTIONAL_TYPES, DirectionalConfig, DirectionalExtractor,
    SPARSE_CANONICAL_TYPES,
)
from graph.relations.proximity import (
    ProximityConfig, ProximityExtractor,
)
from graph.schema import Edge, GraphRef
from representations.mesh import MeshRepresentation


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
LEGACY_REPLICA_RELATIONS = REPO_ROOT / "scenes" / "replica_room_0" / "computed_relations" / "scene_graph.json"


def _build_oracle_artifacts() -> EntityArtifacts:
    """Build the Replica oracle EntityArtifacts via P1.04+P1.05 pipeline."""
    capture = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    adapter = OracleReplicaAdapter()
    repr_bundle = adapter.reconstruct(
        capture, ReconstructionConfig(name="oracle_replica", version="0.1", params={}),
    )
    representation = MeshRepresentation(bundle=repr_bundle)
    extractor = OracleReplicaExtractor()
    return extractor.extract(
        representation,
        InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )


def _load_legacy_edge_keys() -> set[tuple[str, str, str, str, str]]:
    """Load the 5,414-edge legacy artifact as a set of edge keys."""
    sg = json.loads(LEGACY_REPLICA_RELATIONS.read_text())
    return {
        ("entity", r["source"], r["type"], "entity", r["target"])
        for r in sg["relations"]
    }


def _make_synthetic_entities(centroids: list[tuple[str, tuple[float, float, float]]]) -> EntityArtifacts:
    """Construct an EntityArtifacts bundle from (uid, centroid) tuples
    for unit-test scenarios that don't need the full Replica scene."""
    frame = SceneFrame(
        gravity=(0.0, 0.0, -1.0),
        canonical_forward=None,
        canonical_right=None,
        units="meters",
        notes="synthetic test",
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
            n_entities=len(entities),
            n_structural_surfaces=0,
            runtime_seconds=0.0,
            coverage_score=None,
            notes="",
        ),
        notes={},
    )


# ---------- absence of FAR ----------

def test_directional_no_FAR_emitted_compat() -> None:
    artifacts = _build_oracle_artifacts()
    edges, _ = DirectionalExtractor().extract(artifacts, DirectionalConfig(mode="compat"))
    for e in edges:
        if e.type == "FAR":
            raise AssertionError("FAR must never be emitted by any extractor")


def test_directional_no_FAR_emitted_sparse() -> None:
    artifacts = _build_oracle_artifacts()
    edges, _ = DirectionalExtractor().extract(artifacts, DirectionalConfig(mode="sparse"))
    for e in edges:
        if e.type == "FAR":
            raise AssertionError("FAR must never be emitted by any extractor")


def test_proximity_no_FAR_emitted_compat() -> None:
    artifacts = _build_oracle_artifacts()
    edges, _ = ProximityExtractor().extract(artifacts, ProximityConfig(mode="compat"))
    for e in edges:
        if e.type == "FAR":
            raise AssertionError("FAR must never be emitted by any extractor")


def test_proximity_no_FAR_emitted_sparse() -> None:
    artifacts = _build_oracle_artifacts()
    edges, _ = ProximityExtractor().extract(artifacts, ProximityConfig(mode="sparse"))
    for e in edges:
        if e.type == "FAR":
            raise AssertionError("FAR must never be emitted by any extractor")


# ---------- exact legacy reproduction in compat mode ----------

def test_compat_reproduces_legacy_replica_artifact_exactly() -> None:
    artifacts = _build_oracle_artifacts()
    dir_edges, _ = DirectionalExtractor().extract(artifacts, DirectionalConfig(mode="compat"))
    prox_edges, _ = ProximityExtractor().extract(artifacts, ProximityConfig(mode="compat"))
    produced_keys = {edge_key(e) for e in dir_edges + prox_edges}
    legacy_keys = _load_legacy_edge_keys()

    if produced_keys != legacy_keys:
        missing = legacy_keys - produced_keys
        extra = produced_keys - legacy_keys
        raise AssertionError(
            f"compat reproduction diff: missing={len(missing)}, extra={len(extra)}\n"
            f"  sample missing: {sorted(missing)[:5]}\n"
            f"  sample extra:   {sorted(extra)[:5]}"
        )

    if len(dir_edges) + len(prox_edges) != 5414:
        raise AssertionError(
            f"expected 5414 physical edges in compat mode, "
            f"got {len(dir_edges) + len(prox_edges)}"
        )


def test_compat_directional_type_counts_match_legacy() -> None:
    """Per-type breakdown: 1537 LEFT_OF, 1537 RIGHT_OF, 232 ABOVE, 232
    BELOW, 825 BEHIND, 825 IN_FRONT_OF, 226 NEAR."""
    artifacts = _build_oracle_artifacts()
    dir_edges, dir_diag = DirectionalExtractor().extract(artifacts, DirectionalConfig(mode="compat"))
    prox_edges, prox_diag = ProximityExtractor().extract(artifacts, ProximityConfig(mode="compat"))
    expected = {
        "LEFT_OF": 1537, "RIGHT_OF": 1537,
        "ABOVE": 232, "BELOW": 232,
        "BEHIND": 825, "IN_FRONT_OF": 825,
    }
    for t, c in expected.items():
        got = dir_diag.physical_edges_per_type.get(t, 0)
        if got != c:
            raise AssertionError(f"{t}: expected {c}, got {got}")
    if prox_diag.physical_edges_per_type.get("NEAR", 0) != 226:
        raise AssertionError(
            f"NEAR: expected 226, got {prox_diag.physical_edges_per_type.get('NEAR', 0)}"
        )


# ---------- determinism ----------

def _edges_to_keys_sorted(edges: list[Edge]) -> list[tuple]:
    return sorted([edge_key(e) for e in edges])


def test_directional_compat_deterministic() -> None:
    artifacts = _build_oracle_artifacts()
    e1, _ = DirectionalExtractor().extract(artifacts, DirectionalConfig(mode="compat"))
    e2, _ = DirectionalExtractor().extract(artifacts, DirectionalConfig(mode="compat"))
    if _edges_to_keys_sorted(e1) != _edges_to_keys_sorted(e2):
        raise AssertionError("directional compat output not deterministic")
    if [e.edge_id for e in e1] != [e.edge_id for e in e2]:
        raise AssertionError("directional compat edge_ids not deterministic")


def test_directional_sparse_deterministic() -> None:
    artifacts = _build_oracle_artifacts()
    cfg = DirectionalConfig(mode="sparse")
    e1, _ = DirectionalExtractor().extract(artifacts, cfg)
    e2, _ = DirectionalExtractor().extract(artifacts, cfg)
    if _edges_to_keys_sorted(e1) != _edges_to_keys_sorted(e2):
        raise AssertionError("directional sparse output not deterministic")


def test_proximity_compat_deterministic() -> None:
    artifacts = _build_oracle_artifacts()
    e1, _ = ProximityExtractor().extract(artifacts, ProximityConfig(mode="compat"))
    e2, _ = ProximityExtractor().extract(artifacts, ProximityConfig(mode="compat"))
    if _edges_to_keys_sorted(e1) != _edges_to_keys_sorted(e2):
        raise AssertionError("proximity compat output not deterministic")


def test_proximity_sparse_deterministic() -> None:
    artifacts = _build_oracle_artifacts()
    e1, _ = ProximityExtractor().extract(artifacts, ProximityConfig(mode="sparse"))
    e2, _ = ProximityExtractor().extract(artifacts, ProximityConfig(mode="sparse"))
    if _edges_to_keys_sorted(e1) != _edges_to_keys_sorted(e2):
        raise AssertionError("proximity sparse output not deterministic")


# ---------- sparse-mode shape ----------

def test_directional_sparse_emits_only_canonical_types() -> None:
    artifacts = _build_oracle_artifacts()
    edges, diag = DirectionalExtractor().extract(artifacts, DirectionalConfig(mode="sparse"))
    non_canonical = [e for e in edges if e.type not in SPARSE_CANONICAL_TYPES]
    if non_canonical:
        raise AssertionError(
            f"sparse directional emitted non-canonical types: "
            f"{sorted({e.type for e in non_canonical})}"
        )
    # Every emitted type should be in DIRECTIONAL_TYPES (sanity)
    for e in edges:
        if e.type not in DIRECTIONAL_TYPES:
            raise AssertionError(f"unexpected type {e.type!r} in directional output")


def test_proximity_sparse_emits_each_pair_once() -> None:
    artifacts = _build_oracle_artifacts()
    edges, _ = ProximityExtractor().extract(artifacts, ProximityConfig(mode="sparse"))
    seen_unordered: set[frozenset[str]] = set()
    for e in edges:
        pair = frozenset({e.source.uid, e.target.uid})
        if pair in seen_unordered:
            raise AssertionError(
                f"sparse proximity emitted NEAR twice for unordered pair {sorted(pair)}"
            )
        seen_unordered.add(pair)
    # Each edge should carry the symmetric=True evidence flag.
    for e in edges:
        if e.evidence.get("symmetric") is not True:
            raise AssertionError(
                "sparse NEAR edges must carry evidence={'symmetric': True, ...}"
            )


# ---------- physical vs logical edge counts ----------

def test_diagnostics_physical_vs_logical_compat_directional() -> None:
    artifacts = _build_oracle_artifacts()
    _, diag = DirectionalExtractor().extract(artifacts, DirectionalConfig(mode="compat"))
    if diag.physical_edges_total != 5188:
        raise AssertionError(
            f"compat directional physical: expected 5188, got {diag.physical_edges_total}"
        )
    # Logical: each inverse pair counted once.
    if diag.logical_edges_total != 2594:
        raise AssertionError(
            f"compat directional logical: expected 2594, got {diag.logical_edges_total}"
        )
    if diag.logical_edges_total * 2 != diag.physical_edges_total:
        raise AssertionError(
            f"compat directional: physical should be exactly 2x logical; "
            f"got physical={diag.physical_edges_total}, logical={diag.logical_edges_total}"
        )


def test_diagnostics_physical_vs_logical_compat_proximity() -> None:
    artifacts = _build_oracle_artifacts()
    _, diag = ProximityExtractor().extract(artifacts, ProximityConfig(mode="compat"))
    if diag.physical_edges_total != 226:
        raise AssertionError(
            f"compat proximity physical: expected 226, got {diag.physical_edges_total}"
        )
    if diag.logical_edges_total != 113:
        raise AssertionError(
            f"compat proximity logical: expected 113, got {diag.logical_edges_total}"
        )


def test_diagnostics_physical_equals_logical_in_sparse() -> None:
    """Sparse mode never duplicates: physical == logical for both
    extractors."""
    artifacts = _build_oracle_artifacts()
    _, ddiag = DirectionalExtractor().extract(artifacts, DirectionalConfig(mode="sparse"))
    _, pdiag = ProximityExtractor().extract(artifacts, ProximityConfig(mode="sparse"))
    if ddiag.physical_edges_total != ddiag.logical_edges_total:
        raise AssertionError(
            f"sparse directional: physical {ddiag.physical_edges_total} != "
            f"logical {ddiag.logical_edges_total}"
        )
    if pdiag.physical_edges_total != pdiag.logical_edges_total:
        raise AssertionError(
            f"sparse proximity: physical {pdiag.physical_edges_total} != "
            f"logical {pdiag.logical_edges_total}"
        )


# ---------- config type-mismatch rejection ----------

def test_directional_rejects_proximity_config() -> None:
    artifacts = _build_oracle_artifacts()
    try:
        DirectionalExtractor().extract(artifacts, ProximityConfig(mode="compat"))
    except TypeError as e:
        if "DirectionalConfig" not in str(e):
            raise AssertionError(f"unexpected TypeError text: {e}")
        return
    raise AssertionError("expected TypeError for ProximityConfig given to DirectionalExtractor")


def test_proximity_rejects_directional_config() -> None:
    artifacts = _build_oracle_artifacts()
    try:
        ProximityExtractor().extract(artifacts, DirectionalConfig(mode="compat"))
    except TypeError as e:
        if "ProximityConfig" not in str(e):
            raise AssertionError(f"unexpected TypeError text: {e}")
        return
    raise AssertionError("expected TypeError for DirectionalConfig given to ProximityExtractor")


# ---------- logical-edge counter unit tests ----------

def test_count_logical_edges_inverse_pair_counts_once() -> None:
    artifacts = _make_synthetic_entities([("a", (0.0, 0.0, 0.0)), ("b", (2.0, 0.0, 0.0))])
    # Compat mode: emits both LEFT_OF(a,b) and RIGHT_OF(b,a) [legacy logic]
    # The two are the same logical fact.
    edges, _ = DirectionalExtractor().extract(artifacts, DirectionalConfig(mode="compat"))
    physical = len(edges)
    logical = count_logical_edges(edges)
    if physical != 2:
        raise AssertionError(f"expected 2 physical directional edges, got {physical}")
    if logical != 1:
        raise AssertionError(
            f"inverse pair should be 1 logical fact; got physical={physical}, logical={logical}"
        )


def test_count_logical_edges_symmetric_pair_counts_once() -> None:
    artifacts = _make_synthetic_entities([("a", (0.0, 0.0, 0.0)), ("b", (0.5, 0.0, 0.0))])
    edges, _ = ProximityExtractor().extract(artifacts, ProximityConfig(mode="compat"))
    physical = len(edges)
    logical = count_logical_edges(edges)
    if physical != 2:
        raise AssertionError(f"expected 2 physical NEAR edges, got {physical}")
    if logical != 1:
        raise AssertionError(
            f"symmetric pair should be 1 logical fact; got physical={physical}, logical={logical}"
        )


# ---------- compat / sparse logical equivalence on overlapping subset ----------

def test_compat_logical_count_matches_sparse_unordered_count_for_proximity() -> None:
    """A sanity check: NEAR is centroid-based in both modes with the
    same threshold; therefore the set of unordered NEAR pairs should be
    identical between compat (2x physical) and sparse (1x physical)."""
    artifacts = _build_oracle_artifacts()
    compat_edges, compat_diag = ProximityExtractor().extract(artifacts, ProximityConfig(mode="compat"))
    sparse_edges, sparse_diag = ProximityExtractor().extract(artifacts, ProximityConfig(mode="sparse"))
    if compat_diag.logical_edges_total != sparse_diag.logical_edges_total:
        raise AssertionError(
            f"compat logical {compat_diag.logical_edges_total} != "
            f"sparse logical {sparse_diag.logical_edges_total}; "
            "centroid-based NEAR with matching thresholds should agree"
        )


TESTS = [
    test_directional_no_FAR_emitted_compat,
    test_directional_no_FAR_emitted_sparse,
    test_proximity_no_FAR_emitted_compat,
    test_proximity_no_FAR_emitted_sparse,
    test_compat_reproduces_legacy_replica_artifact_exactly,
    test_compat_directional_type_counts_match_legacy,
    test_directional_compat_deterministic,
    test_directional_sparse_deterministic,
    test_proximity_compat_deterministic,
    test_proximity_sparse_deterministic,
    test_directional_sparse_emits_only_canonical_types,
    test_proximity_sparse_emits_each_pair_once,
    test_diagnostics_physical_vs_logical_compat_directional,
    test_diagnostics_physical_vs_logical_compat_proximity,
    test_diagnostics_physical_equals_logical_in_sparse,
    test_directional_rejects_proximity_config,
    test_proximity_rejects_directional_config,
    test_count_logical_edges_inverse_pair_counts_once,
    test_count_logical_edges_symmetric_pair_counts_once,
    test_compat_logical_count_matches_sparse_unordered_count_for_proximity,
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

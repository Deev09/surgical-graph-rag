"""P1.03 tests: BundleCorrespondence builders.

Run: python tests/eval/test_bundle_correspondence.py

Covers:
  - All three methods (shared_source_ref, iou_match, manual).
  - Both entity and surface matching (where the method supports surfaces).
  - IoU + scale-aware centroid-distance fallback per §10 match rule.
  - Manual validation of uids against bundle contents.
  - Bundle-level fields populated correctly (matched + unmatched lists).
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

from common.types import Plane, SceneFrame
from eval.bundle_correspondence import (
    aabb_diag, aabb_iou, centroid_distance,
    correspond_iou_match, correspond_manual, correspond_shared_source_ref,
)
from extractors.base import (
    EntityArtifact, EntityArtifacts, EntityIdentity, ExtractionDiagnostics,
    StructuralSurface,
)


def _frame() -> SceneFrame:
    return SceneFrame(
        gravity=(0.0, 0.0, -1.0), canonical_forward=None,
        canonical_right=None, units="meters", notes="",
    )


def _ent(
    *, uid: str, label: str, source_ref: str,
    centroid: tuple[float, float, float],
    half_extent: float = 0.25,
) -> EntityArtifact:
    h = half_extent
    return EntityArtifact(
        identity=EntityIdentity(
            object_uid=uid, display_label=label, aliases=[],
            source_instance_ref=source_ref,
        ),
        bbox_aabb=(
            (centroid[0]-h, centroid[1]-h, centroid[2]-h),
            (centroid[0]+h, centroid[1]+h, centroid[2]+h),
        ),
        bbox_obb=None, centroid=centroid, geometry_handle=None,
        semantic_hypotheses=[], embedding=None, extraction_diagnostics={},
    )


def _surface(uid: str, kind: str = "floor") -> StructuralSurface:
    return StructuralSurface(
        surface_uid=uid, surface_type=kind,
        plane=Plane(a=0.0, b=0.0, c=1.0, d=0.0), polygon=None, confidence=1.0,
    )


def _bundle(
    *, hash_: str, scene_id: str = "test",
    entities: list[EntityArtifact] | None = None,
    surfaces: list[StructuralSurface] | None = None,
) -> EntityArtifacts:
    ents = entities or []
    surfs = surfaces or []
    return EntityArtifacts(
        schema_version=1, bundle_hash=hash_, scene_id=scene_id, frame=_frame(),
        representation_hash="repr", extractor_name="test", extractor_version="0.1",
        entities=ents, structural_surfaces=surfs, geometry_store_path=None,
        diagnostics=ExtractionDiagnostics(
            n_entities=len(ents), n_structural_surfaces=len(surfs),
            runtime_seconds=0.0, coverage_score=None, notes="",
        ),
        notes={},
    )


# ---------- geometry helpers ----------

def test_aabb_iou_identical_boxes() -> None:
    b = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    if aabb_iou(b, b) != 1.0:
        raise AssertionError(f"identical AABB IoU should be 1.0")


def test_aabb_iou_disjoint() -> None:
    a = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    b = ((10.0, 0.0, 0.0), (11.0, 1.0, 1.0))
    if aabb_iou(a, b) != 0.0:
        raise AssertionError("disjoint AABB IoU should be 0.0")


def test_aabb_iou_half_overlap() -> None:
    a = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    b = ((0.5, 0.0, 0.0), (1.5, 1.0, 1.0))
    # Intersection volume = 0.5, union = 1 + 1 - 0.5 = 1.5
    expected = 0.5 / 1.5
    if abs(aabb_iou(a, b) - expected) > 1e-9:
        raise AssertionError(f"expected ~{expected}, got {aabb_iou(a, b)}")


def test_aabb_diag_unit_cube() -> None:
    b = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    expected = 3.0 ** 0.5
    if abs(aabb_diag(b) - expected) > 1e-9:
        raise AssertionError(f"expected sqrt(3), got {aabb_diag(b)}")


def test_centroid_distance() -> None:
    if abs(centroid_distance((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)) - 5.0) > 1e-9:
        raise AssertionError("expected 5.0 for 3-4-5 triangle")


# ---------- shared_source_ref ----------

def test_shared_source_ref_matches_entities() -> None:
    src = _bundle(hash_="src", entities=[
        _ent(uid="obj_1", label="a", source_ref="42", centroid=(0.0, 0.0, 0.0)),
        _ent(uid="obj_2", label="b", source_ref="43", centroid=(1.0, 0.0, 0.0)),
    ])
    tgt = _bundle(hash_="tgt", entities=[
        _ent(uid="obj_a", label="a", source_ref="42", centroid=(0.0, 0.0, 0.0)),
        _ent(uid="obj_c", label="c", source_ref="99", centroid=(5.0, 5.0, 5.0)),
    ])
    c = correspond_shared_source_ref(src, tgt)
    if c.entity_pairs != [("obj_1", "obj_a")]:
        raise AssertionError(f"expected [(obj_1, obj_a)], got {c.entity_pairs}")
    if c.unmatched_source_entities != ["obj_2"]:
        raise AssertionError(f"unmatched_src: {c.unmatched_source_entities}")
    if c.unmatched_target_entities != ["obj_c"]:
        raise AssertionError(f"unmatched_tgt: {c.unmatched_target_entities}")
    if c.method != "shared_source_ref":
        raise AssertionError(f"method: {c.method}")
    if c.score.get("entity:obj_1->obj_a") != 1.0:
        raise AssertionError(f"score: {c.score}")


def test_shared_source_ref_matches_surfaces() -> None:
    src = _bundle(hash_="src", surfaces=[_surface("floor_0"), _surface("wall_0", "wall")])
    tgt = _bundle(hash_="tgt", surfaces=[_surface("floor_0"), _surface("wall_1", "wall")])
    c = correspond_shared_source_ref(src, tgt)
    if c.surface_pairs != [("floor_0", "floor_0")]:
        raise AssertionError(f"surface_pairs: {c.surface_pairs}")
    if c.unmatched_source_surfaces != ["wall_0"]:
        raise AssertionError(f"unmatched_src_surf: {c.unmatched_source_surfaces}")
    if c.unmatched_target_surfaces != ["wall_1"]:
        raise AssertionError(f"unmatched_tgt_surf: {c.unmatched_target_surfaces}")


def test_shared_source_ref_skips_empty_refs() -> None:
    """Empty source_instance_ref must not match across bundles."""
    src = _bundle(hash_="src", entities=[
        _ent(uid="obj_1", label="x", source_ref="", centroid=(0.0, 0.0, 0.0)),
    ])
    tgt = _bundle(hash_="tgt", entities=[
        _ent(uid="obj_a", label="x", source_ref="", centroid=(0.0, 0.0, 0.0)),
    ])
    c = correspond_shared_source_ref(src, tgt)
    if c.entity_pairs:
        raise AssertionError("empty refs should not match")


def test_shared_source_ref_duplicate_source_ref_raises() -> None:
    src = _bundle(hash_="src", entities=[
        _ent(uid="obj_1", label="a", source_ref="42", centroid=(0.0, 0.0, 0.0)),
        _ent(uid="obj_2", label="b", source_ref="42", centroid=(1.0, 0.0, 0.0)),
    ])
    tgt = _bundle(hash_="tgt", entities=[])
    try:
        correspond_shared_source_ref(src, tgt)
    except ValueError as e:
        if "duplicate source_instance_ref" not in str(e):
            raise AssertionError(f"unexpected error text: {e}")
        return
    raise AssertionError("expected ValueError for duplicate source_instance_ref")


def test_shared_source_ref_bundle_hashes_recorded() -> None:
    src = _bundle(hash_="src_xyz")
    tgt = _bundle(hash_="tgt_abc")
    c = correspond_shared_source_ref(src, tgt)
    if c.source_bundle_hash != "src_xyz" or c.target_bundle_hash != "tgt_abc":
        raise AssertionError("bundle hashes not recorded")


# ---------- iou_match ----------

def test_iou_match_identical_bboxes() -> None:
    src = _bundle(hash_="src", entities=[
        _ent(uid="obj_1", label="a", source_ref="", centroid=(0.0, 0.0, 0.0)),
    ])
    tgt = _bundle(hash_="tgt", entities=[
        _ent(uid="obj_a", label="a", source_ref="", centroid=(0.0, 0.0, 0.0)),
    ])
    c = correspond_iou_match(src, tgt)
    if c.entity_pairs != [("obj_1", "obj_a")]:
        raise AssertionError(f"expected [(obj_1, obj_a)], got {c.entity_pairs}")
    if c.score["entity:obj_1->obj_a"] != 1.0:
        raise AssertionError(f"identical bboxes should score 1.0, got {c.score}")


def test_iou_match_below_threshold_disjoint_centroids_unmatched() -> None:
    src = _bundle(hash_="src", entities=[
        _ent(uid="obj_1", label="a", source_ref="", centroid=(0.0, 0.0, 0.0),
             half_extent=0.1),
    ])
    tgt = _bundle(hash_="tgt", entities=[
        _ent(uid="obj_a", label="a", source_ref="", centroid=(5.0, 5.0, 5.0),
             half_extent=0.1),
    ])
    c = correspond_iou_match(src, tgt)
    if c.entity_pairs:
        raise AssertionError("disjoint far entities should not match")
    if c.unmatched_source_entities != ["obj_1"]:
        raise AssertionError(f"unmatched_src: {c.unmatched_source_entities}")


def test_iou_match_centroid_fallback_for_small_objects() -> None:
    """Two small objects with no AABB overlap but close centroids
    (within the scale-aware cap) should match via centroid fallback."""
    src = _bundle(hash_="src", entities=[
        _ent(uid="obj_1", label="lamp", source_ref="",
             centroid=(0.0, 0.0, 0.0), half_extent=0.05),
    ])
    # Centroid 0.1 m apart; source bbox is tiny (half_extent=0.05, diag~=0.17).
    # cap = min(0.5 * 0.17, 0.3) = 0.087. 0.1 > 0.087 so centroid fallback fails.
    tgt = _bundle(hash_="tgt", entities=[
        _ent(uid="obj_a", label="lamp", source_ref="",
             centroid=(0.1, 0.0, 0.0), half_extent=0.05),
    ])
    c = correspond_iou_match(src, tgt)
    if c.entity_pairs:
        raise AssertionError(
            "centroid distance 0.1 > scale-aware cap (~0.087) — should NOT match"
        )

    # Now move target within the cap: centroid 0.05 apart.
    tgt2 = _bundle(hash_="tgt2", entities=[
        _ent(uid="obj_a", label="lamp", source_ref="",
             centroid=(0.05, 0.0, 0.0), half_extent=0.05),
    ])
    c2 = correspond_iou_match(src, tgt2)
    if c2.entity_pairs != [("obj_1", "obj_a")]:
        raise AssertionError(
            f"centroid 0.05 within cap should match; got {c2.entity_pairs}"
        )


def test_iou_match_absolute_cap_for_large_objects() -> None:
    """For a large source bbox, scale * diag would be huge but the
    absolute cap (0.30 m default) clamps it."""
    # Source bbox diag is sqrt(3 * 4) = ~3.46; scale * diag = 1.73, far above 0.3 cap
    src = _bundle(hash_="src", entities=[
        _ent(uid="obj_1", label="table", source_ref="",
             centroid=(0.0, 0.0, 0.0), half_extent=1.0),
    ])
    # Target centroid 0.5 m away. cap = min(1.73, 0.3) = 0.3. 0.5 > 0.3 — no match.
    tgt = _bundle(hash_="tgt", entities=[
        _ent(uid="obj_a", label="table", source_ref="",
             centroid=(0.5, 0.0, 0.0), half_extent=0.1),
    ])
    c = correspond_iou_match(src, tgt)
    # IoU is 0 (no overlap). Centroid 0.5 > 0.3 absolute cap. Should not match.
    if c.entity_pairs:
        raise AssertionError("absolute cap should prevent match at 0.5m for large source")


def test_iou_match_greedy_prefers_higher_iou() -> None:
    """When one source could match two targets (one high IoU, one low),
    the high-IoU candidate wins."""
    src = _bundle(hash_="src", entities=[
        _ent(uid="obj_1", label="a", source_ref="",
             centroid=(0.0, 0.0, 0.0), half_extent=0.5),
    ])
    tgt = _bundle(hash_="tgt", entities=[
        # Target A: nearly identical bbox → high IoU
        _ent(uid="obj_high", label="a", source_ref="",
             centroid=(0.05, 0.0, 0.0), half_extent=0.5),
        # Target B: smaller overlap → lower IoU
        _ent(uid="obj_low", label="a", source_ref="",
             centroid=(0.3, 0.0, 0.0), half_extent=0.5),
    ])
    c = correspond_iou_match(src, tgt)
    matched = [p for p in c.entity_pairs if p[0] == "obj_1"]
    if not matched or matched[0][1] != "obj_high":
        raise AssertionError(
            f"greedy assignment should prefer high-IoU match; got {c.entity_pairs}"
        )


def test_iou_match_threshold_respected() -> None:
    """An IoU of 0.1 should not produce an IoU-pass match at the default
    threshold of 0.3 (it may still match via centroid fallback)."""
    src = _bundle(hash_="src", entities=[
        _ent(uid="obj_1", label="a", source_ref="",
             centroid=(0.0, 0.0, 0.0), half_extent=0.5),
    ])
    # Move target far enough that IoU is small AND centroid > cap.
    tgt = _bundle(hash_="tgt", entities=[
        _ent(uid="obj_a", label="a", source_ref="",
             centroid=(0.9, 0.0, 0.0), half_extent=0.5),
    ])
    iou = aabb_iou(src.entities[0].bbox_aabb, tgt.entities[0].bbox_aabb)
    if iou >= 0.3:
        raise AssertionError(f"test premise broken: IoU {iou} >= 0.3")
    c = correspond_iou_match(src, tgt)
    if c.entity_pairs:
        raise AssertionError(
            f"IoU {iou:.3f} below threshold AND centroid 0.9 > 0.3 cap → "
            f"should not match; got {c.entity_pairs}"
        )


def test_iou_match_surfaces_not_attempted_in_phase_1() -> None:
    src = _bundle(hash_="src", surfaces=[_surface("floor_0")])
    tgt = _bundle(hash_="tgt", surfaces=[_surface("floor_0")])
    c = correspond_iou_match(src, tgt)
    if c.surface_pairs:
        raise AssertionError(
            "iou_match should not match surfaces in Phase 1; got "
            f"{c.surface_pairs}"
        )
    if c.unmatched_source_surfaces != ["floor_0"]:
        raise AssertionError(
            f"unmatched surfaces should list both: {c.unmatched_source_surfaces}"
        )


# ---------- manual ----------

def _write_manual_pairs(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_manual_basic_load() -> None:
    src = _bundle(hash_="src", entities=[
        _ent(uid="obj_1", label="a", source_ref="", centroid=(0.0, 0.0, 0.0)),
    ], surfaces=[_surface("floor_src")])
    tgt = _bundle(hash_="tgt", entities=[
        _ent(uid="obj_a", label="a", source_ref="", centroid=(5.0, 0.0, 0.0)),
    ], surfaces=[_surface("floor_tgt")])
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pairs.json"
        _write_manual_pairs(p, {
            "entity_pairs": [["obj_1", "obj_a"]],
            "surface_pairs": [["floor_src", "floor_tgt"]],
        })
        c = correspond_manual(src, tgt, p)
    if c.entity_pairs != [("obj_1", "obj_a")]:
        raise AssertionError(f"entity_pairs: {c.entity_pairs}")
    if c.surface_pairs != [("floor_src", "floor_tgt")]:
        raise AssertionError(f"surface_pairs: {c.surface_pairs}")
    if c.method != "manual":
        raise AssertionError(f"method: {c.method}")


def test_manual_unknown_source_entity_raises() -> None:
    src = _bundle(hash_="src", entities=[
        _ent(uid="obj_1", label="a", source_ref="", centroid=(0.0, 0.0, 0.0)),
    ])
    tgt = _bundle(hash_="tgt", entities=[
        _ent(uid="obj_a", label="a", source_ref="", centroid=(0.0, 0.0, 0.0)),
    ])
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pairs.json"
        _write_manual_pairs(p, {
            "entity_pairs": [["obj_99", "obj_a"]],
        })
        try:
            correspond_manual(src, tgt, p)
        except ValueError as e:
            if "unknown source entity" not in str(e):
                raise AssertionError(f"unexpected error text: {e}")
            return
    raise AssertionError("expected ValueError for unknown source entity")


def test_manual_unknown_target_entity_raises() -> None:
    src = _bundle(hash_="src", entities=[
        _ent(uid="obj_1", label="a", source_ref="", centroid=(0.0, 0.0, 0.0)),
    ])
    tgt = _bundle(hash_="tgt", entities=[])
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pairs.json"
        _write_manual_pairs(p, {
            "entity_pairs": [["obj_1", "obj_99"]],
        })
        try:
            correspond_manual(src, tgt, p)
        except ValueError as e:
            if "unknown target entity" not in str(e):
                raise AssertionError(f"unexpected error text: {e}")
            return
    raise AssertionError("expected ValueError for unknown target entity")


def test_manual_unknown_surface_raises() -> None:
    src = _bundle(hash_="src", surfaces=[_surface("floor_0")])
    tgt = _bundle(hash_="tgt", surfaces=[_surface("floor_0")])
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pairs.json"
        _write_manual_pairs(p, {
            "surface_pairs": [["wall_99", "floor_0"]],
        })
        try:
            correspond_manual(src, tgt, p)
        except ValueError as e:
            if "unknown source surface" not in str(e):
                raise AssertionError(f"unexpected error text: {e}")
            return
    raise AssertionError("expected ValueError for unknown source surface")


def test_manual_empty_pairs_returns_empty_correspondence() -> None:
    src = _bundle(hash_="src", entities=[
        _ent(uid="obj_1", label="a", source_ref="", centroid=(0.0, 0.0, 0.0)),
    ])
    tgt = _bundle(hash_="tgt", entities=[
        _ent(uid="obj_a", label="a", source_ref="", centroid=(0.0, 0.0, 0.0)),
    ])
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pairs.json"
        _write_manual_pairs(p, {"entity_pairs": [], "surface_pairs": []})
        c = correspond_manual(src, tgt, p)
    if c.entity_pairs or c.surface_pairs:
        raise AssertionError("expected no pairs from empty manual file")
    if c.unmatched_source_entities != ["obj_1"]:
        raise AssertionError(f"unmatched_src: {c.unmatched_source_entities}")


# ---------- end-to-end: oracle vs oracle on Replica ----------

def test_iou_match_on_two_oracle_replica_runs_matches_all_entities() -> None:
    """Run the oracle Replica extractor twice (same input → same output)
    and confirm iou_match pairs all 73 entities. This is a sanity check
    that the IoU path works on real fixture data."""
    from adapters.base import ReconstructionConfig
    from adapters.oracle_replica import (
        OracleReplicaAdapter, build_replica_capture_bundle,
    )
    from extractors.base import InstanceExtractorConfig
    from extractors.oracle_replica import OracleReplicaExtractor
    from representations.mesh import MeshRepresentation

    scene_dir = REPO_ROOT / "scenes" / "replica_room_0"
    cap = build_replica_capture_bundle(scene_dir)
    repr_bundle = OracleReplicaAdapter().reconstruct(
        cap, ReconstructionConfig(name="oracle_replica", version="0.1", params={}),
    )
    representation = MeshRepresentation(bundle=repr_bundle)
    a = OracleReplicaExtractor().extract(
        representation, InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )
    b = OracleReplicaExtractor().extract(
        representation, InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )
    c = correspond_iou_match(a, b)
    if len(c.entity_pairs) != 73:
        raise AssertionError(
            f"expected all 73 entities to match between identical oracle runs; "
            f"got {len(c.entity_pairs)}"
        )
    # IoU should be 1.0 for every pair since the bundles are identical.
    if not all(s == 1.0 for s in c.score.values()):
        bad = [k for k, v in c.score.items() if v != 1.0]
        raise AssertionError(f"expected IoU=1.0 for all matches; non-1 keys: {bad[:5]}")


def test_shared_source_ref_on_two_oracle_replica_runs_matches_all() -> None:
    """source_instance_ref is preserved across reruns of the oracle
    extractor, so shared_source_ref should also match all 73 entities."""
    from adapters.base import ReconstructionConfig
    from adapters.oracle_replica import (
        OracleReplicaAdapter, build_replica_capture_bundle,
    )
    from extractors.base import InstanceExtractorConfig
    from extractors.oracle_replica import OracleReplicaExtractor
    from representations.mesh import MeshRepresentation

    scene_dir = REPO_ROOT / "scenes" / "replica_room_0"
    cap = build_replica_capture_bundle(scene_dir)
    repr_bundle = OracleReplicaAdapter().reconstruct(
        cap, ReconstructionConfig(name="oracle_replica", version="0.1", params={}),
    )
    representation = MeshRepresentation(bundle=repr_bundle)
    a = OracleReplicaExtractor().extract(
        representation, InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )
    b = OracleReplicaExtractor().extract(
        representation, InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )
    c = correspond_shared_source_ref(a, b)
    if len(c.entity_pairs) != 73:
        raise AssertionError(
            f"expected all 73 entities to match by source_instance_ref; "
            f"got {len(c.entity_pairs)}"
        )


TESTS = [
    test_aabb_iou_identical_boxes,
    test_aabb_iou_disjoint,
    test_aabb_iou_half_overlap,
    test_aabb_diag_unit_cube,
    test_centroid_distance,
    test_shared_source_ref_matches_entities,
    test_shared_source_ref_matches_surfaces,
    test_shared_source_ref_skips_empty_refs,
    test_shared_source_ref_duplicate_source_ref_raises,
    test_shared_source_ref_bundle_hashes_recorded,
    test_iou_match_identical_bboxes,
    test_iou_match_below_threshold_disjoint_centroids_unmatched,
    test_iou_match_centroid_fallback_for_small_objects,
    test_iou_match_absolute_cap_for_large_objects,
    test_iou_match_greedy_prefers_higher_iou,
    test_iou_match_threshold_respected,
    test_iou_match_surfaces_not_attempted_in_phase_1,
    test_manual_basic_load,
    test_manual_unknown_source_entity_raises,
    test_manual_unknown_target_entity_raises,
    test_manual_unknown_surface_raises,
    test_manual_empty_pairs_returns_empty_correspondence,
    test_iou_match_on_two_oracle_replica_runs_matches_all_entities,
    test_shared_source_ref_on_two_oracle_replica_runs_matches_all,
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

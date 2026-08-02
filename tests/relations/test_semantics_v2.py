"""semantics_v2 S1 tests: D1 / D2 / D3 definitions on synthetic geometry.

Run: python tests/relations/test_semantics_v2.py
Protocol: docs/semantics_v2_track_protocol.md (frozen constants).
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.types import Plane, SceneFrame
from extractors.base import (
    EntityArtifact, EntityArtifacts, EntityIdentity, ExtractionDiagnostics,
    SemanticHypothesis, StructuralSurface,
)
from extractors.serde import CURRENT_SCHEMA_VERSION as ENT_SCHEMA_VERSION
from graph.relations.attached_to import AttachedToConfig, AttachedToExtractor
from graph.relations.attached_to_v2 import AttachedToV2Config, AttachedToV2Extractor
from graph.relations.on_entity_surface_v2 import (
    OnEntitySurfaceV2Config, OnEntitySurfaceV2Extractor,
    V2_SUPPORT_CLASS_ALLOWLIST,
)


def _frame():
    return SceneFrame(gravity=(0.0, 0.0, -1.0), canonical_forward=None,
                      canonical_right=None, units="meters", notes="")


def _entity(uid, label, lo, hi):
    cx = tuple((lo[i] + hi[i]) / 2 for i in range(3))
    return EntityArtifact(
        identity=EntityIdentity(object_uid=uid, display_label=label,
                                aliases=[], source_instance_ref=uid),
        bbox_aabb=(lo, hi), bbox_obb=None, centroid=cx,
        geometry_handle=None,
        semantic_hypotheses=[SemanticHypothesis(label=label, confidence=1.0,
                                                source="t")],
        embedding=None, extraction_diagnostics={},
    )


def _wall_y0():
    return StructuralSurface(
        surface_uid="wall_y0", surface_type="wall",
        plane=Plane(a=0.0, b=1.0, c=0.0, d=0.0),
        polygon=[(-3.0, 0.0, 0.0), (3.0, 0.0, 0.0),
                 (3.0, 0.0, 2.5), (-3.0, 0.0, 2.5)],
        confidence=1.0, source="habitat_label")


def _wall_x0():
    return StructuralSurface(
        surface_uid="wall_x0", surface_type="wall",
        plane=Plane(a=1.0, b=0.0, c=0.0, d=0.0),
        polygon=[(0.0, -3.0, 0.0), (0.0, 3.0, 0.0),
                 (0.0, 3.0, 2.5), (0.0, -3.0, 2.5)],
        confidence=1.0, source="habitat_label")


def _floor():
    return StructuralSurface(
        surface_uid="floor_0", surface_type="floor",
        plane=Plane(a=0.0, b=0.0, c=1.0, d=0.0),
        polygon=[(-3.0, -3.0, 0.0), (3.0, -3.0, 0.0),
                 (3.0, 3.0, 0.0), (-3.0, 3.0, 0.0)],
        confidence=1.0, source="habitat_label")


def _arts(entities, surfaces):
    return EntityArtifacts(
        schema_version=ENT_SCHEMA_VERSION, bundle_hash="ent_t", scene_id="t",
        frame=_frame(), representation_hash="rep_t",
        extractor_name="test", extractor_version="0.0",
        entities=entities, structural_surfaces=surfaces,
        geometry_store_path=None,
        diagnostics=ExtractionDiagnostics(
            n_entities=len(entities), n_structural_surfaces=len(surfaces),
            runtime_seconds=0.0, coverage_score=None, notes=""),
        notes={},
    )


def _attached(entities):
    edges, _ = AttachedToV2Extractor().extract(
        _arts(entities, [_wall_y0(), _floor()]), AttachedToV2Config())
    return edges


def test_d1_elevated_mount_and_widened_threshold():
    vent = _entity("vent", "vent", (0.5, 0.0, 1.0), (0.9, 0.06, 1.2))
    gap_vent = _entity("gap_vent", "vent", (1.2, 0.08, 1.0), (1.6, 0.14, 1.2))
    edges = _attached([vent, gap_vent])
    got = {(e.source.uid, e.evidence["mount_disjunct"]) for e in edges}
    if got != {("vent", "a_elevated"), ("gap_vent", "a_elevated")}:
        raise AssertionError(f"elevated mounts wrong: {got}")
    # the 0.08 m gap case must NOT fire under the frozen v1 2 cm semantics
    v1_edges, _ = AttachedToExtractor().extract(
        _arts([gap_vent], [_wall_y0(), _floor()]), AttachedToConfig())
    if v1_edges:
        raise AssertionError("v1 must reject the 8 cm gap (2 cm band)")


def test_d1_low_thin_panel_vs_low_deep_box():
    panel = _entity("panel", "blinds", (0.5, 0.0, -0.05), (1.2, 0.07, 0.35))
    box = _entity("box", "cabinet", (1.5, 0.0, 0.0), (2.1, 0.30, 0.25))
    edges = _attached([panel, box])
    got = {(e.source.uid, e.evidence["mount_disjunct"]) for e in edges}
    if got != {("panel", "b_low_thin_panel")}:
        raise AssertionError(f"thin-panel disjunct wrong: {got}")


def test_d1_deep_furniture_rejected_and_per_pair_edges():
    sofa = _entity("sofa", "sofa", (0.5, 0.0, 0.0), (2.0, 0.9, 0.9))
    corner = _entity("corner_vent", "vent", (0.0, 0.0, 1.5), (0.06, 0.06, 1.7))
    edges, _ = AttachedToV2Extractor().extract(
        _arts([sofa, corner], [_wall_y0(), _wall_x0(), _floor()]),
        AttachedToV2Config())
    by_src = {}
    for e in edges:
        by_src.setdefault(e.source.uid, []).append(e.target.uid)
    if "sofa" in by_src:
        raise AssertionError(f"deep furniture must not attach: {by_src}")
    if sorted(by_src.get("corner_vent", [])) != ["wall_x0", "wall_y0"]:
        raise AssertionError(f"per-pair emission wrong: {by_src}")


def _support(entities):
    edges, _ = OnEntitySurfaceV2Extractor().extract(
        _arts(entities, [_floor()]), OnEntitySurfaceV2Config())
    return edges


def test_d2_contained_rest_and_smallest_footprint():
    sofa = _entity("sofa", "sofa", (0.0, 0.5, 0.0), (2.0, 1.4, 0.9))
    bed = _entity("bed", "bed", (-0.5, 0.0, 0.0), (3.0, 2.0, 0.6))
    cushion = _entity("cushion", "cushion", (0.5, 0.7, 0.5), (1.0, 1.2, 0.85))
    edges = _support([sofa, bed, cushion])
    contained = [e for e in edges
                 if e.evidence.get("disjunct") == "contained_rest"]
    pairs = {(e.source.uid, e.target.uid) for e in contained}
    if ("cushion", "sofa") not in pairs:
        raise AssertionError(f"cushion must rest in the sofa: {pairs}")
    if any(s == "cushion" and t == "bed" for s, t in pairs):
        raise AssertionError("must assign to the SMALLEST-footprint supporter")


def test_d2_floor_supported_excluded():
    sofa = _entity("sofa", "sofa", (0.0, 0.5, 0.0), (2.0, 1.4, 0.9))
    basket = _entity("basket", "basket", (0.6, 0.8, 0.005), (0.9, 1.1, 0.3))
    edges = _support([sofa, basket])
    if any(e.source.uid == "basket" for e in edges):
        raise AssertionError("floor-supported entity must not contained-rest")


def test_d3_new_anchor_classes_and_no_duplicates():
    if not {"cabinet", "nightstand", "bed"} <= set(V2_SUPPORT_CLASS_ALLOWLIST):
        raise AssertionError("D3 classes missing from the v2 allowlist")
    nightstand = _entity("nightstand", "nightstand",
                         (0.0, 0.5, 0.0), (0.6, 1.1, 0.55))
    lamp = _entity("lamp", "lamp", (0.2, 0.7, 0.55), (0.4, 0.9, 0.9))
    edges = _support([nightstand, lamp])
    pairs = [(e.source.uid, e.target.uid) for e in edges
             if e.source.uid == "lamp"]
    if pairs.count(("lamp", "nightstand")) != 1:
        raise AssertionError(f"lamp-on-nightstand must be exactly one edge "
                             f"(v1 top-rest, no contained duplicate): {pairs}")


TESTS = [
    test_d1_elevated_mount_and_widened_threshold,
    test_d1_low_thin_panel_vs_low_deep_box,
    test_d1_deep_furniture_rejected_and_per_pair_edges,
    test_d2_contained_rest_and_smallest_footprint,
    test_d2_floor_supported_excluded,
    test_d3_new_anchor_classes_and_no_duplicates,
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

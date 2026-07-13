"""Phase 8 E5 tests: threshold sweep on a synthetic scene (no dataset).

Scene: one wall (y=2, interior -y, from the P5 smoke fixture geometry), one
floor, and two objects:
  - obj_stable   0.005 m off the wall -> in contact at every grid point,
  - obj_fragile  0.024 m off the wall -> out at the default 0.02 band, IN at
                 the 1.25x point -> must be flagged fragile.

Run: python tests/tools/test_threshold_sweep.py
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
from tools.threshold_sweep import GRID, evaluate_variant, sweep_family, variant_run


def _frame() -> SceneFrame:
    return SceneFrame(gravity=(0.0, 0.0, -1.0), canonical_forward=None,
                      canonical_right=None, units="meters", notes="")


def _entity(uid: str, lo, hi, centroid) -> EntityArtifact:
    return EntityArtifact(
        identity=EntityIdentity(object_uid=uid, display_label=uid,
                                aliases=[], source_instance_ref=uid),
        bbox_aabb=(lo, hi), bbox_obb=None, centroid=centroid,
        geometry_handle=None,
        semantic_hypotheses=[SemanticHypothesis(label=uid, confidence=1.0, source="t")],
        embedding=None, extraction_diagnostics={},
    )


def _wall() -> StructuralSurface:
    return StructuralSurface(
        surface_uid="synth_wall_north", surface_type="wall",
        plane=Plane(a=0.0, b=-1.0, c=0.0, d=2.0),
        polygon=[(0.0, 2.0, 0.0), (2.0, 2.0, 0.0), (2.0, 2.0, 2.0), (0.0, 2.0, 2.0)],
        confidence=1.0, source="habitat_label",
    )


def _floor() -> StructuralSurface:
    return StructuralSurface(
        surface_uid="synth_floor", surface_type="floor",
        plane=Plane(a=0.0, b=0.0, c=1.0, d=0.0),
        polygon=[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)],
        confidence=1.0, source="habitat_label",
    )


def _artifacts() -> EntityArtifacts:
    entities = [
        _entity("obj_stable", (0.8, 1.895, 0.5), (1.2, 1.995, 0.9), (1.0, 1.945, 0.7)),
        _entity("obj_fragile", (0.8, 1.926, 0.5), (1.2, 1.976, 0.9), (1.0, 1.951, 0.7)),
    ]
    surfaces = [_wall(), _floor()]
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


FRAGILE_EDGE = ["CONTACTS_SURFACE", "obj_fragile", "synth_wall_north"]


def test_baseline_default_edges():
    arts = _artifacts()
    edges, answers = evaluate_variant(
        arts, variant_run("contacts_surface"), ("what is against the wall?",))
    if ("CONTACTS_SURFACE", "obj_stable", "synth_wall_north") not in edges:
        raise AssertionError(f"stable object must contact at default band: {edges}")
    if tuple(FRAGILE_EDGE) in edges:
        raise AssertionError("fragile object (0.024 m gap) must NOT contact at 0.02")
    if answers["what is against the wall?"] != ["obj_stable"]:
        raise AssertionError(f"unexpected baseline answer: {answers}")


def test_sweep_flags_fragile_edge():
    out = sweep_family(_artifacts(), "contacts_surface")
    if FRAGILE_EDGE not in out["fragile_edges"]:
        raise AssertionError(f"fragile edge missing: {out['fragile_edges']}")
    pts = out["params"]["contact_threshold_m"]["points"]
    if pts["1.25"]["edges_added"] != [FRAGILE_EDGE]:
        raise AssertionError(f"1.25x should add exactly the fragile edge: {pts['1.25']}")
    flips = pts["1.25"]["answer_flips"]
    if flips.get("what is against the wall?", {}).get("gained") != ["obj_fragile"]:
        raise AssertionError(f"1.25x answer flip not recorded: {flips}")


def test_sweep_x1_point_matches_default():
    out = sweep_family(_artifacts(), "contacts_surface")
    for param, data in out["params"].items():
        p = data["points"]["1.0"]
        if p["edges_added"] or p["edges_removed"] or p["answer_flips"]:
            raise AssertionError(f"x1.0 must be identical to default for {param}: {p}")
    if out["baseline"]["edge_count"] != 1:
        raise AssertionError(f"expected 1 baseline edge: {out['baseline']}")


def test_stable_edge_survives_tight_band():
    out = sweep_family(_artifacts(), "contacts_surface")
    pts = out["params"]["contact_threshold_m"]["points"]
    if pts["0.5"]["edges_removed"]:
        raise AssertionError(
            f"stable object (0.005 m gap) must survive the 0.5x band: {pts['0.5']}")


def test_grid_shape():
    if GRID != (0.5, 0.75, 1.0, 1.25, 1.5):
        raise AssertionError(f"grid changed — update fixtures/docs: {GRID}")


TESTS = [
    test_baseline_default_edges,
    test_sweep_flags_fragile_edge,
    test_sweep_x1_point_matches_default,
    test_stable_edge_survives_tight_band,
    test_grid_shape,
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

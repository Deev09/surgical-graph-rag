"""Phase 8 F4 tests: opt-in room-scale-flat exclusion (wall-to-wall rugs).

Run: python tests/relations/test_room_scale_flat.py

Synthetic 6x6 room: four walls + floor, a wall-to-wall rug (area_frac 0.99,
height 0.05), a cabinet against one wall, and a room-spanning tall wardrobe
(area_frac 0.72, height 2.0 -- the height gate must keep it). Properties:
  1. defaults (flag off) preserve frozen behavior AND the config-hash payload;
  2. flag on: rug excluded from CONTACTS_SURFACE, cabinet edge intact;
  3. flag on: rug excluded from NEAR_SURFACE wall edges ONLY -- its
     near-floor edge survives, and CONTACTS stays a subset of NEAR(wall);
  4. the height gate keeps tall large-footprint furniture;
  5. defaults mirror across the two configs; validation rejects bad values.
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
from graph.builder import _config_hash_payload
from graph.relations.contacts_surface import (
    DEFAULT_ROOM_SCALE_FLAT_MAX_HEIGHT_M,
    DEFAULT_ROOM_SCALE_FLAT_MIN_AREA_FRAC,
    ContactsSurfaceConfig,
    ContactsSurfaceExtractor,
)
from graph.relations.surface import SurfaceProximityConfig, SurfaceProximityExtractor


def _frame() -> SceneFrame:
    return SceneFrame(
        gravity=(0.0, 0.0, -1.0), canonical_forward=None,
        canonical_right=None, units="meters", notes="",
    )


def _entity(uid: str, lo, hi) -> EntityArtifact:
    centroid = tuple((lo[i] + hi[i]) / 2.0 for i in range(3))
    return EntityArtifact(
        identity=EntityIdentity(
            object_uid=uid, display_label=uid, aliases=[], source_instance_ref=uid,
        ),
        bbox_aabb=(tuple(lo), tuple(hi)), bbox_obb=None, centroid=centroid,
        geometry_handle=None,
        semantic_hypotheses=[SemanticHypothesis(label=uid, confidence=1.0, source="t")],
        embedding=None, extraction_diagnostics={},
    )


def _wall(uid: str, a: float, b: float, d: float, polygon) -> StructuralSurface:
    return StructuralSurface(
        surface_uid=uid, surface_type="wall",
        plane=Plane(a=a, b=b, c=0.0, d=d),
        polygon=[tuple(v) for v in polygon],
        confidence=1.0, source="habitat_label",
    )


def _room_surfaces() -> list[StructuralSurface]:
    # 6x6 room, walls at x=+-3 / y=+-3 (normals pointing inward), floor z=0
    h = 2.5
    return [
        _wall("wall_xp", -1.0, 0.0, 3.0,
              [(3, -3, 0), (3, 3, 0), (3, 3, h), (3, -3, h)]),
        _wall("wall_xn", 1.0, 0.0, 3.0,
              [(-3, -3, 0), (-3, 3, 0), (-3, 3, h), (-3, -3, h)]),
        _wall("wall_yp", 0.0, -1.0, 3.0,
              [(-3, 3, 0), (3, 3, 0), (3, 3, h), (-3, 3, h)]),
        _wall("wall_yn", 0.0, 1.0, 3.0,
              [(-3, -3, 0), (3, -3, 0), (3, -3, h), (-3, -3, h)]),
        StructuralSurface(
            surface_uid="floor_1", surface_type="floor",
            plane=Plane(a=0.0, b=0.0, c=1.0, d=0.0),
            polygon=[(-3, -3, 0), (3, -3, 0), (3, 3, 0), (-3, 3, 0)],
            confidence=1.0, source="habitat_label"),
    ]


RUG = _entity("rug", (-2.99, -2.99, 0.0), (2.99, 2.99, 0.05))        # frac 0.99
CABINET = _entity("cabinet", (2.61, -0.5, 0.0), (2.99, 0.5, 1.0))    # against wall_xp
WARDROBE = _entity("wardrobe", (-2.99, -2.4, 0.0), (2.99, 2.4, 2.0))  # frac 0.80, tall


def _artifacts(entities) -> EntityArtifacts:
    surfaces = _room_surfaces()
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


def _contacts(entities, config):
    edges, diag = ContactsSurfaceExtractor().extract(_artifacts(entities), config)
    return edges, diag


def _near(entities, config):
    edges, diag = SurfaceProximityExtractor().extract(_artifacts(entities), config)
    return edges, diag


def test_defaults_off_preserve_behavior_and_hash():
    edges, _ = _contacts([RUG, CABINET], ContactsSurfaceConfig())
    rug_walls = {e.target.uid for e in edges if e.source.uid == "rug"}
    if len(rug_walls) != 4:
        raise AssertionError(
            f"frozen behavior: wall-to-wall rug must contact all 4 walls, got {rug_walls}")
    for payload in (
        _config_hash_payload(ContactsSurfaceConfig()),
        _config_hash_payload(SurfaceProximityConfig(use_polygon_clip=True)),
    ):
        leaked = [k for k in payload if k.startswith(("exclude_room", "room_scale"))]
        if leaked:
            raise AssertionError(f"default config hash payload must not change: {leaked}")
    enabled = _config_hash_payload(ContactsSurfaceConfig(exclude_room_scale_flat=True))
    if "exclude_room_scale_flat" not in enabled:
        raise AssertionError("non-default flag must be a hash input")


def test_flag_on_excludes_rug_from_contacts():
    cfg = ContactsSurfaceConfig(exclude_room_scale_flat=True)
    edges, diag = _contacts([RUG, CABINET], cfg)
    by_src = {}
    for e in edges:
        by_src.setdefault(e.source.uid, set()).add(e.target.uid)
    if "rug" in by_src:
        raise AssertionError(f"rug must be excluded from CONTACTS_SURFACE: {by_src['rug']}")
    if by_src.get("cabinet") != {"wall_xp"}:
        raise AssertionError(f"cabinet edge must survive: {by_src.get('cabinet')}")
    # the floor still records its usual surface_type_not_wall rejection;
    # every rug-vs-WALL pair must be the F4 exclusion, never a clause failure
    wall_reasons = {r.rejected_reason for r in diag.rejection_samples
                    if r.source.uid == "rug" and r.target.uid.startswith("wall")}
    if wall_reasons != {"room_scale_flat_excluded"}:
        raise AssertionError(f"expected room_scale_flat_excluded rejections, got {wall_reasons}")
    sample = next(r for r in diag.rejection_samples
                  if r.source.uid == "rug" and r.target.uid.startswith("wall"))
    if sample.evidence.get("area_frac") is None or sample.evidence["area_frac"] < 0.9:
        raise AssertionError(f"exclusion evidence must carry area_frac: {sample.evidence}")


def test_flag_on_near_wall_excluded_floor_preserved():
    cfg = SurfaceProximityConfig(use_polygon_clip=True, exclude_room_scale_flat=True)
    edges, diag = _near([RUG, CABINET], cfg)
    rug_targets = {e.target.uid for e in edges if e.source.uid == "rug"}
    if any(t.startswith("wall") for t in rug_targets):
        raise AssertionError(f"rug must lose NEAR wall edges: {rug_targets}")
    if "floor_1" not in rug_targets:
        raise AssertionError(f"rug near-FLOOR edge must survive: {rug_targets}")
    reasons = {r.rejected_reason for r in diag.rejection_samples if r.source.uid == "rug"}
    if "room_scale_flat_excluded" not in reasons:
        raise AssertionError(f"expected room_scale_flat_excluded rejections, got {reasons}")
    # subset guard: CONTACTS(flag on) still a subset of NEAR-wall(flag on)
    c_edges, _ = _contacts([RUG, CABINET], ContactsSurfaceConfig(exclude_room_scale_flat=True))
    c_pairs = {(e.source.uid, e.target.uid) for e in c_edges}
    n_pairs = {(e.source.uid, e.target.uid) for e in edges
               if e.target.uid.startswith("wall")}
    if not c_pairs <= n_pairs:
        raise AssertionError(f"CONTACTS must remain subset of NEAR(wall): {c_pairs - n_pairs}")


def test_height_gate_keeps_tall_furniture():
    cfg = ContactsSurfaceConfig(exclude_room_scale_flat=True)
    edges, _ = _contacts([WARDROBE], cfg)
    if not edges:
        raise AssertionError(
            "room-spanning but TALL furniture (frac 0.72, h 2.0) must not be excluded")


def test_defaults_mirror_and_validation():
    sp = SurfaceProximityConfig()
    if (sp.room_scale_flat_min_area_frac != DEFAULT_ROOM_SCALE_FLAT_MIN_AREA_FRAC
            or sp.room_scale_flat_max_height_m != DEFAULT_ROOM_SCALE_FLAT_MAX_HEIGHT_M):
        raise AssertionError("F4 defaults drifted between the two configs")
    for bad in (0.0, -1.0, float("inf")):
        try:
            ContactsSurfaceConfig(exclude_room_scale_flat=True,
                                  room_scale_flat_min_area_frac=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"min_area_frac={bad} must raise")


TESTS = [
    test_defaults_off_preserve_behavior_and_hash,
    test_flag_on_excludes_rug_from_contacts,
    test_flag_on_near_wall_excluded_floor_preserved,
    test_height_gate_keeps_tall_furniture,
    test_defaults_mirror_and_validation,
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

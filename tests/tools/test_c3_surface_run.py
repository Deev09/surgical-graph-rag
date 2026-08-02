"""C3.0-S runner/artifact tests; synthetic and frozen-report inputs only."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.types import Plane, SceneFrame
from extractors.base import (
    EntityArtifact, EntityArtifacts, EntityIdentity, ExtractionDiagnostics,
    StructuralSurface,
)
from extractors.serde import dump_entity_artifacts, load_entity_artifacts
from tools.c3_surface_run import (
    _candidate_arts, _surface_slice, artifact_payload,
    surface_geometry_metrics, verify_artifact,
)


def _surface(uid="mesh_floor_0", st="floor", z=0.0, source="mesh_region_fit"):
    if st in ("floor", "ceiling"):
        normal = (0.0, 0.0, 1.0 if st == "floor" else -1.0)
        d = -z * normal[2]
        poly = [(0.0, 0.0, z), (2.0, 0.0, z),
                (2.0, 2.0, z), (0.0, 2.0, z)]
    else:
        normal, d = (1.0, 0.0, 0.0), 0.0
        poly = [(0.0, 0.0, 0.0), (0.0, 2.0, 0.0),
                (0.0, 2.0, 2.0), (0.0, 0.0, 2.0)]
    return StructuralSurface(
        surface_uid=uid, surface_type=st,
        plane=Plane(a=normal[0], b=normal[1], c=normal[2], d=d),
        polygon=poly, confidence=1.0, source=source)


def _bundle(surface):
    ent = EntityArtifact(
        identity=EntityIdentity("obj_1", "chair", [], "synthetic:1"),
        bbox_aabb=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), bbox_obb=None,
        centroid=(0.5, 0.5, 0.5), geometry_handle=None)
    return EntityArtifacts(
        schema_version=2, bundle_hash="b_original", scene_id="synthetic",
        frame=SceneFrame(gravity=(0.0, 0.0, -1.0), canonical_forward=None,
                         canonical_right=None, units="meters", notes="test"),
        representation_hash="r_original", extractor_name="test",
        extractor_version="1", entities=[ent], structural_surfaces=[surface],
        geometry_store_path=None,
        diagnostics=ExtractionDiagnostics(1, 1, 0.0, 1.0, "test"), notes={})


def test_artifact_hash_is_deterministic_and_tamper_evident():
    frame = {"world_from_raw_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
             "world_from_raw_translation": [0, 0, 0], "yaw_derotation_deg": 0}
    args = ("replica_room_2", "room_2/mesh.ply", "a" * 64, frame,
            [_surface()], {"n_faces": 2})
    a, b = artifact_payload(*args), artifact_payload(*args)
    if a != b:
        raise AssertionError("artifact payload is nondeterministic")
    verify_artifact(a)
    bad = json.loads(json.dumps(a))
    bad["surfaces"][0]["confidence"] = 0.5
    try:
        verify_artifact(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered artifact was accepted")


def test_runner_import_does_not_load_oracle_or_evaluator_modules():
    forbidden = [
        "demo.replica_habitat_import",
        "demo.replica_mesh_import",
        "graph.builder",
        "reasoner.router",
        "tools.c1_joint_ceiling",
    ]
    code = (
        "import sys; import tools.c3_surface_run; "
        f"bad=[name for name in {forbidden!r} if name in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)


def test_identical_surface_geometry_scores_perfectly():
    surfaces = [_surface(), _surface("mesh_wall_00", "wall"),
                _surface("mesh_ceiling_0", "ceiling", 3.0)]
    metrics = surface_geometry_metrics(surfaces, surfaces)
    for st in ("floor", "wall", "ceiling"):
        row = metrics["by_type"][st]
        if row["oracle_area_coverage"] != 1.0 or row["estimated_spill"] != 0.0:
            raise AssertionError(f"identical {st} geometry not perfect: {row}")
    if metrics["compatible_plane_angular_error_median_deg"] != 0.0:
        raise AssertionError("identical planes have angular error")


def test_surface_override_preserves_frozen_bundle():
    original_surface = _surface("floor_oracle", source="habitat_label")
    b = _bundle(original_surface)
    c = _candidate_arts(b, [_surface()], "f" * 64)
    if b.bundle_hash != "b_original" or b.structural_surfaces != [original_surface]:
        raise AssertionError("candidate construction mutated frozen B")
    if c.bundle_hash == b.bundle_hash or c.entities is not b.entities:
        raise AssertionError("candidate must differ only by isolated surface payload")
    if c.structural_surfaces[0].source != "mesh_region_fit":
        raise AssertionError("candidate provenance missing")


def test_mesh_region_fit_round_trips_entity_serde():
    bundle = _bundle(_surface())
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        dump_entity_artifacts(bundle, root)
        loaded = load_entity_artifacts(root)
        if loaded.structural_surfaces != bundle.structural_surfaces:
            raise AssertionError("mesh_region_fit surface failed serde round-trip")


def test_room2_frozen_surface_slice_anchor():
    report = json.loads((REPO_ROOT / "runs/mvp_v0/replica_room_2_mvp.json").read_text())
    qa = {"per_relation": report["variants"]["B"]["per_relation"]}
    row = _surface_slice(qa)
    if row != {"n_hit": 15, "n_cited": 16, "n_expected": 29,
               "micro_precision": 0.9375, "micro_recall": 0.517241}:
        raise AssertionError(f"protocol anchor drifted: {row}")


TESTS = [
    test_artifact_hash_is_deterministic_and_tamper_evident,
    test_runner_import_does_not_load_oracle_or_evaluator_modules,
    test_identical_surface_geometry_scores_perfectly,
    test_surface_override_preserves_frozen_bundle,
    test_mesh_region_fit_round_trips_entity_serde,
    test_room2_frozen_surface_slice_anchor,
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
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

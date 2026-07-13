"""C1.00-C1.02 tests: segmentation bundle round-trip, exact evaluator, and
anonymous candidate artifacts — all on synthetic binary PLY pairs (no dataset).

Run: python tests/segmenter/test_c1_pipeline.py

Synthetic scene: two unit cubes (oracle object_id 1 = "table" at origin,
object_id 2 = "chair" at x+5), written as BOTH a semantic PLY (faces carry
uint16 object_id) and a raw PLY (no attribution), sharing one vertex array —
the same invariant the real Replica pair satisfies.

Contract gates encoded here: G3 (deterministic round-trip), G4 (boxes
reproduced from assigned vertices), G5 (candidate carries no oracle labels
or surfaces).
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmenter.base import (
    SegmentationOutput, load_segmentation_output, save_segmentation_output,
    sha256_file,
)
from segmenter.candidate import _yaw_obb, build_candidate_artifacts
from segmenter.ply import parse_vertices
from tools.c1_exact_eval import evaluate, oracle_vertex_membership


_CUBE_TRIS = [
    (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
    (0, 4, 5), (0, 5, 1), (3, 2, 6), (3, 6, 7),
    (0, 3, 7), (0, 7, 4), (1, 5, 6), (1, 6, 2),
]


def _cube_verts(offset):
    ox, oy, oz = offset
    return [(ox + x, oy + y, oz + z)
            for x, y, z in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]]


def _two_cube_mesh(with_wall_cube: bool = False):
    """(xyz, tris, oids) — cube A oid=1 "table", cube B oid=2 "chair",
    optional cube C oid=3 "wall" (for the structural-removal test)."""
    offsets = [(0, 0, 0), (5, 0, 0)] + ([(10, 0, 0)] if with_wall_cube else [])
    verts: list[tuple] = []
    for off in offsets:
        verts += _cube_verts(off)
    xyz = np.array(verts, dtype=np.float64)
    tris, oids = [], []
    for j in range(len(offsets)):
        for t in _CUBE_TRIS:
            tris.append(tuple(8 * j + i for i in t))
            oids.append(j + 1)
    return xyz, np.array(tris, dtype=np.int64), np.array(oids, dtype=np.int64)


def _vertex_block(xyz):
    out = bytearray()
    for x, y, z in xyz:
        out += struct.pack("<fff", x, y, z)          # x y z
        out += struct.pack("<fff", 0.0, 0.0, 1.0)    # nx ny nz
        out += struct.pack("BBB", 128, 128, 128)     # r g b
    return bytes(out)


def _header(n_vert, n_face, with_oid):
    lines = ["ply", "format binary_little_endian 1.0",
             f"element vertex {n_vert}"]
    for p in ("x", "y", "z", "nx", "ny", "nz"):
        lines.append(f"property float {p}")
    for p in ("red", "green", "blue"):
        lines.append(f"property uchar {p}")
    lines.append(f"element face {n_face}")
    lines.append("property list uchar uint vertex_indices")
    if with_oid:
        lines.append("property ushort object_id")
    lines.append("end_header")
    return ("\n".join(lines) + "\n").encode("ascii")


def _write_semantic_ply(path, xyz, tris, oids):
    body = bytearray(_vertex_block(xyz))
    for t, o in zip(tris, oids):
        body += struct.pack("<BIIIH", 3, *[int(i) for i in t], int(o))
    path.write_bytes(_header(len(xyz), len(tris), True) + bytes(body))


def _write_raw_ply(path, xyz, tris):
    body = bytearray(_vertex_block(xyz))
    for t in tris:
        body += struct.pack("<BIII", 3, *[int(i) for i in t])
    path.write_bytes(_header(len(xyz), len(tris), False) + bytes(body))


def _scene(tmp: Path, pred_ids: np.ndarray, with_wall_cube: bool = False):
    """Write room dir (raw + semantic + info json) and a saved segmentation
    bundle with the given per-vertex prediction. Returns (room, bundle_dir)."""
    xyz, tris, oids = _two_cube_mesh(with_wall_cube)

    def obj(i, cls, cx):
        return {"id": i, "class_name": cls,
                "oriented_bbox": {
                    "abb": {"center": [cx, 0.5, 0.5], "sizes": [1.0, 1.0, 1.0]},
                    "orientation": {"rotation": [0.0, 0.0, 0.0, 1.0],
                                    "translation": [0.0, 0.0, 0.0]}}}
    objects = [obj(1, "table", 0.5), obj(2, "chair", 5.5)]
    if with_wall_cube:
        objects.append(obj(3, "wall", 10.5))
    room = tmp / "room"
    (room / "habitat").mkdir(parents=True)
    _write_raw_ply(room / "mesh.ply", xyz, tris)
    _write_semantic_ply(room / "habitat" / "mesh_semantic.ply", xyz, tris, oids)
    (room / "habitat" / "info_semantic.json").write_text(json.dumps({
        "gravity_dir": [0.0, 0.0, -1.0],
        "objects": objects,
    }), encoding="utf-8")
    seg = SegmentationOutput(
        input_mesh_sha256=sha256_file(room / "mesh.ply"),
        n_vertices=len(xyz),
        segmenter_name="test_seg", segmenter_version="0.0",
        config_params_json="{}",
        vertex_instance_ids=pred_ids.astype(np.int64),
    ).finalize()
    bundle = tmp / "bundle"
    save_segmentation_output(seg, bundle)
    return room, bundle


PERFECT = np.array([10] * 8 + [20] * 8)  # pred id-space independent of oracle's


def test_bundle_roundtrip_deterministic():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _, bundle = _scene(tmp, PERFECT)
        a = load_segmentation_output(bundle)
        b = load_segmentation_output(bundle)
        if a.output_sha256 != b.output_sha256:
            raise AssertionError("round-trip hash must be deterministic")
        if not np.array_equal(a.vertex_instance_ids, b.vertex_instance_ids):
            raise AssertionError("assignments must round-trip exactly")
        # tampering must be detected (G3)
        ids = np.load(bundle / "vertex_instance_ids.npy")
        ids[0] = 99
        np.save(bundle / "vertex_instance_ids.npy", ids)
        try:
            load_segmentation_output(bundle)
        except ValueError:
            pass
        else:
            raise AssertionError("tampered bundle must fail the hash check")


def test_validation_failures():
    seg = SegmentationOutput(
        input_mesh_sha256="x", n_vertices=10,
        segmenter_name="t", segmenter_version="0",
        config_params_json="{}",
        vertex_instance_ids=np.zeros(9, dtype=np.int64))
    try:
        save_segmentation_output(seg, Path("/nonexistent-should-not-write"))
    except ValueError:
        pass
    else:
        raise AssertionError("length mismatch must raise")
    seg2 = SegmentationOutput(
        input_mesh_sha256="x", n_vertices=4,
        segmenter_name="t", segmenter_version="0",
        config_params_json="{}",
        vertex_instance_ids=np.array([-2, 0, 1, 1]))
    try:
        save_segmentation_output(seg2, Path("/nonexistent-should-not-write"))
    except ValueError:
        pass
    else:
        raise AssertionError("ids below -1 must raise")


def test_oracle_membership_majority_and_ties():
    # vertex 0: faces vote oid 5,5,3 -> majority 5; vertex 1: 3,5 tie -> smallest 3
    vidx = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 3]])
    oid = np.array([5, 5, 3])
    m = oracle_vertex_membership(vidx, oid, 5)
    if m[0] != 5:
        raise AssertionError(f"majority vote failed: {m}")
    if m[1] != 3:
        raise AssertionError(f"tie must go to smallest id: {m}")
    if m[4] != -1:
        raise AssertionError(f"untouched vertex must be background: {m}")


def test_exact_eval_perfect():
    with tempfile.TemporaryDirectory() as td:
        room, bundle = _scene(Path(td), PERFECT)
        r = evaluate(room, bundle)
        if r["n_matched"] != 2 or any(m["iou"] != 1.0 for m in r["matches"]):
            raise AssertionError(f"perfect assignment must match both at IoU 1: {r['matches']}")
        if r["recall_at_iou"]["0.75"] != 1.0:
            raise AssertionError("perfect recall expected")
        if r["support_owner"]["n_oracle_objects"] != 2:  # table + chair
            raise AssertionError(f"support-owner set wrong: {r['support_owner']}")
        if (r["over_segmentation"]["n_oracle_objects_split"]
                or r["under_segmentation"]["n_predictions_merging"]):
            raise AssertionError("perfect assignment must show no split/merge")


def test_exact_eval_split_and_merge():
    with tempfile.TemporaryDirectory() as td:
        # cube A split into two predictions (ids 10/11); cube B merged with
        # nothing but prediction 10 also grabs half of cube B -> merge case
        pred = np.array([10] * 4 + [11] * 4 + [10] * 4 + [-1] * 4)
        room, bundle = _scene(Path(td), pred)
        r = evaluate(room, bundle)
        if r["over_segmentation"]["n_oracle_objects_split"] != 1:
            raise AssertionError(f"cube A must count as split: {r['over_segmentation']}")
        if r["under_segmentation"]["n_predictions_merging"] != 1:
            raise AssertionError(f"pred 10 must count as merging: {r['under_segmentation']}")
        if abs(r["unassigned_vertex_frac_pred"] - 4 / 16) > 1e-9:
            raise AssertionError(f"unassigned fraction wrong: {r}")


def test_candidate_anonymous_and_boxes():
    with tempfile.TemporaryDirectory() as td:
        room, bundle = _scene(Path(td), PERFECT)
        seg = load_segmentation_output(bundle)
        eye = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        arts = build_candidate_artifacts(
            room / "mesh.ply", seg, "synthetic",
            rotation=eye, z_translation=0.0, bundle_dir=bundle, min_vertices=4)
        if len(arts.entities) != 2 or arts.structural_surfaces:
            raise AssertionError("candidate must have 2 segments, 0 surfaces")
        for e in arts.entities:
            if not e.identity.display_label.startswith("segment_"):
                raise AssertionError(f"label must be anonymous: {e.identity.display_label}")
            if e.semantic_hypotheses:
                raise AssertionError("no semantic hypotheses allowed (G5)")
        if arts.notes["semantic_source"] != "none" or arts.notes["surface_source"] != "none":
            raise AssertionError("notes must prove oracle-free provenance (G5)")
        # G4: boxes reproduced from assigned vertices
        xyz = parse_vertices(room / "mesh.ply")
        for e in arts.entities:
            inst = int(e.identity.source_instance_ref.split(":")[1])
            pts = xyz[seg.vertex_instance_ids == inst]
            lo = tuple(float(v) for v in pts.min(axis=0))
            hi = tuple(float(v) for v in pts.max(axis=0))
            if e.bbox_aabb != (lo, hi):
                raise AssertionError(f"box not reproduced from vertices: {e.bbox_aabb}")
            if f"#{inst}" not in (e.geometry_handle or ""):
                raise AssertionError(f"geometry_handle must reference instance: {e.geometry_handle}")


def test_candidate_fail_conditions():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        room, bundle = _scene(tmp, PERFECT)
        seg = load_segmentation_output(bundle)
        eye = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        # wrong mesh -> input hash mismatch
        other = tmp / "other.ply"
        other.write_bytes((room / "mesh.ply").read_bytes() + b"x")
        for kwargs, name in (
            (dict(mesh_path=other), "hash mismatch"),
            (dict(min_vertices=1000), "empty retained set"),
        ):
            try:
                build_candidate_artifacts(
                    kwargs.get("mesh_path", room / "mesh.ply"), seg, "s",
                    rotation=eye, z_translation=0.0,
                    min_vertices=kwargs.get("min_vertices", 4))
            except ValueError:
                pass
            else:
                raise AssertionError(f"{name} must raise")


def test_derived_bundle_injects_and_removes():
    from segmenter.derived import build_c1_eval_bundle
    with tempfile.TemporaryDirectory() as td:
        # third cube is oracle "wall"; predictions: 10->table, 20->chair,
        # 30->wall-cube, 40 claims nothing real... keep 3 perfect segments
        pred = np.array([10] * 8 + [20] * 8 + [30] * 8)
        room, bundle = _scene(Path(td), pred, with_wall_cube=True)
        arts, rep = build_c1_eval_bundle(room, bundle, "synthetic",
                                         z_translation=0.0, min_vertices=4)
        labels = {e.identity.object_uid: e.identity.display_label
                  for e in arts.entities}
        if labels != {"obj_10": "table", "obj_20": "chair"}:
            raise AssertionError(f"oracle labels must inject onto matches: {labels}")
        if rep["n_removed_structural_or_dropped"] != 1:
            raise AssertionError(f"wall-matched segment must be removed+recorded: {rep}")
        if rep["removed_structural_or_dropped"][0]["oracle_class"] != "wall":
            raise AssertionError(f"removal record wrong: {rep}")
        if arts.bundle_hash.startswith("c1cand_") or not arts.bundle_hash.startswith("c1eval_"):
            raise AssertionError(f"derived bundle needs a distinct hash: {arts.bundle_hash}")
        if arts.notes["semantic_source"] != "oracle_correspondence":
            raise AssertionError("provenance must say oracle_correspondence")
        if "isolation_statement" not in arts.notes:
            raise AssertionError("isolation statement required on enriched bundles")


def test_derived_bundle_keeps_unmatched_anonymous():
    from segmenter.derived import build_c1_eval_bundle
    with tempfile.TemporaryDirectory() as td:
        # prediction 50 covers only background-free... split cube A: 10 gets
        # half, 50 gets the other half -> 50 loses the greedy match and must
        # SURVIVE as an anonymous segment, not be dropped
        pred = np.array([10] * 4 + [50] * 4 + [20] * 8)
        room, bundle = _scene(Path(td), pred)
        arts, rep = build_c1_eval_bundle(room, bundle, "synthetic",
                                         z_translation=0.0, min_vertices=4)
        labels = {e.identity.object_uid: e.identity.display_label
                  for e in arts.entities}
        anon = [u for u, l in labels.items() if l.startswith("segment_")]
        if len(anon) != 1 or rep["n_unmatched_kept_anonymous"] != 1:
            raise AssertionError(f"unmatched prediction must stay, anonymous: {labels} {rep}")


def test_yaw_obb_recovers_rotation():
    rng_pts = []
    import math
    theta = math.radians(30.0)
    for x in np.linspace(-1.0, 1.0, 21):
        for y in np.linspace(-0.25, 0.25, 7):
            rx = x * math.cos(theta) - y * math.sin(theta)
            ry = x * math.sin(theta) + y * math.cos(theta)
            rng_pts.append((rx, ry, 0.4))
    obb = _yaw_obb(np.array(rng_pts))
    ex, ey, ez = obb.extents
    if abs(ex - 1.0) > 0.02 or abs(ey - 0.25) > 0.02 or abs(ez) > 1e-9:
        raise AssertionError(f"OBB extents must recover the tight box: {obb.extents}")


TESTS = [
    test_bundle_roundtrip_deterministic,
    test_validation_failures,
    test_oracle_membership_majority_and_ties,
    test_exact_eval_perfect,
    test_exact_eval_split_and_merge,
    test_candidate_anonymous_and_boxes,
    test_candidate_fail_conditions,
    test_derived_bundle_injects_and_removes,
    test_derived_bundle_keeps_unmatched_anonymous,
    test_yaw_obb_recovers_rotation,
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

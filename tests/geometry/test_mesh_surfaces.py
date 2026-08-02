"""C3.0-S Stage 0: synthetic-only raw-mesh surface-estimator tests.

Run: python tests/geometry/test_mesh_surfaces.py
No Replica scene estimator is invoked here.
"""
from __future__ import annotations

import builtins
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

from extractors.base import StructuralSurface
from extractors.serde import dump_entity_artifacts, load_entity_artifacts
from geometry.mesh_surfaces import (
    RawTriangleMesh,
    estimate_from_ply,
    estimate_structural_surfaces,
    load_raw_triangle_mesh,
)


def _add_quad(vertices, faces, a, b, c, d):
    start = len(vertices)
    vertices.extend([a, b, c, d])
    faces.extend([(start, start + 1, start + 2),
                  (start, start + 2, start + 3)])


def _room_mesh(*, tilted_wall: bool = False, malformed: bool = False) -> RawTriangleMesh:
    v, f = [], []
    # 4 m x 4 m x 3 m room; disconnected planar patches are deliberate.
    _add_quad(v, f, (-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0))
    _add_quad(v, f, (-2, -2, 3), (-2, 2, 3), (2, 2, 3), (2, -2, 3))
    _add_quad(v, f, (-2, -2, 0), (-2, 2, 0), (-2, 2, 3), (-2, -2, 3))
    _add_quad(v, f, (2, -2, 0), (2, -2, 3), (2, 2, 3), (2, 2, 0))
    _add_quad(v, f, (-2, -2, 0), (-2, -2, 3), (2, -2, 3), (2, -2, 0))
    if tilted_wall:
        _add_quad(v, f, (-2, 2, 0), (2, 2.35, 0), (2, 2.35, 3), (-2, 2, 3))
    else:
        _add_quad(v, f, (-2, 2, 0), (2, 2, 0), (2, 2, 3), (-2, 2, 3))
    # Interior table top (< horizontal mesh-area gate) and a cabinet side
    # (> wall area gate, but rejected by the scene-boundary percentile gate).
    _add_quad(v, f, (-0.5, -0.5, 0.8), (0.5, -0.5, 0.8),
              (0.5, 0.5, 0.8), (-0.5, 0.5, 0.8))
    _add_quad(v, f, (0.5, -0.6, 0), (0.5, 0.6, 0),
              (0.5, 0.6, 2), (0.5, -0.6, 2))
    if malformed:
        # Add a third triangle on one floor edge -> non-manifold boundary;
        # this isolated patch is below the area threshold and may not create
        # a fallback rectangle or extra surface.
        a = len(v)
        v.extend([(0, 0, 1.2), (0.2, 0, 1.2),
                  (0.1, 0.2, 1.2), (0.1, 0.1, 1.2)])
        f.extend([(a, a + 1, a + 2), (a, a + 1, a + 3),
                  (a, a + 1, a + 2)])
    xyz = np.asarray(v, dtype=np.float64)
    return RawTriangleMesh(xyz=xyz, faces=np.asarray(f, dtype=np.int64),
                           rgb=np.zeros((len(xyz), 3), dtype=np.uint8))


def _surface_signature(surfaces: list[StructuralSurface]) -> str:
    rows = []
    for s in surfaces:
        rows.append({
            "uid": s.surface_uid, "type": s.surface_type,
            "plane": [s.plane.a, s.plane.b, s.plane.c, s.plane.d],
            "polygon": s.polygon, "confidence": s.confidence,
            "source": s.source,
        })
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


def _write_ascii(path: Path, mesh: RawTriangleMesh) -> None:
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(mesh.xyz)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        f"element face {len(mesh.faces)}",
        "property list uchar int vertex_indices", "end_header",
    ]
    lines += [f"{p[0]} {p[1]} {p[2]} 0 0 0" for p in mesh.xyz]
    lines += [f"3 {q[0]} {q[1]} {q[2]}" for q in mesh.faces]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_binary(path: Path, mesh: RawTriangleMesh) -> None:
    header = "\n".join([
        "ply", "format binary_little_endian 1.0",
        f"element vertex {len(mesh.xyz)}", "property float x",
        "property float y", "property float z", "property uchar red",
        "property uchar green", "property uchar blue",
        f"element face {len(mesh.faces)}",
        "property list uint8 int32 vertex_indices", "end_header", "",
    ]).encode("ascii")
    body = bytearray()
    for p in mesh.xyz:
        body += struct.pack("<fffBBB", float(p[0]), float(p[1]), float(p[2]), 0, 0, 0)
    for q in mesh.faces:
        body += struct.pack("<Biii", 3, int(q[0]), int(q[1]), int(q[2]))
    path.write_bytes(header + body)


def test_ascii_and_binary_parsers_match() -> None:
    mesh = _room_mesh()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        a, b = root / "a.ply", root / "b.ply"
        _write_ascii(a, mesh)
        _write_binary(b, mesh)
        ma, mb = load_raw_triangle_mesh(a), load_raw_triangle_mesh(b)
        if not np.allclose(ma.xyz, mb.xyz, atol=1e-6):
            raise AssertionError("ASCII/binary xyz mismatch")
        if not np.array_equal(ma.faces, mb.faces):
            raise AssertionError("ASCII/binary faces mismatch")


def test_synthetic_room_emits_only_structural_set() -> None:
    result = estimate_structural_surfaces(_room_mesh())
    counts = {t: sum(s.surface_type == t for s in result.surfaces)
              for t in ("floor", "wall", "ceiling")}
    if counts != {"floor": 1, "wall": 4, "ceiling": 1}:
        raise AssertionError(f"expected 1/4/1 structural set, got {counts}; "
                             f"diag={result.diagnostics}")
    if any(s.source != "mesh_region_fit" for s in result.surfaces):
        raise AssertionError("all surfaces require mesh_region_fit provenance")


def test_tilted_boundary_wall_retained_and_cabinet_rejected() -> None:
    result = estimate_structural_surfaces(_room_mesh(tilted_wall=True))
    walls = [s for s in result.surfaces if s.surface_type == "wall"]
    if len(walls) != 4:
        raise AssertionError(f"tilted boundary wall should survive: {len(walls)}")
    if any(abs(s.plane.d) < 1.0 for s in walls):
        raise AssertionError("interior cabinet plane leaked into wall set")


def test_deterministic_output_and_uid_order() -> None:
    mesh = _room_mesh()
    a = estimate_structural_surfaces(mesh)
    b = estimate_structural_surfaces(mesh)
    if _surface_signature(a.surfaces) != _surface_signature(b.surfaces):
        raise AssertionError("identical estimator runs differ")
    uids = [s.surface_uid for s in a.surfaces]
    if uids[0] != "mesh_floor_0" or uids[-1] != "mesh_ceiling_0":
        raise AssertionError(f"surface ordering drifted: {uids}")


def test_nonmanifold_patch_never_creates_fallback_surface() -> None:
    clean = estimate_structural_surfaces(_room_mesh())
    malformed = estimate_structural_surfaces(_room_mesh(malformed=True))
    if _surface_signature(clean.surfaces) != _surface_signature(malformed.surfaces):
        raise AssertionError("malformed patch must be rejected, not surfaced")


def test_estimate_from_ply_reads_only_supplied_mesh() -> None:
    mesh = _room_mesh()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "only_allowed.ply"
        _write_binary(path, mesh)
        opened: list[Path] = []
        real_open = builtins.open

        def recording_open(file, *args, **kwargs):
            opened.append(Path(file).resolve())
            return real_open(file, *args, **kwargs)

        builtins.open = recording_open
        try:
            estimate_from_ply(path, np.eye(3), np.zeros(3))
        finally:
            builtins.open = real_open
        if opened != [path.resolve()]:
            raise AssertionError(f"estimator read outside supplied mesh: {opened}")


TESTS = [
    test_ascii_and_binary_parsers_match,
    test_synthetic_room_emits_only_structural_set,
    test_tilted_boundary_wall_retained_and_cabinet_rejected,
    test_deterministic_output_and_uid_order,
    test_nonmanifold_patch_never_creates_fallback_surface,
    test_estimate_from_ply_reads_only_supplied_mesh,
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

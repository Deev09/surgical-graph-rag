"""Deterministic raw-mesh structural-surface estimator for C3.0-S.

Protocol: docs/c3_0_mesh_surfaces_protocol.md.  This module is deliberately
oracle-free: it accepts only a raw triangle PLY and a caller-supplied rigid
frame transform.  It never imports Replica metadata, entities, keys, graph
code, or evaluation code.

The estimator is conservative.  A malformed/non-manifold component is
rejected instead of being replaced with a scene-sized box or convex hull.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from common.types import Plane
from extractors.base import StructuralSurface


@dataclass(frozen=True)
class MeshSurfaceConfig:
    horizontal_angle_deg: float = 12.0
    vertical_angle_deg: float = 12.0
    region_normal_deg: float = 8.0
    region_plane_residual_m: float = 0.02
    component_rms_max_m: float = 0.015
    horizontal_mesh_area_min_m2: float = 1.5
    horizontal_projected_area_min_m2: float = 1.0
    floor_ceiling_separation_min_m: float = 1.8
    vertical_mesh_area_min_m2: float = 1.5
    boundary_percentile_low: float = 2.0
    boundary_percentile_high: float = 98.0
    boundary_distance_m: float = 0.15
    boundary_simplify_m: float = 0.01
    dedup_normal_deg: float = 3.0
    dedup_offset_m: float = 0.02
    dedup_polygon_iou: float = 0.80
    polygon_grid_m: float = 0.01
    degenerate_face_area_m2: float = 1e-8

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    def sha256(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


DEFAULT_CONFIG = MeshSurfaceConfig()


@dataclass(frozen=True)
class RawTriangleMesh:
    xyz: np.ndarray       # float64 [V,3]
    faces: np.ndarray     # int64 [F,3]
    rgb: np.ndarray       # uint8 [V,3]
    # C3.0-SR: quads are accepted and split on the fixed v0-v2 diagonal
    # ((v0,v1,v2)+(v0,v2,v3), the rule already frozen for Replica quads in
    # the C1-M2 triangulation); this records how many source faces were
    # quads. Zero for pure-triangle inputs.
    n_source_quads: int = 0


@dataclass(frozen=True)
class SurfaceEstimate:
    surfaces: list[StructuralSurface]
    diagnostics: dict[str, Any]


@dataclass
class _Component:
    family: int  # 1 horizontal, 2 vertical
    seed_face: int
    face_ids: np.ndarray
    mesh_area: float
    projected_area: float
    normal: np.ndarray
    d: float
    rms: float
    polygon: np.ndarray


_PLY_SCALAR_DTYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_ply_header(f) -> tuple[str, list[dict[str, Any]]]:
    first = f.readline()
    if first.strip() != b"ply":
        raise ValueError("not a PLY file")
    fmt: str | None = None
    elements: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    while True:
        line_b = f.readline()
        if not line_b:
            raise ValueError("PLY header missing end_header")
        try:
            line = line_b.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("non-ASCII bytes inside PLY header") from exc
        if line == "end_header":
            break
        if not line or line.startswith("comment") or line.startswith("obj_info"):
            continue
        parts = line.split()
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            current = {"name": parts[1], "count": int(parts[2]), "properties": []}
            elements.append(current)
        elif parts[0] == "property":
            if current is None:
                raise ValueError("property appears before element")
            if parts[1] == "list":
                current["properties"].append(
                    {"kind": "list", "count_type": parts[2],
                     "item_type": parts[3], "name": parts[4]})
            else:
                current["properties"].append(
                    {"kind": "scalar", "type": parts[1], "name": parts[2]})
    if fmt not in ("ascii", "binary_little_endian"):
        raise ValueError(f"unsupported PLY format {fmt!r}")
    return fmt, elements


def _elements_by_name(elements: list[dict[str, Any]]) -> tuple[dict, dict]:
    by_name = {e["name"]: e for e in elements}
    if "vertex" not in by_name or "face" not in by_name:
        raise ValueError("PLY requires vertex and face elements")
    vertex, face = by_name["vertex"], by_name["face"]
    vnames = [p["name"] for p in vertex["properties"]]
    if not all(n in vnames for n in ("x", "y", "z")):
        raise ValueError("PLY vertices require x/y/z")
    list_props = [p for p in face["properties"] if p["kind"] == "list"]
    if len(list_props) != 1:
        raise ValueError("PLY faces require exactly one list property")
    return vertex, face


def _load_ascii_mesh(f, vertex: dict, face: dict) -> RawTriangleMesh:
    n_v, n_f = int(vertex["count"]), int(face["count"])
    props = vertex["properties"]
    names = [p["name"] for p in props]
    xi, yi, zi = names.index("x"), names.index("y"), names.index("z")
    rgb_idx = [names.index(n) if n in names else None for n in ("red", "green", "blue")]
    xyz = np.empty((n_v, 3), dtype=np.float64)
    rgb = np.zeros((n_v, 3), dtype=np.uint8)
    for i in range(n_v):
        line = f.readline()
        if not line:
            raise ValueError(f"ASCII PLY ended at vertex {i}/{n_v}")
        vals = line.split()
        if len(vals) != len(props):
            raise ValueError(f"vertex {i} property count {len(vals)} != {len(props)}")
        xyz[i] = (float(vals[xi]), float(vals[yi]), float(vals[zi]))
        for c, idx in enumerate(rgb_idx):
            if idx is not None:
                rgb[i, c] = int(vals[idx])
    tris: list[tuple[int, int, int]] = []
    n_quads = 0
    for i in range(n_f):
        line = f.readline()
        if not line:
            raise ValueError(f"ASCII PLY ended at face {i}/{n_f}")
        vals = line.split()
        arity = int(vals[0]) if vals else -1
        if arity == 3 and len(vals) >= 4:
            tris.append((int(vals[1]), int(vals[2]), int(vals[3])))
        elif arity == 4 and len(vals) >= 5:
            v0, v1, v2, v3 = (int(vals[1]), int(vals[2]),
                              int(vals[3]), int(vals[4]))
            tris.append((v0, v1, v2))
            tris.append((v0, v2, v3))
            n_quads += 1
        else:
            raise ValueError("mesh_region_fit_v1 accepts face arity 3 or 4 "
                             f"only (face {i} has arity {arity})")
    return RawTriangleMesh(xyz=xyz, faces=np.asarray(tris, dtype=np.int64),
                           rgb=rgb, n_source_quads=n_quads)


def _load_binary_mesh(f, vertex: dict, face: dict) -> RawTriangleMesh:
    n_v, n_f = int(vertex["count"]), int(face["count"])
    fields: list[tuple[str, str]] = []
    for p in vertex["properties"]:
        if p["kind"] != "scalar" or p["type"] not in _PLY_SCALAR_DTYPES:
            raise ValueError(f"unsupported binary vertex property {p!r}")
        fields.append((p["name"], _PLY_SCALAR_DTYPES[p["type"]]))
    vdtype = np.dtype(fields, align=False)
    vbytes = f.read(n_v * vdtype.itemsize)
    if len(vbytes) != n_v * vdtype.itemsize:
        raise ValueError("binary PLY ended inside vertex data")
    verts = np.frombuffer(vbytes, dtype=vdtype, count=n_v)
    xyz = np.column_stack([verts["x"], verts["y"], verts["z"]]).astype(np.float64)
    rgb = np.zeros((n_v, 3), dtype=np.uint8)
    for c, name in enumerate(("red", "green", "blue")):
        if name in verts.dtype.names:
            rgb[:, c] = verts[name].astype(np.uint8)

    lp = [p for p in face["properties"] if p["kind"] == "list"][0]
    if lp["count_type"] not in ("uchar", "uint8"):
        raise ValueError("binary face list count must be uint8")
    if lp["item_type"] not in ("int", "int32", "uint", "uint32"):
        raise ValueError("binary face indices must be 32-bit integers")
    item_dtype = _PLY_SCALAR_DTYPES[lp["item_type"]]
    idt = np.dtype(item_dtype)
    fbytes = f.read()

    def uniform(k: int) -> np.ndarray | None:
        """Fast path when every record is arity k (fixed record size)."""
        rec_size = 1 + k * idt.itemsize
        if len(fbytes) != n_f * rec_size:
            return None
        counts = np.frombuffer(fbytes, dtype=np.uint8)[0::rec_size]
        if not np.all(counts == k):
            return None
        rec = np.frombuffer(fbytes, dtype=np.dtype(
            [("n", "u1")] + [(f"v{j}", idt.str) for j in range(k)],
            align=False), count=n_f)
        return np.column_stack([rec[f"v{j}"] for j in range(k)]).astype(np.int64)

    n_quads = 0
    tri = uniform(3)
    if tri is not None:
        faces = tri
    else:
        quad = uniform(4)
        if quad is not None:
            # fixed v0-v2 diagonal, split triangles adjacent in face order
            faces = np.empty((2 * n_f, 3), dtype=np.int64)
            faces[0::2] = quad[:, (0, 1, 2)]
            faces[1::2] = quad[:, (0, 2, 3)]
            n_quads = n_f
        else:
            # mixed-arity walk: arity 3 or 4 only, anything else hard-fails
            tris: list[tuple[int, int, int]] = []
            off = 0
            for i in range(n_f):
                if off >= len(fbytes):
                    raise ValueError("binary PLY ended inside face data")
                arity = fbytes[off]
                off += 1
                end = off + arity * idt.itemsize
                if arity not in (3, 4) or end > len(fbytes):
                    raise ValueError(
                        "mesh_region_fit_v1 accepts face arity 3 or 4 only "
                        f"(face {i} has arity {arity})")
                idx = np.frombuffer(fbytes[off:end], dtype=idt).astype(np.int64)
                off = end
                tris.append((idx[0], idx[1], idx[2]))
                if arity == 4:
                    tris.append((idx[0], idx[2], idx[3]))
                    n_quads += 1
            if off != len(fbytes):
                raise ValueError("binary PLY has trailing bytes after faces")
            faces = np.asarray(tris, dtype=np.int64)
    return RawTriangleMesh(xyz=xyz, faces=faces, rgb=rgb,
                           n_source_quads=n_quads)


def load_raw_triangle_mesh(path: Path) -> RawTriangleMesh:
    """Load the raw geometry formats used by Replica plus synthetic fixtures."""
    with open(path, "rb") as f:
        fmt, elements = _read_ply_header(f)
        vertex, face = _elements_by_name(elements)
        mesh = (_load_ascii_mesh(f, vertex, face) if fmt == "ascii"
                else _load_binary_mesh(f, vertex, face))
    if not np.isfinite(mesh.xyz).all():
        raise ValueError("raw mesh contains non-finite vertices")
    if mesh.faces.size and (mesh.faces.min() < 0 or mesh.faces.max() >= len(mesh.xyz)):
        raise ValueError("raw mesh face index out of range")
    return mesh


def transform_mesh(mesh: RawTriangleMesh, rotation: np.ndarray,
                   translation: np.ndarray) -> RawTriangleMesh:
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("rotation must be 3x3 and translation must be length 3")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("frame transform contains non-finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("frame rotation is not orthonormal")
    xyz = np.einsum("ij,nj->ni", rotation, mesh.xyz) + translation[None, :]
    return RawTriangleMesh(xyz=xyz, faces=mesh.faces.copy(), rgb=mesh.rgb.copy(),
                           n_source_quads=mesh.n_source_quads)


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=np.float64)
    axis = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, axis)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v


def _polygon_area_2d(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    return 0.5 * abs(float(np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) -
                           np.dot(poly[:, 1], np.roll(poly[:, 0], -1))))


def _project_polygon(poly: np.ndarray, normal: np.ndarray,
                     origin: np.ndarray) -> np.ndarray:
    u, v = _plane_basis(normal)
    rel = poly - origin[None, :]
    return np.column_stack([rel @ u, rel @ v])


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized winding-agnostic ray casting; boundary counts as inside."""
    if len(polygon) < 3:
        return np.zeros(len(points), dtype=bool)
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    boundary = np.zeros(len(points), dtype=bool)
    eps = 1e-12
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Boundary test before ray crossing.
        dx, dy = xj - xi, yj - yi
        cross = (x - xi) * dy - (y - yi) * dx
        dot = (x - xi) * dx + (y - yi) * dy
        on = (np.abs(cross) <= eps * max(1.0, abs(dx) + abs(dy))) & \
             (dot >= -eps) & (dot <= dx * dx + dy * dy + eps)
        boundary |= on
        hit = ((yi > y) != (yj > y)) & \
              (x < (xj - xi) * (y - yi) / (yj - yi + eps) + xi)
        inside ^= hit
        j = i
    return inside | boundary


def _polygon_iou_grid(poly_a: np.ndarray, normal: np.ndarray,
                      poly_b: np.ndarray, step: float) -> float:
    """Deterministic 1 cm cell-center IoU used only for deduplication."""
    origin = poly_a[0]
    a2 = _project_polygon(poly_a, normal, origin)
    b2 = _project_polygon(poly_b, normal, origin)
    lo = np.minimum(a2.min(axis=0), b2.min(axis=0))
    hi = np.maximum(a2.max(axis=0), b2.max(axis=0))
    nx = max(1, int(math.ceil((hi[0] - lo[0]) / step)))
    ny = max(1, int(math.ceil((hi[1] - lo[1]) / step)))
    if nx * ny > 5_000_000:
        raise ValueError("polygon IoU grid exceeds deterministic safety cap")
    xs = lo[0] + (np.arange(nx) + 0.5) * step
    ys = lo[1] + (np.arange(ny) + 0.5) * step
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    ina = _points_in_polygon(pts, a2)
    inb = _points_in_polygon(pts, b2)
    union = int(np.count_nonzero(ina | inb))
    return (int(np.count_nonzero(ina & inb)) / union) if union else 0.0


def _face_geometry(mesh: RawTriangleMesh, cfg: MeshSurfaceConfig):
    tri = mesh.xyz[mesh.faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    area = 0.5 * norm
    normals = np.zeros_like(cross)
    good = area >= cfg.degenerate_face_area_m2
    normals[good] = cross[good] / norm[good, None]
    family = np.zeros(len(mesh.faces), dtype=np.int8)
    az = np.abs(normals[:, 2])
    family[good & (az >= math.cos(math.radians(cfg.horizontal_angle_deg)))] = 1
    family[good & (az <= math.sin(math.radians(cfg.vertical_angle_deg)))] = 2
    return tri, normals, area, family, good


def _face_adjacency(faces: np.ndarray, family: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    n = len(faces)
    edge = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edge.sort(axis=1)
    owner = np.concatenate([np.arange(n), np.arange(n), np.arange(n)])
    order = np.lexsort((edge[:, 1], edge[:, 0]))
    edge, owner = edge[order], owner[order]
    change = np.ones(len(edge), dtype=bool)
    change[1:] = np.any(edge[1:] != edge[:-1], axis=1)
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(edge)]
    counts = ends - starts
    manifold = starts[counts == 2]
    pairs = np.column_stack([owner[manifold], owner[manifold + 1]])
    keep = (family[pairs[:, 0]] != 0) & (family[pairs[:, 0]] == family[pairs[:, 1]])
    pairs = pairs[keep]
    if len(pairs):
        directed = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
        dorder = np.lexsort((directed[:, 1], directed[:, 0]))
        directed = directed[dorder]
        degree = np.bincount(directed[:, 0], minlength=n)
        offsets = np.zeros(n + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(degree)
        neighbors = directed[:, 1].astype(np.int64, copy=False)
    else:
        offsets = np.zeros(n + 1, dtype=np.int64)
        neighbors = np.empty(0, dtype=np.int64)
    return offsets, neighbors, int(np.count_nonzero(counts > 2))


def _region_components(mesh: RawTriangleMesh, tri: np.ndarray, normals: np.ndarray,
                       area: np.ndarray, family: np.ndarray,
                       offsets: np.ndarray, neighbors: np.ndarray,
                       cfg: MeshSurfaceConfig) -> list[tuple[int, int, np.ndarray]]:
    visited = np.zeros(len(mesh.faces), dtype=bool)
    cos_gate = math.cos(math.radians(cfg.region_normal_deg))
    out: list[tuple[int, int, np.ndarray]] = []
    for seed in np.flatnonzero(family):
        seed = int(seed)
        if visited[seed]:
            continue
        fam = int(family[seed])
        seed_n = normals[seed]
        seed_d = -float(np.dot(seed_n, tri[seed, 0]))
        visited[seed] = True
        queue = [seed]
        comp: list[int] = []
        qpos = 0
        while qpos < len(queue):
            cur = queue[qpos]
            qpos += 1
            comp.append(cur)
            for nb_v in neighbors[offsets[cur]:offsets[cur + 1]]:
                nb = int(nb_v)
                if visited[nb] or int(family[nb]) != fam:
                    continue
                if abs(float(np.dot(seed_n, normals[nb]))) < cos_gate:
                    continue
                residual = np.max(np.abs(tri[nb] @ seed_n + seed_d))
                if residual > cfg.region_plane_residual_m:
                    continue
                visited[nb] = True
                queue.append(nb)
        ids = np.asarray(comp, dtype=np.int64)
        if float(area[ids].sum()) >= min(cfg.horizontal_mesh_area_min_m2,
                                         cfg.vertical_mesh_area_min_m2):
            out.append((fam, seed, ids))
    return out


def _boundary_loop(mesh: RawTriangleMesh, face_ids: np.ndarray,
                   normal: np.ndarray, cfg: MeshSurfaceConfig) -> np.ndarray | None:
    f = mesh.faces[face_ids]
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges.sort(axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    change = np.ones(len(edges), dtype=bool)
    change[1:] = np.any(edges[1:] != edges[:-1], axis=1)
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(edges)]
    boundary = edges[starts[ends - starts == 1]]
    if len(boundary) < 3:
        return None
    adj: dict[int, list[int]] = {}
    for a_v, b_v in boundary:
        a, b = int(a_v), int(b_v)
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    if any(len(v) != 2 for v in adj.values()):
        return None
    for v in adj.values():
        v.sort()
    unused = {tuple(map(int, e)) for e in boundary}
    loops: list[list[int]] = []
    while unused:
        start_edge = min(unused)
        start, nxt = start_edge
        loop = [start]
        prev, cur = start, nxt
        unused.discard((min(prev, cur), max(prev, cur)))
        while cur != start:
            loop.append(cur)
            choices = [x for x in adj[cur] if x != prev]
            if len(choices) != 1:
                return None
            nxt = choices[0]
            edge_key = (min(cur, nxt), max(cur, nxt))
            if edge_key not in unused and nxt != start:
                return None
            unused.discard(edge_key)
            prev, cur = cur, nxt
            if len(loop) > len(adj) + 1:
                return None
        loops.append(loop)
    origin = mesh.xyz[loops[0][0]]
    scored = []
    for loop in loops:
        pts = mesh.xyz[np.asarray(loop, dtype=np.int64)]
        area2 = _polygon_area_2d(_project_polygon(pts, normal, origin))
        scored.append((area2, -min(loop), pts))
    poly = max(scored, key=lambda x: (x[0], x[1]))[2]
    kept = [poly[0]]
    for p in poly[1:]:
        if float(np.linalg.norm(p - kept[-1])) >= cfg.boundary_simplify_m:
            kept.append(p)
    if len(kept) >= 2 and float(np.linalg.norm(kept[-1] - kept[0])) < cfg.boundary_simplify_m:
        kept.pop()
    if len(kept) < 3:
        return None
    return np.asarray(kept, dtype=np.float64)


def _fit_component(mesh: RawTriangleMesh, tri: np.ndarray, normals: np.ndarray,
                   area: np.ndarray, fam: int, seed: int, ids: np.ndarray,
                   cfg: MeshSurfaceConfig) -> _Component | None:
    pts = tri[ids].reshape(-1, 3)
    weights = np.repeat(area[ids] / 3.0, 3)
    wsum = float(weights.sum())
    if wsum <= 0.0:
        return None
    mean = np.sum(pts * weights[:, None], axis=0) / wsum
    centered = pts - mean
    cov = (centered * weights[:, None]).T @ centered / wsum
    eigvals, eigvecs = np.linalg.eigh(cov)
    n = eigvecs[:, int(np.argmin(eigvals))]
    n /= np.linalg.norm(n)
    if fam == 1:
        if n[2] < 0:
            n = -n
    else:
        scene_center = (mesh.xyz.min(axis=0) + mesh.xyz.max(axis=0)) / 2.0
        if float(np.dot(n, scene_center - mean)) < 0:
            n = -n
    d = -float(np.dot(n, mean))
    residual = pts @ n + d
    rms = math.sqrt(float(np.sum(weights * residual * residual) / wsum))
    if rms > cfg.component_rms_max_m:
        return None
    projected_area = float(np.sum(area[ids] * np.abs(normals[ids] @ n)))
    poly = _boundary_loop(mesh, ids, n, cfg)
    if poly is None:
        return None
    # Orthogonally place boundary points on the fitted plane.
    poly = poly - (poly @ n + d)[:, None] * n[None, :]
    return _Component(
        family=fam, seed_face=seed, face_ids=ids,
        mesh_area=float(area[ids].sum()), projected_area=projected_area,
        normal=n, d=d, rms=rms, polygon=poly,
    )


def _deduplicate(components: list[_Component], cfg: MeshSurfaceConfig) -> list[_Component]:
    kept: list[_Component] = []
    cos_gate = math.cos(math.radians(cfg.dedup_normal_deg))
    order = sorted(components, key=lambda c: (-c.mesh_area, c.seed_face))
    for comp in order:
        duplicate = False
        for prior in kept:
            dot = float(np.dot(comp.normal, prior.normal))
            if abs(dot) < cos_gate:
                continue
            d = comp.d if dot >= 0 else -comp.d
            if abs(d - prior.d) > cfg.dedup_offset_m:
                continue
            if _polygon_iou_grid(prior.polygon, prior.normal, comp.polygon,
                                 cfg.polygon_grid_m) >= cfg.dedup_polygon_iou:
                duplicate = True
                break
        if not duplicate:
            kept.append(comp)
    return kept


def _wall_sort_key(comp: _Component) -> tuple[float, float, str]:
    az = math.atan2(float(comp.normal[1]), float(comp.normal[0]))
    digest = hashlib.sha256(np.ascontiguousarray(comp.polygon).tobytes()).hexdigest()
    return (round(az, 12), round(comp.d, 12), digest)


def _as_surface(comp: _Component, uid: str, surface_type: str,
                cfg: MeshSurfaceConfig) -> StructuralSurface:
    n = comp.normal.copy()
    d = float(comp.d)
    if surface_type == "ceiling" and n[2] > 0:
        n, d = -n, -d
    confidence = max(0.0, min(1.0, 1.0 - comp.rms / cfg.component_rms_max_m))
    return StructuralSurface(
        surface_uid=uid,
        surface_type=surface_type,  # type: ignore[arg-type]
        plane=Plane(a=float(n[0]), b=float(n[1]), c=float(n[2]), d=d),
        polygon=[tuple(map(float, p)) for p in comp.polygon],
        confidence=round(confidence, 6),
        source="mesh_region_fit",  # type: ignore[arg-type]
    )


def estimate_structural_surfaces(mesh: RawTriangleMesh,
                                 cfg: MeshSurfaceConfig = DEFAULT_CONFIG) -> SurfaceEstimate:
    """Estimate floor/walls/ceiling from an already canonicalized raw mesh."""
    tri, normals, area, family, good = _face_geometry(mesh, cfg)
    offsets, neighbors, n_nonmanifold = _face_adjacency(mesh.faces, family)
    raw_components = _region_components(
        mesh, tri, normals, area, family, offsets, neighbors, cfg)
    fit: list[_Component] = []
    n_boundary_rejected = n_rms_rejected = 0
    for fam, seed, ids in raw_components:
        comp = _fit_component(mesh, tri, normals, area, fam, seed, ids, cfg)
        if comp is None:
            # Boundary and RMS are intentionally combined in the public count;
            # neither authorizes a fallback surface.
            n_boundary_rejected += 1
            continue
        fit.append(comp)

    horizontal = [c for c in fit if c.family == 1 and
                  c.mesh_area >= cfg.horizontal_mesh_area_min_m2 and
                  c.projected_area >= cfg.horizontal_projected_area_min_m2]
    vertical = [c for c in fit if c.family == 2 and
                c.mesh_area >= cfg.vertical_mesh_area_min_m2]

    invalid_reason: str | None = None
    floor: _Component | None = None
    ceiling: _Component | None = None
    if len(horizontal) >= 2:
        horizontal.sort(key=lambda c: ((-c.d / c.normal[2]), c.seed_face))
        floor, ceiling = horizontal[0], horizontal[-1]
        z_floor = -floor.d / floor.normal[2]
        z_ceil = -ceiling.d / ceiling.normal[2]
        if z_ceil - z_floor < cfg.floor_ceiling_separation_min_m:
            floor = ceiling = None
            invalid_reason = "floor_ceiling_separation_below_minimum"
    else:
        invalid_reason = "fewer_than_two_qualifying_horizontal_components"

    walls: list[_Component] = []
    for comp in vertical:
        projection = mesh.xyz @ comp.normal
        low, high = np.percentile(
            projection, [cfg.boundary_percentile_low, cfg.boundary_percentile_high])
        offset = -comp.d
        if min(abs(offset - float(low)), abs(offset - float(high))) <= cfg.boundary_distance_m:
            walls.append(comp)
    walls = _deduplicate(walls, cfg)
    walls.sort(key=_wall_sort_key)

    surfaces: list[StructuralSurface] = []
    if floor is not None:
        surfaces.append(_as_surface(floor, "mesh_floor_0", "floor", cfg))
    for i, wall in enumerate(walls):
        surfaces.append(_as_surface(wall, f"mesh_wall_{i:02d}", "wall", cfg))
    if ceiling is not None:
        surfaces.append(_as_surface(ceiling, "mesh_ceiling_0", "ceiling", cfg))

    diagnostics = {
        "estimator": "mesh_region_fit_v1",
        "config": cfg.to_dict(),
        "config_sha256": cfg.sha256(),
        "n_vertices": int(len(mesh.xyz)),
        "n_faces": int(len(mesh.faces)),
        "n_source_quads": int(mesh.n_source_quads),
        "n_degenerate_faces": int(np.count_nonzero(~good)),
        "n_horizontal_candidate_faces": int(np.count_nonzero(family == 1)),
        "n_vertical_candidate_faces": int(np.count_nonzero(family == 2)),
        "n_nonmanifold_mesh_edges": n_nonmanifold,
        "n_region_components_over_area_floor": int(len(raw_components)),
        "n_fit_components": int(len(fit)),
        "n_rejected_fit_or_boundary": int(n_boundary_rejected + n_rms_rejected),
        "n_qualifying_horizontal": int(len(horizontal)),
        "n_qualifying_vertical_before_boundary": int(len(vertical)),
        "n_walls": int(len(walls)),
        "invalid_reason": invalid_reason,
    }
    return SurfaceEstimate(surfaces=surfaces, diagnostics=diagnostics)


def estimate_from_ply(mesh_path: Path, rotation: Iterable[Iterable[float]],
                      translation: Iterable[float],
                      cfg: MeshSurfaceConfig = DEFAULT_CONFIG) -> SurfaceEstimate:
    raw = load_raw_triangle_mesh(mesh_path)
    canonical = transform_mesh(raw, np.asarray(rotation, dtype=np.float64),
                               np.asarray(translation, dtype=np.float64))
    return estimate_structural_surfaces(canonical, cfg)

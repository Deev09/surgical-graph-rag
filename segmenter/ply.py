"""Strict binary-PLY vertex reader for raw Replica mesh.ply.

Property-driven (builds the numpy dtype from the header's vertex property
list), so it tolerates prop-order drift but fails loudly on unknown types or
ascii PLY. Only vertices are read — the raw mesh's faces carry no object
attribution, and the C1 pipeline needs positions only (dense per-vertex
instance assignment references vertices by index).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_PLY_TYPES = {
    "float": "<f4", "float32": "<f4", "double": "<f8",
    "uchar": "u1", "uint8": "u1", "char": "i1",
    "short": "<i2", "ushort": "<u2",
    "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
}


def parse_vertices(ply_path: Path) -> np.ndarray:
    """Return xyz [n_vertices, 3] float64 from a binary_little_endian PLY."""
    raw = ply_path.read_bytes()
    end = raw.find(b"end_header\n")
    if end < 0:
        raise ValueError(f"no end_header in {ply_path}")
    header = raw[:end].decode("ascii", "replace")
    body = raw[end + len(b"end_header\n"):]

    if "format binary_little_endian" not in header:
        raise ValueError(f"expected binary_little_endian PLY: {ply_path}")

    n_vert = None
    props: list[tuple[str, str]] = []
    in_vertex = False
    for line in header.splitlines():
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            n_vert = int(parts[2])
            in_vertex = True
        elif parts[:1] == ["element"]:
            in_vertex = False
        elif in_vertex and parts[:1] == ["property"]:
            if parts[1] == "list":
                raise ValueError(f"list property on vertices unsupported: {line}")
            if parts[1] not in _PLY_TYPES:
                raise ValueError(f"unknown vertex property type: {line}")
            props.append((parts[2], _PLY_TYPES[parts[1]]))
    if n_vert is None:
        raise ValueError(f"no vertex element in {ply_path}")
    names = [n for n, _ in props]
    if not {"x", "y", "z"} <= set(names):
        raise ValueError(f"vertex properties missing x/y/z: {names}")

    verts = np.frombuffer(body, dtype=np.dtype(props), count=n_vert)
    return np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)

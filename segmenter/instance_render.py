"""Point-splat renders of a single instance's colored vertices (C2.0).

Protocol: docs/c2_matched_labels_protocol.md — three orthographic views
(top XY, front XZ, side YZ), 224x224, white background, depth-sorted 2 px
splats, 8 % margin, instance-only crop. Deterministic; numpy + PIL only.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

SIZE = 224
MARGIN = 0.08
SPLAT = 2          # splat half-size in px (2 -> 5x5 with the center)

# (horizontal axis, vertical axis, depth axis) per view; depth ascending is
# drawn first so nearer vertices overwrite farther ones.
VIEWS = {
    "top": (0, 1, 2),     # x right, y up, +z toward viewer
    "front": (0, 2, 1),   # x right, z up, looking along -y
    "side": (1, 2, 0),    # y right, z up, looking along -x
}


def render_views(xyz: np.ndarray, rgb: np.ndarray) -> dict[str, Image.Image]:
    """xyz [n,3] float64 (gravity-aligned), rgb [n,3] uint8 -> 3 PIL images."""
    if len(xyz) == 0:
        raise ValueError("cannot render an empty instance")
    out: dict[str, Image.Image] = {}
    for name, (ha, va, da) in VIEWS.items():
        h, v, d = xyz[:, ha], xyz[:, va], xyz[:, da]
        span = max(float(h.max() - h.min()), float(v.max() - v.min()), 1e-6)
        scale = SIZE * (1 - 2 * MARGIN) / span
        cx = (h.max() + h.min()) / 2.0
        cy = (v.max() + v.min()) / 2.0
        px = np.clip((h - cx) * scale + SIZE / 2, 0, SIZE - 1).astype(np.int32)
        py = np.clip(SIZE / 2 - (v - cy) * scale, 0, SIZE - 1).astype(np.int32)
        order = np.argsort(d, kind="stable")      # far first, near overwrites
        img = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
        for dy in range(-SPLAT, SPLAT + 1):
            for dx in range(-SPLAT, SPLAT + 1):
                yy = np.clip(py[order] + dy, 0, SIZE - 1)
                xx = np.clip(px[order] + dx, 0, SIZE - 1)
                img[yy, xx] = rgb[order]
        out[name] = Image.fromarray(img)
    return out

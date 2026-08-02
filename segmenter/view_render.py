"""C1-P1 deterministic view contract: RGB + vertex-id point-splat renders.

Protocol: docs/c1_p1_multiview_proposals_protocol.md ("Deterministic view
contract"). Pure numpy, CPU, deterministic. RGB and id buffers share
exactly the same projection, depth test, and draw order — the id buffer
records the frontmost source vertex at every covered pixel (-1 empty).

Fixed contract values (frozen by the protocol; not parameters):
  five origins at z0+1.60 m (center, ±0.18*span on each XY axis), eight
  yaw headings 0..315°, pitch -10°, roll 0, vertical FOV 90°, 1024x1024,
  near 0.05 m, far = bbox diagonal + 1 m, 3 px splat diameter.
Background pixels are black (0,0,0) — a declared rendering choice shared
by RGB and manifest hashes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SIZE = 1024
VFOV_DEG = 90.0
NEAR_M = 0.05
PITCH_DEG = -10.0
EYE_HEIGHT_M = 1.60
ORIGIN_FRAC = 0.18
YAWS_DEG = tuple(range(0, 360, 45))
SPLAT_OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))


@dataclass(frozen=True)
class Camera:
    origin: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float = PITCH_DEG


def camera_set(xyz: np.ndarray) -> list[Camera]:
    """The 40 frozen cameras from the canonical-frame bounding box."""
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
    sx, sy = hi[0] - lo[0], hi[1] - lo[1]
    z = lo[2] + EYE_HEIGHT_M
    origins = [(cx, cy, z),
               (cx - ORIGIN_FRAC * sx, cy, z), (cx + ORIGIN_FRAC * sx, cy, z),
               (cx, cy - ORIGIN_FRAC * sy, z), (cx, cy + ORIGIN_FRAC * sy, z)]
    return [Camera(origin=o, yaw_deg=float(yaw))
            for o in origins for yaw in YAWS_DEG]


def _basis(cam: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw = math.radians(cam.yaw_deg)
    pitch = math.radians(cam.pitch_deg)
    fwd = np.array([math.cos(yaw) * math.cos(pitch),
                    math.sin(yaw) * math.cos(pitch),
                    math.sin(pitch)])
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    up = np.cross(right, fwd)
    return fwd, right, up


def render_view(xyz: np.ndarray, rgb: np.ndarray, cam: Camera,
                far_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rgb_image uint8 [S,S,3], id_buffer int32 [S,S], -1 empty)."""
    fwd, right, up = _basis(cam)
    d = xyz - np.asarray(cam.origin)[None, :]
    zc = d @ fwd
    vis = (zc > NEAR_M) & (zc < far_m)
    idx = np.flatnonzero(vis)
    focal = (SIZE / 2.0) / math.tan(math.radians(VFOV_DEG) / 2.0)
    u = SIZE / 2.0 + focal * (d[idx] @ right) / zc[idx]
    v = SIZE / 2.0 - focal * (d[idx] @ up) / zc[idx]
    px = np.floor(u).astype(np.int64)
    py = np.floor(v).astype(np.int64)
    on = (px >= 0) & (px < SIZE) & (py >= 0) & (py < SIZE)
    idx, px, py = idx[on], px[on], py[on]
    # Expand splats FIRST, then depth-sort all (vertex, offset) samples
    # jointly and paint far->near in one pass — per-offset passes would
    # let a far vertex overwrite a nearer one's earlier-offset pixel.
    k = len(SPLAT_OFFSETS)
    off = np.asarray(SPLAT_OFFSETS, dtype=np.int64)      # [k, (dy,dx)]
    yy = np.clip(py[:, None] + off[None, :, 0], 0, SIZE - 1).ravel()
    xx = np.clip(px[:, None] + off[None, :, 1], 0, SIZE - 1).ravel()
    samp_idx = np.repeat(idx, k)
    order = np.argsort(-zc[samp_idx], kind="stable")
    yy, xx, samp_idx = yy[order], xx[order], samp_idx[order]
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    ids = np.full((SIZE, SIZE), -1, dtype=np.int32)
    img[yy, xx] = rgb[samp_idx]
    ids[yy, xx] = samp_idx.astype(np.int32)
    return img, ids


def render_all(xyz: np.ndarray, rgb: np.ndarray):
    """Yield (view_index, camera, rgb_image, id_buffer) for all 40 views."""
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    far = float(np.linalg.norm(hi - lo)) + 1.0
    for i, cam in enumerate(camera_set(xyz)):
        img, ids = render_view(xyz, rgb, cam, far)
        yield i, cam, img, ids

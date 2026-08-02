"""C1-P1 Stage 0: implementation-validity tests (zero scene inference).

Run: python tests/segmenter/test_c1p1_stage0.py

Per docs/c1_p1_multiview_proposals_protocol.md Stage 0: RGB/id agreement
under depth+occlusion, exact lifting, co-membership separation and
retention, byte-determinism, generator isolation, and bank conformance.
Synthetic geometry only.
"""
from __future__ import annotations

import builtins
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmenter.proposal_fusion import (
    build_bank, edge_confidence, lift_mask, mesh_edges,
)
from segmenter.view_render import SIZE, Camera, render_view


def test_rgb_and_id_agree_under_occlusion():
    # two points on the same ray: near red at 1m, far blue at 3m
    cam = Camera(origin=(0.0, 0.0, 0.0), yaw_deg=0.0, pitch_deg=0.0)
    xyz = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    rgb = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
    img, ids = render_view(xyz, rgb, cam, far_m=10.0)
    c = SIZE // 2
    if ids[c, c] != 0:
        raise AssertionError(f"nearer vertex must win the id buffer: {ids[c, c]}")
    if tuple(img[c, c]) != (255, 0, 0):
        raise AssertionError(f"nearer vertex must win the RGB buffer: {img[c, c]}")
    # every covered pixel's RGB must be the recorded vertex's color
    cov = ids >= 0
    if not np.array_equal(img[cov], rgb[ids[cov]]):
        raise AssertionError("RGB and id buffers disagree somewhere")


def test_lift_returns_exact_visible_ids():
    rng = np.random.default_rng(0)
    xyz = np.column_stack([np.full(60, 2.0),
                           rng.uniform(-0.5, 0.5, 60),
                           rng.uniform(-0.5, 0.5, 60)])
    rgb = np.full((60, 3), 128, dtype=np.uint8)
    cam = Camera(origin=(0.0, 0.0, 0.0), yaw_deg=0.0, pitch_deg=0.0)
    _, ids = render_view(xyz, rgb, cam, far_m=10.0)
    mask = np.ones((SIZE, SIZE), dtype=bool)          # full-image mask
    lifted = lift_mask(mask, ids)
    visible = np.unique(ids[ids >= 0])
    if not np.array_equal(lifted, visible):
        raise AssertionError("full-image lift must equal the visible set")
    empty = lift_mask(np.zeros((SIZE, SIZE), dtype=bool), ids)
    if len(empty):
        raise AssertionError("empty mask must lift to nothing")


def _two_object_views():
    """Two 4-vertex objects joined by one bridging edge; 3 views where 2D
    masks always separate them."""
    faces = np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7],
                      [3, 4, 5]])                     # last face bridges
    n = 8
    a, b = np.arange(0, 4), np.arange(4, 8)
    views = [{"visible": np.arange(n), "masks": [a, b]} for _ in range(3)]
    return faces, n, views


def test_comembership_separates_and_retains():
    faces, n, views = _two_object_views()
    edges = mesh_edges(faces)
    co_vis, conf = edge_confidence(edges, n, views)
    for (u, v), c in zip(edges, conf):
        same = (u < 4) == (v < 4)
        if same and c != 1.0:
            raise AssertionError(f"intra-object edge ({u},{v}) conf {c}")
        if not same and c != 0.0:
            raise AssertionError(f"bridge edge ({u},{v}) conf {c}")
    # min-vertices floor would drop these tiny objects; verify separation
    # at component level with the floor relaxed via a bigger synthetic
    import segmenter.proposal_fusion as pf
    old = (pf.MIN_COMPONENT_VERTICES, pf.MAX_COMPONENT_FRAC)
    pf.MIN_COMPONENT_VERTICES, pf.MAX_COMPONENT_FRAC = 2, 1.0
    try:
        bank = build_bank(edges, co_vis, conf, n)
    finally:
        pf.MIN_COMPONENT_VERTICES, pf.MAX_COMPONENT_FRAC = old
    sets = {tuple(p["vertices"].tolist()) for p in bank}
    if (0, 1, 2, 3) not in sets or (4, 5, 6, 7) not in sets:
        raise AssertionError(f"objects must be retained separately: {sets}")
    for p in bank:
        s = set(p["vertices"].tolist())
        if s & {0, 1, 2, 3} and s & {4, 5, 6, 7}:
            raise AssertionError(f"objects merged across the bridge: {s}")


def test_render_and_fusion_deterministic():
    rng = np.random.default_rng(1)
    xyz = rng.uniform(-1, 1, (500, 3)) + np.array([3.0, 0, 0])
    rgb = rng.integers(0, 255, (500, 3)).astype(np.uint8)
    cam = Camera(origin=(0.0, 0.0, 0.0), yaw_deg=0.0)
    i1, d1 = render_view(xyz, rgb, cam, far_m=20.0)
    i2, d2 = render_view(xyz, rgb, cam, far_m=20.0)
    if not (np.array_equal(i1, i2) and np.array_equal(d1, d2)):
        raise AssertionError("render must be byte-identical")
    faces, n, views = _two_object_views()
    edges = mesh_edges(faces)
    r1 = edge_confidence(edges, n, views)
    r2 = edge_confidence(edges, n, views)
    if not (np.array_equal(r1[0], r2[0]) and np.array_equal(r1[1], r2[1])):
        raise AssertionError("fusion must be deterministic")


def test_generator_reads_only_supplied_mesh():
    from tests.segmenter.test_c1_pipeline import PERFECT, _scene
    from tools.c1p1_render import render_scene
    with tempfile.TemporaryDirectory() as td:
        room, _ = _scene(Path(td), PERFECT)
        out = Path(td) / "views"
        frame = {"world_from_raw_rotation": np.eye(3).tolist(),
                 "world_from_raw_translation": [0.0, 0.0, 0.0]}
        opened: list[str] = []
        real_open = builtins.open

        def recording_open(file, *a, **k):
            opened.append(str(file))
            return real_open(file, *a, **k)

        builtins.open = recording_open
        try:
            render_scene(room / "mesh.ply", frame, out, "testsha",
                         "synthetic_scene")
        finally:
            builtins.open = real_open
        bad = [p for p in opened
               if "info_semantic" in p or "mesh_semantic" in p
               or "phase8" in p or "_qa.json" in p]
        if bad:
            raise AssertionError(f"generator read forbidden inputs: {bad}")
        m = json.loads((out / "manifest.json").read_text())
        if m["contract"]["n_views"] != 40:
            raise AssertionError(f"view contract broken: {m['contract']}")


def test_bank_conformance_and_frozen_bundle_untouched():
    faces, n, views = _two_object_views()
    edges = mesh_edges(faces)
    co_vis, conf = edge_confidence(edges, n, views)
    import segmenter.proposal_fusion as pf
    old = (pf.MIN_COMPONENT_VERTICES, pf.MAX_COMPONENT_FRAC)
    pf.MIN_COMPONENT_VERTICES, pf.MAX_COMPONENT_FRAC = 2, 1.0
    try:
        bank = build_bank(edges, co_vis, conf, n)
    finally:
        pf.MIN_COMPONENT_VERTICES, pf.MAX_COMPONENT_FRAC = old
    for p in bank:
        v = p["vertices"]
        if v.dtype != np.int64 or v.min() < 0 or v.max() >= n:
            raise AssertionError(f"bank vertex indexing invalid: {v}")
        if not np.array_equal(v, np.sort(v)):
            raise AssertionError("bank vertex sets must be sorted")
        if p["cut"] not in (0.25, 0.50, 0.75):
            raise AssertionError(f"unknown cut {p['cut']}")
    # the P1 bank is a SEPARATE artifact: nothing under bundles_ms02 is
    # written by any c1p1 tool (they only ever open it read-only)
    import tools.c1p1_fuse as cf
    import tools.c1p1_render as cr
    for mod in (cr, cf):
        src = Path(mod.__file__).read_text()
        if "bundles_ms02" in src:
            raise AssertionError(f"{mod.__name__} must not touch the frozen "
                                 "Mask3D bundle")


TESTS = [
    test_rgb_and_id_agree_under_occlusion,
    test_lift_returns_exact_visible_ids,
    test_comembership_separates_and_retains,
    test_render_and_fusion_deterministic,
    test_generator_reads_only_supplied_mesh,
    test_bank_conformance_and_frozen_bundle_untouched,
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

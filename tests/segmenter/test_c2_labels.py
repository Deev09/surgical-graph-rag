"""Tests for the C2.0 label path (renderer, color parser, label_override).

Run: python tests/segmenter/test_c2_labels.py

CLIP itself is an optional dependency — labeler tests self-skip when
torch/open_clip are not importable (the frozen pipeline never needs them).
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmenter.instance_render import SIZE, render_views
from segmenter.ply import parse_vertices, parse_vertices_with_colors
from tests.segmenter.test_c1_pipeline import PERFECT, _scene


def test_render_views_deterministic_and_bounded():
    rng_xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                        [1.0, 2.0, 0.5], [0.0, 2.0, 1.0]])
    rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [10, 10, 10]],
                   dtype=np.uint8)
    v1 = render_views(rng_xyz, rgb)
    v2 = render_views(rng_xyz, rgb)
    if set(v1) != {"top", "front", "side"}:
        raise AssertionError(f"expected 3 views: {set(v1)}")
    for name in v1:
        a1, a2 = np.asarray(v1[name]), np.asarray(v2[name])
        if a1.shape != (SIZE, SIZE, 3) or not np.array_equal(a1, a2):
            raise AssertionError(f"view {name} not deterministic/224px")
        if (a1 == 255).all():
            raise AssertionError(f"view {name} rendered nothing")


def test_parse_vertices_with_colors_matches_positions():
    with tempfile.TemporaryDirectory() as td:
        room, _ = _scene(Path(td), PERFECT)
        xyz_only = parse_vertices(room / "mesh.ply")
        xyz, rgb = parse_vertices_with_colors(room / "mesh.ply")
        if not np.array_equal(xyz, xyz_only):
            raise AssertionError("positions must match plain parser")
        if rgb.shape != (len(xyz), 3) or rgb.dtype != np.uint8:
            raise AssertionError(f"bad rgb: {rgb.shape} {rgb.dtype}")
        if not (rgb == 128).all():
            raise AssertionError("synthetic fixture writes 128,128,128")


def test_label_override_swaps_labels_only():
    from segmenter.derived import build_c1_eval_bundle
    with tempfile.TemporaryDirectory() as td:
        room, bundle = _scene(Path(td), PERFECT)
        c1, rep1 = build_c1_eval_bundle(room, bundle, "s", min_vertices=4)
        # matched preds are 10 (table) and 20 (chair)
        c2, rep2 = build_c1_eval_bundle(room, bundle, "s", min_vertices=4,
                                        label_override={10: "desk",
                                                        20: "chair"})
        if rep2["provenance"] != "learned_labels_c2":
            raise AssertionError(f"provenance: {rep2['provenance']}")
        lab1 = {e.identity.object_uid: e.identity.display_label
                for e in c1.entities}
        lab2 = {e.identity.object_uid: e.identity.display_label
                for e in c2.entities}
        if set(lab1) != set(lab2):
            raise AssertionError("entity sets must be identical")
        if lab2["obj_10"] != "desk" or lab2["obj_20"] != "chair":
            raise AssertionError(f"override not applied: {lab2}")
        if lab1["obj_10"] != "table":
            raise AssertionError(f"C1 oracle label must be table: {lab1}")
        inj = rep2["oracle_injections"][0]
        if "learned_label" not in inj or "oracle_class" not in inj:
            raise AssertionError(f"injection must record both: {inj}")
        # missing override for a matched pred must be a hard error
        try:
            build_c1_eval_bundle(room, bundle, "s", min_vertices=4,
                                 label_override={10: "desk"})
            raise AssertionError("partial override must raise")
        except ValueError:
            pass


def test_clip_labeler_smoke():
    try:
        import open_clip  # noqa: F401
        import torch      # noqa: F401
    except ImportError:
        print("  (skipped: torch/open_clip not installed — optional dep)")
        return
    from segmenter.clip_labeler import ClipLabeler
    xyz = np.random.default_rng(0).uniform(0, 1, (200, 3))
    rgb = np.full((200, 3), 100, dtype=np.uint8)
    views = render_views(xyz, rgb)
    lab = ClipLabeler()
    r1 = lab.classify(list(views.values()), ["chair", "table", "vase"])
    r2 = lab.classify(list(views.values()), ["chair", "table", "vase"])
    if r1 != r2:
        raise AssertionError("labeler must be deterministic")
    if {x["label"] for x in r1} != {"chair", "table", "vase"}:
        raise AssertionError(f"ranking must cover the vocabulary: {r1}")


TESTS = [
    test_render_views_deterministic_and_bounded,
    test_parse_vertices_with_colors_matches_positions,
    test_label_override_swaps_labels_only,
    test_clip_labeler_smoke,
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

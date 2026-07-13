"""Tests for the C1 closeout paths: c1_failure_classes, c1_reresolve, and
the c1_run_v2 report schema (B-relative PR + merge-aware attribution).

Run: python tests/segmenter/test_closeout_paths.py

Synthetic three-cube scene for failure classes: cube A "table" recovered,
cube B "chair" merged into A's winning mask DESPITE a viable raw proposal,
cube C "lamp" with no raw proposal at all.
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmenter.base import SegmentationOutput, load_segmentation_output, \
    save_segmentation_output, sha256_file
from segmenter.mask_resolve import MaskResolveConfig, resolve_masks
from tests.segmenter.test_c1_pipeline import (
    _CUBE_TRIS, _cube_verts, _scene, _write_raw_ply, _write_semantic_ply,
    PERFECT,
)
from tools.c1_failure_classes import classify
from tools.c1_reresolve import reresolve


def _three_entity_scene(tmp: Path):
    """A 'table' (oid1), B 'chair' (oid2), C 'lamp' (oid3) — all entities.
    Raw masks: m0 (.9) = A + half of B; m1 (.5) = all of B; nothing covers C.
    Resolution at min_vertices=5 leaves m1's 4 surviving vertices unclaimed:
    A recovered, B merged-with-viable-proposal, C no_raw_proposal."""
    verts = _cube_verts((0, 0, 0)) + _cube_verts((5, 0, 0)) + _cube_verts((10, 0, 0))
    xyz = np.array(verts, dtype=np.float64)
    tris, oids = [], []
    for j in range(3):
        for t in _CUBE_TRIS:
            tris.append(tuple(8 * j + i for i in t))
            oids.append(j + 1)
    tris = np.array(tris, dtype=np.int64)
    oids = np.array(oids, dtype=np.int64)

    def obj(i, cls, cx):
        return {"id": i, "class_name": cls,
                "oriented_bbox": {
                    "abb": {"center": [cx, 0.5, 0.5], "sizes": [1.0, 1.0, 1.0]},
                    "orientation": {"rotation": [0.0, 0.0, 0.0, 1.0],
                                    "translation": [0.0, 0.0, 0.0]}}}
    room = tmp / "room3"
    (room / "habitat").mkdir(parents=True)
    _write_raw_ply(room / "mesh.ply", xyz, tris)
    _write_semantic_ply(room / "habitat" / "mesh_semantic.ply", xyz, tris, oids)
    (room / "habitat" / "info_semantic.json").write_text(json.dumps({
        "gravity_dir": [0.0, 0.0, -1.0],
        "objects": [obj(1, "table", 0.5), obj(2, "chair", 5.5),
                    obj(3, "lamp", 10.5)],
    }), encoding="utf-8")

    masks = np.zeros((2, 24), dtype=bool)
    masks[0, 0:12] = True      # A + half of B, score .9
    masks[1, 8:16] = True      # all of B (viable), score .5
    scores = np.array([0.9, 0.5])
    ids = resolve_masks(masks, scores, MaskResolveConfig(min_score=0.2, min_vertices=5))
    seg = SegmentationOutput(
        input_mesh_sha256=sha256_file(room / "mesh.ply"), n_vertices=24,
        segmenter_name="t", segmenter_version="0", config_params_json="{}",
        vertex_instance_ids=ids).finalize()
    bundle = tmp / "bundle3"
    save_segmentation_output(seg, bundle)
    np.savez_compressed(bundle / "raw_masks.npz",
                        masks_packed=np.packbits(masks, axis=1),
                        n_vertices=np.int64(24), scores=scores)
    return room, bundle


def test_failure_classes_three_way():
    with tempfile.TemporaryDirectory() as td:
        room, bundle = _three_entity_scene(Path(td))
        r = classify(room, bundle)
        if r["counts"] != {"recovered": 1, "merged": 1,
                           "lost_by_resolver": 0, "no_raw_proposal": 1}:
            raise AssertionError(f"failure counts wrong: {r['counts']}")
        if r["n_merged_with_viable_raw_proposal"] != 1:
            raise AssertionError(f"cube B had a viable raw mask (iou 1.0): {r}")
        by_class = {x["class"]: x["failure_class"] for x in r["per_object"]}
        if by_class != {"table": "recovered", "chair": "merged",
                        "lamp": "no_raw_proposal"}:
            raise AssertionError(f"per-object classes wrong: {by_class}")
        # orthogonal proposal-coverage fields (independent of failure_class):
        # table viable via m0 (iou .667), chair viable via m1 (iou 1.0), lamp no
        if r["n_viable_raw_at_05"] != 2 or abs(r["raw_proposal_recall_at_iou"]["0.5"] - 2/3) > 1e-9:
            raise AssertionError(f"raw-proposal coverage wrong: {r['raw_proposal_recall_at_iou']}")
        if r["n_viable_raw_not_recovered"] != 1:
            raise AssertionError("chair is viable-but-not-recovered (composition loss)")
        viable = {x["class"]: x["has_viable_raw_proposal"] for x in r["per_object"]}
        if viable != {"table": True, "chair": True, "lamp": False}:
            raise AssertionError(f"per-object viability wrong: {viable}")


def test_reresolve_confidence_and_provenance():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _, bundle = _three_entity_scene(tmp)
        out = tmp / "re"
        seg = reresolve(bundle, out, min_score=0.8, min_vertices=5)
        if seg.instance_ids() != [0]:
            raise AssertionError(f"only mask 0 (score .9) survives 0.8: {seg.instance_ids()}")
        if seg.instance_confidence != {0: 0.9}:
            raise AssertionError(f"confidence must come from raw scores: {seg.instance_confidence}")
        loaded = load_segmentation_output(out)      # round-trips + validates
        cfg = json.loads(loaded.config_params_json)
        if "reresolved_locally" not in cfg or cfg["reresolved_locally"]["min_score"] != 0.8:
            raise AssertionError(f"provenance missing: {cfg}")
        if not (out / "raw_masks.npz").exists():
            raise AssertionError("raw masks must be carried into the new bundle")
        table = json.loads((out / "instance_table.json").read_text())
        if table[0]["confidence"] != 0.9:
            raise AssertionError(f"instance_table confidence still null: {table}")


def test_c1_run_v2_report_schema():
    from tools.c1_run import main as c1_run_main
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        room, bundle = _scene(tmp, PERFECT)
        rc = c1_run_main([str(room), str(bundle), "synthetic_scene",
                          "--out-dir", str(tmp / "out"), "--min-vertices", "4"])
        if rc != 0:
            raise AssertionError(f"c1_run failed: rc={rc}")
        r = json.loads((tmp / "out" / "synthetic_scene_c1_run.json").read_text())
        if r["schema"] != "c1_run_v2":
            raise AssertionError(f"schema: {r['schema']}")
        pr = r["answer_pr_vs_B"]
        for k in ("micro_recall_vs_B", "micro_precision_vs_B",
                  "support_answer_recall_vs_B"):
            if k not in pr:
                raise AssertionError(f"missing {k} in answer_pr_vs_B")
        if not r["per_question_pr_vs_B"]:
            raise AssertionError("per-question PR must be populated")
        q0 = next(iter(r["per_question_pr_vs_B"].values()))
        if set(q0) != {"precision_vs_B", "recall_vs_B", "n_B", "n_C1", "n_common"}:
            raise AssertionError(f"per-question keys wrong: {q0}")
        for d in r["answer_diffs"]["B_vs_C1_instance_extraction"]:
            if "lost_attribution" not in d or "gained_attribution" not in d:
                raise AssertionError(f"diff missing attributions: {d}")


TESTS = [
    test_failure_classes_three_way,
    test_reresolve_confidence_and_provenance,
    test_c1_run_v2_report_schema,
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

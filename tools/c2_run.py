"""C2.0 — learned labels on matched instances (one-command local run).

  .venv/bin/python tools/c2_run.py <room_dir> <ms02_bundle_dir> <scene_id> \
      [--key eval/questions/phase8/<scene>_qa.json] [--out-dir runs/phase8_c2]

Protocol: docs/c2_matched_labels_protocol.md (predeclared 2026-08-01,
measurement-first — no accuracy thresholds). Pipeline:

  1. exact evaluator -> the C1 match table (matched entity instances),
  2. per matched instance: gravity-aligned point-splat views of its raw
     mesh.ply vertices+colors -> pinned CLIP zero-shot over the scene's
     class vocabulary (closed-set; declared leak),
  3. label accuracy vs oracle classes (top-1/top-3 + support-class slice),
  4. downstream QA: C1 (oracle labels) vs C2 (learned labels) rows against
     the human key — same entities, same geometry, same surfaces; labels
     are the ONLY difference, so the delta is attributable exactly.

Needs the optional torch/open_clip deps (project venv). Zero GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.question_battery import _runs
from demo.replica_habitat_import import _gravity_align_matrix, _aligned_structural_surfaces
from graph.builder import build_graph
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer
from segmenter.base import load_segmentation_output
from segmenter.derived import ROOM_0_Z_TRANSLATION, STRUCTURAL_CLASSES, DROP_CLASSES, build_c1_eval_bundle
from segmenter.instance_render import render_views
from segmenter.ply import parse_vertices_with_colors
from tools.c1_exact_eval import SUPPORT_OWNER_CLASSES, evaluate
from tools.c1_joint_ceiling import score_against_key


def scene_vocabulary(room_dir: Path) -> list[str]:
    info = json.loads((room_dir / "habitat" / "info_semantic.json").read_text())
    return sorted({o.get("class_name", "") for o in info["objects"]} - {""})


def label_matched_instances(room_dir: Path, bundle_dir: Path) -> dict:
    """Steps 1-2: render + classify every matched ENTITY instance."""
    from segmenter.clip_labeler import ClipLabeler

    report = evaluate(room_dir, bundle_dir)
    seg = load_segmentation_output(bundle_dir)
    xyz, rgb = parse_vertices_with_colors(room_dir / "mesh.ply")
    info = json.loads((room_dir / "habitat" / "info_semantic.json").read_text())
    g = info["gravity_dir"]
    R0 = _gravity_align_matrix((float(g[0]), float(g[1]), float(g[2])))
    R, _, _, _ = _aligned_structural_surfaces(info, R0, ROOM_0_Z_TRANSLATION)
    xyz = np.einsum("ij,nj->ni", R, xyz)

    vocab = scene_vocabulary(room_dir)
    labeler = ClipLabeler()
    ids = seg.vertex_instance_ids

    per_instance = []
    for m in report["matches"]:
        oc = m["oracle_class"]
        if not oc or oc in STRUCTURAL_CLASSES or oc in DROP_CLASSES:
            continue
        sel = ids == m["pred_id"]
        views = render_views(xyz[sel], rgb[sel])
        ranking = labeler.classify(list(views.values()), vocab)
        per_instance.append({
            "pred_id": m["pred_id"], "oracle_id": m["oracle_id"],
            "oracle_class": oc, "iou": m["iou"],
            "n_vertices": int(sel.sum()),
            "top3": ranking[:3],
            "learned_label": ranking[0]["label"],
            "correct_top1": ranking[0]["label"] == oc,
            "correct_top3": oc in [r["label"] for r in ranking[:3]],
        })

    n = len(per_instance)
    sup_rows = [r for r in per_instance
                if r["oracle_class"] in SUPPORT_OWNER_CLASSES]
    accuracy = {
        "n_matched_entity_instances": n,
        "top1": round(sum(r["correct_top1"] for r in per_instance) / n, 4) if n else None,
        "top3": round(sum(r["correct_top3"] for r in per_instance) / n, 4) if n else None,
        "support_class_slice": {
            "n": len(sup_rows),
            "top1": (round(sum(r["correct_top1"] for r in sup_rows)
                           / len(sup_rows), 4) if sup_rows else None),
            "predicted_support_when_oracle_support": (
                round(sum(r["learned_label"] in SUPPORT_OWNER_CLASSES
                          for r in sup_rows) / len(sup_rows), 4)
                if sup_rows else None),
            "predicted_support_when_oracle_not": (
                round(sum(r["learned_label"] in SUPPORT_OWNER_CLASSES
                          for r in per_instance
                          if r["oracle_class"] not in SUPPORT_OWNER_CLASSES)
                      / max(n - len(sup_rows), 1), 4) if n else None),
        },
    }
    from segmenter.clip_labeler import MODEL_NAME, PRETRAINED, PROMPTS
    return {
        "labeler": {"model": MODEL_NAME, "pretrained": PRETRAINED,
                    "weights_sha256": labeler.weights_sha256,
                    "prompts": list(PROMPTS)},
        "source_bundle_output_sha256": seg.output_sha256,
        "vocabulary": vocab,
        "per_instance": per_instance,
        "label_accuracy": accuracy,
    }


def qa_rows(room_dir: Path, bundle_dir: Path, scene_id: str, key: dict,
            learned: dict[int, str]) -> dict:
    """Step 4: C1 vs C2 through the identical frozen downstream."""
    router = Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                    verbalizer=StandardVerbalizer())
    ctx = ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))
    ev = evaluate(room_dir, bundle_dir)
    uid_map = {f"obj_{m['pred_id']}": f"obj_{m['oracle_id']}"
               for m in ev["matches"]}
    canonical = {f"obj_{m['oracle_id']}": m["oracle_class"]
                 for m in ev["matches"] if m["oracle_class"]}

    rows, anchors = {}, {}
    for name, override in (("C1_oracle_labels", None),
                           ("C2_learned_labels", learned)):
        arts, _ = build_c1_eval_bundle(room_dir, bundle_dir, scene_id,
                                       min_vertices=20,
                                       label_override=override)
        bundle, _ = build_graph(arts, _runs(),
                                density_policy="phase2_telemetry_only")
        qa = score_against_key(key, bundle, router, ctx, uid_map)
        # semantic citation: among cited HITS (uid correct), is the
        # DISPLAYED label the canonical class? A uid-correct answer can
        # still verbalize the wrong learned label — score that separately.
        # Labels are looked up by the entity's OWN uid (pred space) —
        # translated-uid keys would collide with unmatched pred uids.
        label_by_uid = {e.identity.object_uid: e.identity.display_label
                        for e in arts.entities}
        n_hits = n_sem = 0
        for q in key["questions"]:
            ans = router.answer(q["question"], bundle, ctx)
            must = set(q["expected_must_contain"])
            for u in ans.cited_uids:
                # same translation rule as the frozen scorer: unmatched
                # pred uids are prefixed so they can never collide with
                # oracle uids in the key
                shown = uid_map[u] if u in uid_map else f"pred:{u}"
                if shown in must:
                    n_hits += 1
                    n_sem += label_by_uid.get(u) == canonical.get(shown)
        rows[name] = {
            "metric_note": ("uid_micro_* score UID/structural MEMBERSHIP "
                            "vs the key (the key cites uids, not names); "
                            "semantic_citation scores whether uid-correct "
                            "citations also carry the canonical label"),
            "uid_micro_precision": (None if qa["micro_precision"] is None
                                    else round(qa["micro_precision"], 4)),
            "uid_micro_recall": (None if qa["micro_recall"] is None
                                 else round(qa["micro_recall"], 4)),
            "support_hits": (qa["per_relation"].get("ON_ENTITY_SURFACE")
                             or {}).get("n_hit"),
            "semantic_citation": {
                "n_uid_correct_citations": n_hits,
                "n_with_canonical_label": n_sem,
                "accuracy": round(n_sem / n_hits, 4) if n_hits else None,
            },
            "n_graph_edges": len(bundle.edges),
            "per_question": {qid: {k: q[k] for k in
                                   ("actual_outcome", "n_cited", "n_hit")}
                             for qid, q in qa["per_question"].items()},
        }
        anchors[name] = {
            cls: sorted(e.identity.object_uid for e in arts.entities
                        if e.identity.display_label == cls)
            for cls in SUPPORT_OWNER_CLASSES}
    integrity = {cls: {"C1": anchors["C1_oracle_labels"][cls],
                       "C2": anchors["C2_learned_labels"][cls],
                       "same": (anchors["C1_oracle_labels"][cls]
                                == anchors["C2_learned_labels"][cls])}
                 for cls in SUPPORT_OWNER_CLASSES
                 if anchors["C1_oracle_labels"][cls]
                 or anchors["C2_learned_labels"][cls]}
    return {"rows": rows, "support_anchor_integrity": integrity}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("room_dir", type=Path)
    ap.add_argument("bundle_dir", type=Path)
    ap.add_argument("scene_id")
    ap.add_argument("--key", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "runs" / "phase8_c2")
    ap.add_argument("--sidecar-dir", type=Path,
                    default=REPO_ROOT / "eval" / "predictions" / "phase8_c2",
                    help="TRACKED sanitized prediction sidecar (pins + "
                         "per-instance labels; consumed by the offline MVP)")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = label_matched_instances(args.room_dir, args.bundle_dir)
    report = {
        "schema": "c2_run_v1",
        "protocol": "docs/c2_matched_labels_protocol.md",
        "scene_id": args.scene_id,
        "bundle_dir": str(args.bundle_dir),
        "isolation": ("labeler saw mesh.ply vertices/colors, bundle "
                      "instance ids, and the scene class vocabulary ONLY; "
                      "oracle labels used solely for scoring"),
        **labels,
    }
    if args.key:
        key = json.loads(args.key.read_text())
        if key.get("answer_key_type") != "human_verified":
            raise ValueError(f"{args.key} is not human_verified")
        learned = {r["pred_id"]: r["learned_label"]
                   for r in labels["per_instance"]}
        report["qa_vs_human_key"] = qa_rows(
            args.room_dir, args.bundle_dir, args.scene_id, key, learned)

    out = args.out_dir / f"{args.scene_id}_c2_run.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    # tracked, sanitized sidecar: pins + predictions only (no local paths)
    args.sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "schema": "c2_label_sidecar_v1",
        "protocol": "docs/c2_matched_labels_protocol.md",
        "scene_id": args.scene_id,
        "labeler": labels["labeler"],
        "applies_to_bundle_output_sha256": labels["source_bundle_output_sha256"],
        "vocabulary": labels["vocabulary"],
        "label_accuracy": labels["label_accuracy"],
        "per_instance": labels["per_instance"],
        "note": ("evaluation-only C2.0 predictions on matched instances; "
                 "consumed by tools/mvp_demo.py for the torch-free C2 row"),
    }
    sc = args.sidecar_dir / f"{args.scene_id}_c2_labels.json"
    sc.write_text(json.dumps(sidecar, indent=1, sort_keys=True) + "\n",
                  encoding="utf-8")

    acc = labels["label_accuracy"]
    print(f"[c2] {args.scene_id}: matched entities={acc['n_matched_entity_instances']} "
          f"top1={acc['top1']} top3={acc['top3']} "
          f"support-slice top1={acc['support_class_slice']['top1']}")
    if args.key:
        for name, r in report["qa_vs_human_key"]["rows"].items():
            sem = r["semantic_citation"]
            print(f"  {name}: uid-P={r['uid_micro_precision']} "
                  f"uid-R={r['uid_micro_recall']} "
                  f"support={r['support_hits']} "
                  f"semantic-citation={sem['accuracy']} "
                  f"({sem['n_with_canonical_label']}/{sem['n_uid_correct_citations']}) "
                  f"edges={r['n_graph_edges']}")
        bad = [c for c, v in
               report["qa_vs_human_key"]["support_anchor_integrity"].items()
               if not v["same"]]
        print(f"  support anchors changed: {bad or 'none'}")
    print(f"report -> {out}")
    print(f"sidecar -> {sc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

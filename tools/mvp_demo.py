"""MVP-v0 demo — one offline command comparing A/B/C1 vs the human keys.

  python3 tools/mvp_demo.py [--out-dir runs/mvp_v0] [--scene SCENE_ID]
                            [--check-determinism] [--no-html]

Spec: docs/mvp_v0_demo_spec.md (approved 2026-08-01, spec defaults).
Runs the SAME frozen graph + Router over variant A (oracle boxes), B
(mesh-derived boxes) and C1 (frozen Mask3D @0.2 instances, oracle labels
injected), scores every human-key question, and emits:

  runs/mvp_v0/<scene>_mvp.json   deterministic per-scene report
  runs/mvp_v0/aggregate.json     deterministic headline table
  runs/mvp_v0/report.html        self-contained HTML (tools/mvp_report_html)
  runs/mvp_v0/run_env.json       NON-deterministic env (excluded from
                                 the determinism check)

Hard-fail guarantees (spec acceptance criteria):
  - every pinned input hash is verified before any work,
  - only human_verified keys are accepted,
  - recomputed room_2 reference rows must equal the committed values,
  - --check-determinism runs everything twice and byte-compares.
No GPU, no network, no writes outside --out-dir.
"""
from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.question_battery import _runs
from demo.replica_habitat_import import import_habitat_room
from demo.replica_mesh_import import import_mesh_room
from graph.builder import build_graph
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer
from segmenter.derived import build_c1_eval_bundle
from tools.c1_exact_eval import evaluate
from tools.c1_joint_ceiling import _OUTCOME_MAP, score_against_key

SCHEMA = "mvp_v0_report_v1"
MS02_DIR = REPO_ROOT / "runs" / "phase8_c1" / "bundles_ms02"
MANIFEST = REPO_ROOT / "docs" / "c1_artifact_manifest.json"
LOCK = REPO_ROOT / "tools" / "replica_scenes.lock.json"
SCENE_MANIFEST = REPO_ROOT / "eval" / "questions" / "phase8" / "scene_manifest.json"

# Predeclared scene set (spec): room_0 has a human key but Mask3D was never
# run on it and v0 spends no GPU -> its C1 column is "not run", never 0.
# C2 rows (spec addendum, owner-instructed 2026-08-01) are EVALUATION-ONLY:
# labels come from the committed C2.0 prediction sidecars
# (eval/predictions/phase8_c2/, pinned to the ms02 bundles) — no torch here.
SCENES = [
    {"scene_id": "replica_room_0", "short": "room_0", "variants": ["A", "B"]},
    {"scene_id": "replica_room_1", "short": "room_1",
     "variants": ["A", "B", "C1", "C2"]},
    {"scene_id": "replica_room_2", "short": "room_2",
     "variants": ["A", "B", "C1", "C2"]},
]
SIDECAR_DIR = REPO_ROOT / "eval" / "predictions" / "phase8_c2"

# Committed reference values (4 dp) the demo must reproduce exactly —
# spec acceptance criterion 3. Sources: runs/phase8_c1/joint_ceiling*,
# runs/phase8_c2/ (C2.0 protocol results).
REFERENCE = {
    "replica_room_2": {
        "A": {"micro_precision": 0.9524, "micro_recall": 0.4082},
        "C1": {"micro_precision": 1.0, "micro_recall": 0.2449},
        "C2": {"micro_precision": 1.0, "micro_recall": 0.2041},
    },
}

ISOLATION = ("C1 labels and structural surfaces are INJECTED from the "
             "oracle via exact vertex correspondence; only instance "
             "boundaries are learned. See docs/mesh_pipeline_contract.md.")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rnd(x, nd=4):
    return None if x is None else round(x, nd)


def verify_inputs(rows: list[dict]) -> dict:
    """Hash-verify every pinned input; return the provenance seed."""
    scene_dirs = {s["scene_id"]: Path(s["room_dir"])
                  for s in json.loads(SCENE_MANIFEST.read_text())["scenes"]}
    lock = {f["relpath"]: f for f in json.loads(LOCK.read_text())["files"]}
    manifest = json.loads(MANIFEST.read_text())["scenes"]
    prov: dict = {"inputs": {}, "keys": {}, "bundles": {}}

    for row in rows:
        sid, short = row["scene_id"], row["short"]
        room = scene_dirs[sid]
        row["room_dir"] = room
        files = {"info_semantic.json": room / "habitat" / "info_semantic.json",
                 "mesh_semantic.ply": room / "habitat" / "mesh_semantic.ply"}
        entry = {}
        for name, path in files.items():
            if not path.exists():
                raise FileNotFoundError(f"{sid}: missing {path}")
            digest = sha256_file(path)
            pin = lock.get(f"{short}/habitat/{name}")
            if pin is not None:
                if pin["sha256"] != digest:
                    raise ValueError(f"{sid}: {name} hash mismatch vs lock")
                entry[name] = {"sha256": digest, "pinned": True}
            else:
                entry[name] = {"sha256": digest, "pinned": False,
                               "note": "not in replica_scenes.lock.json "
                                       "(pre-Phase8 scene); hash recorded, "
                                       "not verified against a pin"}
        prov["inputs"][sid] = entry

        key_path = REPO_ROOT / "eval" / "questions" / "phase8" / f"{sid}_qa.json"
        key = json.loads(key_path.read_text())
        if key.get("answer_key_type") != "human_verified":
            raise ValueError(f"{sid}: key is not human_verified — refused")
        row["key"] = key
        prov["keys"][sid] = {"fixture_id": key["fixture_id"],
                             "sha256": sha256_file(key_path)}

        if "C1" in row["variants"]:
            bundle = MS02_DIR / short
            meta = json.loads((bundle / "meta.json").read_text())
            pinned = manifest[short]["frozen_ms02_bundle"]["output_sha256"]
            if meta["output_sha256"] != pinned:
                raise ValueError(f"{sid}: ms02 bundle hash mismatch vs "
                                 f"docs/c1_artifact_manifest.json")
            row["bundle_dir"] = bundle
            prov["bundles"][sid] = {"output_sha256": meta["output_sha256"],
                                    "path": f"runs/phase8_c1/bundles_ms02/{short}",
                                    "resolver": "frozen MIN_SCORE=0.2 "
                                                "min_vertices=20"}
        if "C2" in row["variants"]:
            sc_path = SIDECAR_DIR / f"{sid}_c2_labels.json"
            sidecar = json.loads(sc_path.read_text())
            if (sidecar["applies_to_bundle_output_sha256"]
                    != prov["bundles"][sid]["output_sha256"]):
                raise ValueError(f"{sid}: C2 sidecar pinned to a different "
                                 f"bundle than the frozen ms02 bundle")
            row["label_override"] = {int(r["pred_id"]): r["learned_label"]
                                     for r in sidecar["per_instance"]}
            prov.setdefault("c2_sidecars", {})[sid] = {
                "path": f"eval/predictions/phase8_c2/{sid}_c2_labels.json",
                "labeler": sidecar["labeler"],
                "label_accuracy": sidecar["label_accuracy"],
                "status": "EVALUATION-ONLY (labels from the committed C2.0 "
                          "sidecar; docs/c2_matched_labels_protocol.md)"}
    return prov


def build_variant_arts(variant: str, row: dict, min_vertices: int = 20):
    """Returns (arts, uid_map or None, extras dict). Keys starting with
    '_' in extras are internal and stripped from the report."""
    room, sid = row["room_dir"], row["scene_id"]
    if variant == "A":
        return import_habitat_room(room, sid), None, {}
    if variant == "B":
        return import_mesh_room(room, sid), None, {}
    override = row["label_override"] if variant == "C2" else None
    ev = evaluate(room, row["bundle_dir"])
    arts, _ = build_c1_eval_bundle(room, row["bundle_dir"], sid,
                                   min_vertices=min_vertices,
                                   label_override=override)
    uid_map = {f"obj_{m['pred_id']}": f"obj_{m['oracle_id']}"
               for m in ev["matches"]}
    iou = {f"obj_{m['pred_id']}": rnd(m["iou"]) for m in ev["matches"]}
    n5 = ev["n_entity_matches_at_iou"]["0.5"]
    extras = {
        "match_iou_by_pred_uid": iou,
        "entity_matches_at_05": f"{n5}/{ev['n_oracle_entity_instances']}",
        "_canonical": {f"obj_{m['oracle_id']}": m["oracle_class"]
                       for m in ev["matches"] if m["oracle_class"]},
    }
    if variant == "C2":
        extras["labels"] = ("EVALUATION-ONLY learned labels from the "
                            "committed C2.0 sidecar")
    return arts, uid_map, extras


def run_variant(variant: str, row: dict, router, ctx,
                min_vertices: int = 20) -> dict:
    arts, uid_map, extras = build_variant_arts(variant, row, min_vertices)
    bundle, _ = build_graph(arts, _runs(), density_policy="phase2_telemetry_only")
    labels = {e.identity.object_uid: e.identity.display_label
              for e in arts.entities}

    # metrics via the UNCHANGED frozen scorer
    qa = score_against_key(row["key"], bundle, router, ctx, uid_map)

    # per-question answer detail (same translation; cross-checked below)
    questions = []
    for q in row["key"]["questions"]:
        ans = router.answer(q["question"], bundle, ctx)
        must, must_not = set(q["expected_must_contain"]), set(q["expected_must_not_contain"])
        cited = []
        n_hit = 0
        for u in sorted(ans.cited_uids):
            if uid_map is None:
                shown, matched_iou = u, None
            elif u in uid_map:
                shown, matched_iou = uid_map[u], extras["match_iou_by_pred_uid"][u]
            else:
                shown, matched_iou = f"pred:{u}", None
            status = ("hit" if shown in must else
                      "violation" if shown in must_not else "extra")
            n_hit += status == "hit"
            cited.append({"uid": shown, "label": labels.get(u, "?"),
                          "status": status,
                          **({"matched_iou": matched_iou}
                             if matched_iou is not None else
                             {"unlabeled_segment": True}
                             if uid_map is not None and u not in uid_map
                             else {})})
        srow = qa["per_question"][q["question_id"]]
        if srow["n_hit"] != n_hit:
            raise AssertionError(f"scorer divergence on {q['question_id']}")
        questions.append({
            "question_id": q["question_id"],
            "question": q["question"],
            "expected_outcome": q["expected_outcome"],
            "actual_outcome": _OUTCOME_MAP.get(ans.outcome, "unknown"),
            "verbalized": ans.text,
            "cited": cited,
            "missed": sorted(must - {c["uid"] for c in cited}),
            "exhaustive": bool(q.get("exhaustive")),
            "precision": rnd(srow.get("precision")),
            "recall": rnd(srow.get("recall")),
        })

    # semantic citation (C1/C2 rows): among uid-correct citations, does the
    # DISPLAYED label match the canonical class? micro P/R score uid
    # membership only — a uid-correct answer can verbalize a wrong learned
    # label, and this metric is where that shows.
    semantic = None
    canonical = extras.pop("_canonical", None)
    if canonical is not None:
        n_hits = n_sem = 0
        for q in questions:
            for c in q["cited"]:
                if c["status"] == "hit":
                    n_hits += 1
                    n_sem += c["label"] == canonical.get(c["uid"])
        semantic = {"n_uid_correct_citations": n_hits,
                    "n_with_canonical_label": n_sem,
                    "accuracy": rnd(n_sem / n_hits) if n_hits else None}

    return {
        "variant": variant,
        **extras,
        "graph_bundle_hash": bundle.bundle_hash,
        "n_entities": len(arts.entities),
        "n_graph_edges": len(bundle.edges),
        "metric_note": ("micro_precision/micro_recall score UID/structural "
                        "MEMBERSHIP vs the key (the key cites uids, not "
                        "names); semantic_citation scores whether "
                        "uid-correct citations also carry the canonical "
                        "label"),
        "micro_precision": rnd(qa["micro_precision"]),
        "micro_recall": rnd(qa["micro_recall"]),
        "semantic_citation": semantic,
        "per_relation": {rel: {k: rnd(v) if isinstance(v, float) else v
                               for k, v in d.items()}
                         for rel, d in qa["per_relation"].items()},
        "questions": questions,
    }


def run_all(out_dir: Path, only_scene: str | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [dict(s) for s in SCENES
            if only_scene is None or s["scene_id"] == only_scene]
    if not rows:
        raise ValueError(f"unknown scene {only_scene!r}")
    prov_seed = verify_inputs(rows)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True).stdout.strip()
    router = Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                    verbalizer=StandardVerbalizer())
    ctx = ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))

    written, headline = [], []
    for row in rows:
        sid = row["scene_id"]
        variants = {v: run_variant(v, row, router, ctx)
                    for v in row["variants"]}
        for v, ref in REFERENCE.get(sid, {}).items():
            if v not in variants:
                continue
            got = {k: variants[v][k] for k in ref}
            if got != ref:
                raise AssertionError(
                    f"REFERENCE MISMATCH {sid}/{v}: expected {ref}, got {got} "
                    f"— the demo must reproduce committed values exactly")
        report = {
            "schema": SCHEMA,
            "scene_id": sid,
            "spec": "docs/mvp_v0_demo_spec.md",
            # key facts embedded so the HTML renders from THIS JSON only
            "key_questions": {q["question_id"]: {
                "question": q["question"],
                "expected_outcome": q["expected_outcome"],
                "exhaustive": bool(q.get("exhaustive")),
                "expected_must_contain": q["expected_must_contain"],
                "expected_must_not_contain": q["expected_must_not_contain"],
                "candidate_labels": q.get("candidate_labels", {}),
            } for q in row["key"]["questions"]},
            "provenance": {
                "git_commit": commit,
                "inputs": prov_seed["inputs"][sid],
                "key": prov_seed["keys"][sid],
                "c1_bundle": prov_seed["bundles"].get(sid),
                "scorer": "tools/c1_joint_ceiling.py::score_against_key",
                "completeness_profile": "oracle (empty means not-in-graph, "
                                        "NOT proven-absent)",
                "isolation_statement": ISOLATION,
            },
            "c1_status": ("frozen Mask3D @0.2 reference"
                          if "C1" in row["variants"] else
                          "not run (no GPU in MVP-v0; see spec)"),
            "variants": variants,
        }
        p = out_dir / f"{sid}_mvp.json"
        p.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n",
                     encoding="utf-8")
        written.append(p)
        for v, r in variants.items():
            headline.append({
                "scene_id": sid, "variant": v,
                "micro_precision": r["micro_precision"],
                "micro_recall": r["micro_recall"],
                "semantic_citation": ((r["semantic_citation"] or {})
                                      .get("accuracy")
                                      if r.get("semantic_citation") else None),
                "support_hits": (r["per_relation"].get("ON_ENTITY_SURFACE")
                                 or {}).get("n_hit"),
                "n_graph_edges": r["n_graph_edges"],
                "entity_matches_at_05": r.get("entity_matches_at_05"),
            })

    agg = {
        "schema": SCHEMA + "_aggregate",
        "spec": "docs/mvp_v0_demo_spec.md",
        "git_commit": commit,
        "comparability": ("Scores are vs human_verified Phase 8 keys — a "
                          "separate track, never comparable to the legacy "
                          "v1 benchmark or per-phase scorecards."),
        "headline": headline,
        "reference_check": "PASSED (room_2 rows reproduce committed values)",
    }
    p = out_dir / "aggregate.json"
    p.write_text(json.dumps(agg, indent=1, sort_keys=True) + "\n",
                 encoding="utf-8")
    written.append(p)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "runs" / "mvp_v0")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--check-determinism", action="store_true")
    ap.add_argument("--no-html", action="store_true")
    args = ap.parse_args(argv)

    import platform
    import time
    t0 = time.time()

    if args.check_determinism:
        a, b = args.out_dir / "_det_a", args.out_dir / "_det_b"
        files_a = run_all(a, args.scene)
        files_b = run_all(b, args.scene)
        if not args.no_html:
            from tools.mvp_report_html import build_html
            files_a.append(build_html(a))
            files_b.append(build_html(b))
        bad = [fa.name for fa, fb in zip(files_a, files_b)
               if not filecmp.cmp(fa, fb, shallow=False)]
        if bad:
            print(f"DETERMINISM FAIL: {bad}")
            return 1
        print(f"determinism check PASSED over {len(files_a)} files "
              f"({time.time()-t0:.0f}s)")
        return 0

    files = run_all(args.out_dir, args.scene)
    if not args.no_html:
        from tools.mvp_report_html import build_html
        files.append(build_html(args.out_dir))
    (args.out_dir / "run_env.json").write_text(json.dumps({
        "note": "NON-deterministic run metadata; excluded from the "
                "determinism check by design",
        "wall_seconds": round(time.time() - t0, 1),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }, indent=1) + "\n", encoding="utf-8")
    for f in files:
        try:
            print(f"wrote {f.relative_to(REPO_ROOT)}")
        except ValueError:
            print(f"wrote {f}")
    print(f"done in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

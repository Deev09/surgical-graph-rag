"""C1-M2C stage 0b — JOINT + task-level ceiling (oracle-guided, zero GPU).

  python3 tools/c1_joint_ceiling.py --room-dir DIR --bundle DIR --scene-id ID \
      [--key eval/questions/phase8/<scene>_qa.json] [--out-dir DIR]

Stage 0 (tools/c1_composition_ceiling.py) measured a PER-ENTITY selection
ceiling: 30/53 room_2 entities have a viable single raw mask. That leaves a
logical gap: per-entity optima may CONFLICT — two near-perfect masks can
claim the same vertices, and a dense assignment must give each vertex to
one instance. Stage 0b closes the gap by measuring what survives JOINTLY:

  1. oracle-guided selection: each viable entity (best single-mask IoU >=
     0.5, any score) nominates its best mask; mask collisions resolved by
     IoU priority (an entity whose best mask is taken falls back to its
     next viable mask, if any),
  2. the selected masks are materialized through the FROZEN resolver
     mechanics (segmenter/mask_resolve.py — highest score wins per vertex,
     min_vertices unclaims) in two variants:
       selected_only       only the nominated masks exist
       selected_plus_rest  nominated masks outrank everything; all other
                           masks follow at their raw scores under the
                           frozen min_score
  3. each variant becomes a real SegmentationOutput bundle and runs the
     REAL downstream: exact evaluator -> derived C1 bundle (oracle labels
     injected) -> graph builder -> Router,
  4. answers are scored against the HUMAN-verified key (and the source
     bundle's delivered composition + variant A are scored identically as
     references).

This is still an ORACLE-GUIDED DIAGNOSTIC — the selection step reads oracle
membership, which no deployable rule may do. Its output is the maximum
human-QA gain any selection-repair rule could achieve on these saved masks.
If that gain is small, the deployable M2C rule is not worth designing.
Selection may nominate masks below the frozen min_score (viability was
always defined score-free); this is recorded per selected mask.
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
from demo.replica_habitat_import import import_habitat_room
from demo.replica_mesh_import import _parse_semantic_ply
from graph.builder import build_graph
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer
from segmenter.base import (
    SegmentationOutput, load_segmentation_output, save_segmentation_output,
)
from segmenter.derived import build_c1_eval_bundle
from segmenter.mask_resolve import MaskResolveConfig, resolve_masks
from tools.c1_exact_eval import (
    STRUCTURAL_OR_DROPPED, evaluate, oracle_vertex_membership,
)
from tools.c1_resolve_sweep import load_raw_masks

FROZEN_MIN_SCORE = 0.2
FROZEN_MIN_VERTICES = 20
_OUTCOME_MAP = {"bindings": "answer", "empty": "empty", "abstain": "defer"}


def select_joint_masks(masks: np.ndarray, oracle: np.ndarray,
                       entity_oids: list[int]) -> dict[int, dict]:
    """Oracle-guided nomination with collision fallback.

    Returns {oracle_id: {mask, iou, rank}} for every entity that secured a
    viable (IoU >= 0.5) mask not already taken by a higher-IoU entity.
    """
    n_oids = int(oracle.max()) + 1
    oracle_sizes = np.bincount(oracle[oracle >= 0], minlength=n_oids)
    per_entity: dict[int, list[tuple[float, int]]] = {}
    for o in entity_oids:
        per_entity[o] = []
    for k in range(masks.shape[0]):
        m_size = int(masks[k].sum())
        if m_size == 0:
            continue
        ids = oracle[masks[k]]
        counts = np.bincount(ids[ids >= 0], minlength=n_oids)
        for o in entity_oids:
            inter = int(counts[o])
            if inter == 0:
                continue
            iou = inter / (m_size + int(oracle_sizes[o]) - inter)
            if iou >= 0.5:
                per_entity[o].append((iou, k))
    for o in per_entity:
        per_entity[o].sort(reverse=True)

    taken: set[int] = set()
    selected: dict[int, dict] = {}
    # entities in order of their best achievable IoU, so collisions cost
    # the entity with the weaker alternative
    order = sorted(per_entity, key=lambda o: -(per_entity[o][0][0]
                                               if per_entity[o] else 0.0))
    for o in order:
        for rank, (iou, k) in enumerate(per_entity[o]):
            if k not in taken:
                taken.add(k)
                selected[o] = {"mask": k, "iou": round(iou, 4), "rank": rank}
                break
    return selected


def materialize(seg_src, masks, scores, selected_ks: list[int],
                variant: str, out_dir: Path,
                min_vertices: int = FROZEN_MIN_VERTICES) -> Path:
    """Run the frozen resolver on the variant's score arrangement and save a
    real SegmentationOutput bundle. Instance ids stay original mask rows."""
    scores = np.asarray(scores, dtype=np.float64)
    sel = np.zeros(len(scores), dtype=bool)
    sel[selected_ks] = True
    if variant == "selected_only":
        s2 = np.where(sel, scores + 2.0, -1.0)
        cfg = MaskResolveConfig(min_score=0.0, min_vertices=min_vertices)
    elif variant == "selected_plus_rest":
        s2 = scores.copy()
        s2[sel] += float(scores.max()) + 2.0
        cfg = MaskResolveConfig(min_score=FROZEN_MIN_SCORE,
                                min_vertices=min_vertices)
    else:
        raise ValueError(variant)
    ids = resolve_masks(masks, s2, cfg)
    out = SegmentationOutput(
        input_mesh_sha256=seg_src.input_mesh_sha256,
        n_vertices=seg_src.n_vertices,
        segmenter_name=f"joint_ceiling_{variant}",
        segmenter_version="0.0-oracle-guided-diagnostic",
        config_params_json=json.dumps({
            "variant": variant,
            "source_bundle_output_sha256": seg_src.output_sha256,
            "resolver": cfg.params(),
            "selected_mask_rows": sorted(int(k) for k in selected_ks),
            "provenance": "ORACLE-GUIDED selection; diagnostic ceiling only",
        }, sort_keys=True),
        vertex_instance_ids=ids,
    ).finalize()
    bundle_dir = out_dir / f"bundle_{variant}"
    save_segmentation_output(out, bundle_dir)
    return bundle_dir


def score_against_key(key: dict, graph_bundle, router, ctx,
                      uid_map: dict[str, str] | None) -> dict:
    """Human-key QA scoring; uid_map translates pred-space uids to oracle
    space (None for reference runs already in oracle space)."""
    per_q, notes = {}, []
    tp = n_cited = n_expected = 0
    rollup: dict[str, list[int]] = {}
    for q in key["questions"]:
        ans = router.answer(q["question"], graph_bundle, ctx)
        cited = set(ans.cited_uids)
        if uid_map is not None:
            # Pred-space uids are obj_<mask_row>, which COLLIDES with the
            # oracle obj_<id> namespace — an untranslated (unmatched) pred
            # uid must never accidentally equal a key uid, so it is
            # prefixed instead of passed through.
            cited = {uid_map[u] if u in uid_map else f"pred:{u}"
                     for u in cited}
        must = set(q["expected_must_contain"])
        must_not = set(q["expected_must_not_contain"])
        got_outcome = _OUTCOME_MAP.get(ans.outcome, "unknown")
        row = {
            "expected_outcome": q["expected_outcome"],
            "actual_outcome": got_outcome,
            "n_cited": len(cited),
            "n_expected": len(must),
            "n_hit": len(cited & must),
            "must_not_violations": sorted(cited & must_not),
            "anonymous_cited": sorted(u for u in cited
                                      if not u.startswith("obj_")
                                      or u.startswith("pred:")),
        }
        if q["expected_outcome"] == "answer" and q.get("exhaustive"):
            row["precision"] = (len(cited & must) / len(cited)) if cited else None
            row["recall"] = (len(cited & must) / len(must)) if must else None
            tp += len(cited & must)
            n_cited += len(cited)
            n_expected += len(must)
            rel = q.get("relation", "?")
            rollup.setdefault(rel, [0, 0, 0])
            rollup[rel][0] += len(cited & must)
            rollup[rel][1] += len(cited)
            rollup[rel][2] += len(must)
        per_q[q["question_id"]] = row
    per_relation = {rel: {"precision": (h / c) if c else None,
                          "recall": (h / e) if e else None,
                          "n_hit": h, "n_cited": c, "n_expected": e}
                    for rel, (h, c, e) in sorted(rollup.items())}
    return {
        "micro_precision": (tp / n_cited) if n_cited else None,
        "micro_recall": (tp / n_expected) if n_expected else None,
        "n_expected_total": n_expected,
        "per_relation": per_relation,
        "per_question": per_q,
    }


def _qa_for_arts(arts, key, router, ctx, uid_map):
    bundle, _ = build_graph(arts, _runs(), density_policy="phase2_telemetry_only")
    qa = score_against_key(key, bundle, router, ctx, uid_map) if key else None
    return qa, len(bundle.edges), len(arts.entities)


def measure_joint(room_dir: Path, src_bundle: Path, scene_id: str,
                  key_path: Path | None, out_dir: Path,
                  min_vertices: int = FROZEN_MIN_VERTICES) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    key = json.loads(key_path.read_text()) if key_path else None
    if key is not None and key.get("answer_key_type") != "human_verified":
        raise ValueError(f"{key_path} is not a human_verified key")

    seg_src = load_segmentation_output(src_bundle)
    masks, scores = load_raw_masks(src_bundle)
    src_eval = evaluate(room_dir, src_bundle)
    _, vidx, oid = _parse_semantic_ply(room_dir / "habitat" / "mesh_semantic.ply")
    oracle = oracle_vertex_membership(vidx, oid, seg_src.n_vertices)

    entity_oids = [int(o) for o, cov in src_eval["oracle_coverage"].items()
                   if cov["class"] and cov["class"] not in STRUCTURAL_OR_DROPPED]
    selected = select_joint_masks(masks, oracle, entity_oids)
    sel_rows = sorted(v["mask"] for v in selected.values())
    n_fallback = sum(1 for v in selected.values() if v["rank"] > 0)
    n_below_frozen = sum(1 for v in selected.values()
                         if scores[v["mask"]] < FROZEN_MIN_SCORE)

    router = Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                    verbalizer=StandardVerbalizer())
    ctx = ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))

    rows: dict[str, dict] = {}

    def run_c1(name: str, bundle_dir: Path):
        ev = evaluate(room_dir, bundle_dir)
        arts, _inj = build_c1_eval_bundle(room_dir, bundle_dir, scene_id,
                                          min_vertices=min_vertices)
        uid_map = {f"obj_{m['pred_id']}": f"obj_{m['oracle_id']}"
                   for m in ev["matches"]}
        qa, n_edges, n_entities = _qa_for_arts(arts, key, router, ctx, uid_map)
        n5 = ev["n_entity_matches_at_iou"]["0.5"]
        rows[name] = {
            "entity_matches_at_05": f"{n5}/{ev['n_oracle_entity_instances']}",
            "entity_recall_at_05": round(n5 / ev["n_oracle_entity_instances"], 4),
            "support_owner": ev["support_owner"],
            "n_pred_instances": ev["n_pred_instances"],
            "n_graph_edges": n_edges,
            "n_entities": n_entities,
            "qa_vs_human_key": qa,
        }

    # reference: variant A (oracle boxes) through the identical QA scorer
    A = import_habitat_room(room_dir, scene_id)
    qa_a, edges_a, ents_a = _qa_for_arts(A, key, router, ctx, None)
    rows["A_oracle_boxes"] = {
        "n_graph_edges": edges_a, "n_entities": ents_a, "qa_vs_human_key": qa_a,
    }
    run_c1("delivered", src_bundle)
    for variant in ("selected_only", "selected_plus_rest"):
        run_c1(f"joint_{variant}",
               materialize(seg_src, masks, scores, sel_rows, variant, out_dir,
                           min_vertices=min_vertices))

    report = {
        "schema": "c1_joint_ceiling_v1",
        "purpose": ("oracle-guided JOINT selection ceiling through the real "
                    "resolver/graph/Router, scored against the human key; "
                    "diagnostic only, NOT a deployable result"),
        "scene_id": scene_id,
        "source_bundle": str(src_bundle),
        "human_key": str(key_path) if key_path else None,
        "selection": {
            "n_entities_considered": len(entity_oids),
            "n_selected": len(selected),
            "n_collision_fallbacks": n_fallback,
            "n_selected_below_frozen_min_score": n_below_frozen,
            "per_entity": {str(o): selected[o] for o in sorted(selected)},
        },
        "rows": rows,
    }
    out = out_dir / f"{scene_id}_joint_ceiling.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--room-dir", required=True, type=Path)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--key", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "runs" / "phase8_c1" / "joint_ceiling")
    args = ap.parse_args(argv)

    r = measure_joint(args.room_dir, args.bundle, args.scene_id,
                      args.key, args.out_dir)
    s = r["selection"]
    print(f"selected {s['n_selected']}/{s['n_entities_considered']} entities "
          f"(fallbacks={s['n_collision_fallbacks']}, "
          f"below-frozen-score={s['n_selected_below_frozen_min_score']})")
    hdr = f"{'row':>22}  {'ent@0.5':>8}  {'QA-P':>6}  {'QA-R':>6}  {'edges':>6}"
    print(hdr)
    for name, row in r["rows"].items():
        qa = row.get("qa_vs_human_key") or {}
        p, rec = qa.get("micro_precision"), qa.get("micro_recall")
        print(f"{name:>22}  {row.get('entity_matches_at_05', '-'):>8}  "
              f"{'-' if p is None else f'{p:.2f}':>6}  "
              f"{'-' if rec is None else f'{rec:.2f}':>6}  "
              f"{row['n_graph_edges']:>6}")
    print(f"report -> {args.out_dir / (r['scene_id'] + '_joint_ceiling.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

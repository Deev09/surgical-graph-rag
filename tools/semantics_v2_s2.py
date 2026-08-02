"""semantics_v2 S2 — variant A first: the new representation ceiling.

  python3 tools/semantics_v2_s2.py

Protocol: docs/semantics_v2_track_protocol.md (signed off; gates
FROZEN). Runs variant A (oracle boxes) under the v2 semantics
(`demo.semantics_v2.runs_v2` + `make_v2_compiler`) on all four keyed
scenes, scores against the UNCHANGED human keys, and evaluates the
frozen proceed gates. Also recomputes each scene's frozen v1 A row and
hard-asserts it matches the committed anchors — proving the v1 track is
untouched at measurement time.

Every emitted table carries the track label: semantics_v2 is a
BENCHMARK-DEFINITION CHANGE, never comparable to the frozen track.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.question_battery import _runs as runs_v1
from demo.replica_habitat_import import import_habitat_room
from demo.semantics_v2 import make_v2_compiler, runs_v2
from graph.builder import build_graph
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer
from tools.c1_joint_ceiling import score_against_key

SCENES = ("replica_room_0", "replica_room_1", "replica_room_2",
          "replica_office_0")
# frozen v1 A anchors (runs/mvp_v0/aggregate.json, committed reports)
V1_ANCHORS = {
    "replica_room_0": (0.85, 0.3469),
    "replica_room_1": (0.8333, 0.2857),
    "replica_room_2": (0.9524, 0.4082),
    "replica_office_0": (1.0, 0.375),
}
GATES = {
    "attached_hits_min": 8,      # room_2, of 14
    "support_hits_min": 8,       # room_2, of 20
    "relation_precision_min": 0.85,
    "aggregate_recall_min": 0.55,
    "aggregate_precision_min": 0.85,
    "all_scenes_precision_floor": 0.80,
}
TRACK_LABEL = ("semantics_v2 track — benchmark-definition change; NOT "
               "comparable to the frozen track")


def scene_row(scene_id: str, runs, router, ctx) -> dict:
    manifest = {s["scene_id"]: Path(s["room_dir"]) for s in json.loads(
        (REPO_ROOT / "eval" / "questions" / "phase8" /
         "scene_manifest.json").read_text())["scenes"]}
    key = json.loads((REPO_ROOT / "eval" / "questions" / "phase8" /
                      f"{scene_id}_qa.json").read_text())
    arts = import_habitat_room(manifest[scene_id], scene_id)
    bundle, _ = build_graph(arts, runs, density_policy="phase2_telemetry_only")
    qa = score_against_key(key, bundle, router, ctx, None)
    viol = sum(len(q["must_not_violations"])
               for q in qa["per_question"].values())
    return {
        "micro_precision": (None if qa["micro_precision"] is None
                            else round(qa["micro_precision"], 4)),
        "micro_recall": (None if qa["micro_recall"] is None
                         else round(qa["micro_recall"], 4)),
        "must_not_violations": viol,
        "n_graph_edges": len(bundle.edges),
        "per_relation": {
            rel: {"n_hit": v["n_hit"], "n_cited": v["n_cited"],
                  "n_expected": v["n_expected"],
                  "precision": (round(v["n_hit"] / v["n_cited"], 4)
                                if v["n_cited"] else None)}
            for rel, v in qa["per_relation"].items()},
    }


def main() -> int:
    out_dir = REPO_ROOT / "runs" / "semantics_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    executor, verbalizer = RulesExecutor(), StandardVerbalizer()
    router_v1 = Router(compiler=RulesCompiler(), executor=executor,
                       verbalizer=verbalizer)
    router_v2 = Router(compiler=make_v2_compiler(), executor=executor,
                       verbalizer=verbalizer)
    ctx = ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))

    rows = {}
    for sid in SCENES:
        v1 = scene_row(sid, runs_v1(), router_v1, ctx)
        ap, ar = V1_ANCHORS[sid]
        if v1["micro_precision"] != ap or v1["micro_recall"] != ar:
            raise AssertionError(
                f"{sid}: frozen v1 A row drifted: "
                f"{v1['micro_precision']}/{v1['micro_recall']} vs {ap}/{ar}")
        v2 = scene_row(sid, runs_v2(), router_v2, ctx)
        rows[sid] = {"A_v1_frozen": v1, "A_v2": v2}

    r2 = rows["replica_room_2"]["A_v2"]
    att = r2["per_relation"].get("ATTACHED_TO", {})
    sup = r2["per_relation"].get("ON_ENTITY_SURFACE", {})
    g = GATES
    gates = {
        "attached_hits": (att.get("n_hit", 0) >= g["attached_hits_min"]),
        "attached_precision": ((att.get("precision") or 0)
                               >= g["relation_precision_min"]),
        "support_hits": (sup.get("n_hit", 0) >= g["support_hits_min"]),
        "support_precision": ((sup.get("precision") or 0)
                              >= g["relation_precision_min"]),
        "aggregate_recall": ((r2["micro_recall"] or 0)
                             >= g["aggregate_recall_min"]),
        "aggregate_precision": ((r2["micro_precision"] or 0)
                                >= g["aggregate_precision_min"]),
        "all_scenes_precision_floor": all(
            (rows[s]["A_v2"]["micro_precision"] or 0)
            >= g["all_scenes_precision_floor"] for s in SCENES),
    }
    report = {
        "schema": "semantics_v2_s2_v1",
        "protocol": "docs/semantics_v2_track_protocol.md",
        "track_label": TRACK_LABEL,
        "rows": rows,
        "gate_thresholds": g,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "decision": ("PROCEED_TO_S3" if all(gates.values())
                     else "STOP_TRACK"),
    }
    out = out_dir / "s2_report.json"
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")

    print(TRACK_LABEL)
    for sid in SCENES:
        a, b = rows[sid]["A_v1_frozen"], rows[sid]["A_v2"]
        print(f"{sid:<20} A-v1 P={a['micro_precision']} R={a['micro_recall']}"
              f"  ->  A-v2 P={b['micro_precision']} R={b['micro_recall']} "
              f"(viol={b['must_not_violations']}, edges={b['n_graph_edges']})")
    print(f"room_2 attached: {att.get('n_hit')}/{att.get('n_expected')} "
          f"cited={att.get('n_cited')} P={att.get('precision')}")
    print(f"room_2 support:  {sup.get('n_hit')}/{sup.get('n_expected')} "
          f"cited={sup.get('n_cited')} P={sup.get('precision')}")
    for k in sorted(gates):
        print(f"  {k}: {'PASS' if gates[k] else 'FAIL'}")
    print(f"DECISION: {report['decision']}")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

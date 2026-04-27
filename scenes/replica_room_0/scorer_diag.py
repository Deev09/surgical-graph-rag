"""Diagnostic-only inspection of score_node behavior on Replica spatial queries.

For each query, list every node that has a matching spatial-relation edge to
the anchor, with the score breakdown and geometric facts. No tuning, no
behavior changes — just print what the existing scorer is doing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from tiny_graph_demo import _label_equiv, parse_query, scene_nodes_from_record
from scoring.geom import _find_anchor_node, _get_xyz
from scoring.spatial import (
    _best_spatial_relation_weight,
    _geometric_spatial_bonus,
    _spatial_xy_salience,
)
from parsers.vocab import rebuild_label_vocab_from_scene


REPO = Path(__file__).resolve().parents[2]
SCENE_PATH = REPO / "scenes" / "replica_room_0" / "computed_relations" / "scene_graph.json"
EXPECTED_PATH = REPO / "scenes" / "replica_room_0" / "expected_answers.json"

QUERIES = [
    "What is to the right of table_4?",
    "What is to the right of the door?",
    "What is behind the rug?",
    "What is in front of sofa_1?",
]


def main() -> None:
    record = json.loads(SCENE_PATH.read_text())
    nodes = scene_nodes_from_record(record)
    rebuild_label_vocab_from_scene(record)
    expected_map = json.loads(EXPECTED_PATH.read_text())

    for query in QUERIES:
        parsed = parse_query(query)
        rel_type = parsed.spatial_relation
        anchor = parsed.relation_anchor
        expected = expected_map.get(query, [])
        anchor_node = _find_anchor_node(nodes, anchor) if anchor else None
        ax, ay, az = _get_xyz(anchor_node) if anchor_node else (0.0, 0.0, 0.0)

        print()
        print("=" * 130)
        print(f"QUERY: {query}")
        print(f"  parsed: intent={parsed.intent}  spatial_relation={rel_type}  anchor={anchor}")
        print(f"  anchor node xyz: ({ax:.2f}, {ay:.2f}, {az:.2f})")
        print(f"  expected: {expected}")
        print("-" * 130)

        rows = []
        for node in nodes:
            if anchor and _label_equiv(str(node["label"]), anchor):
                continue
            w = _best_spatial_relation_weight(node.get("relations", []), rel_type, anchor)
            if w <= 0:
                continue
            geo = _geometric_spatial_bonus(node, rel_type, anchor_node) if anchor_node else 0.0
            salience = _spatial_xy_salience(node, rel_type, anchor_node) if anchor_node else 1.0
            score = (w + geo) * salience

            cx, cy, cz = _get_xyz(node)
            dx, dy, dz = cx - ax, cy - ay, cz - az
            euclid = math.sqrt(dx * dx + dy * dy + dz * dz)
            attrs = node.get("attributes", {}) or {}
            otype = str(attrs.get("type", ""))
            mounted = str(attrs.get("mounted", ""))
            in_exp = "yes" if str(node["label"]) in expected else " no"
            far_but_valid = "yes" if euclid > 2.5 else " no"

            rows.append(
                {
                    "label": str(node["label"]),
                    "score": round(score, 4),
                    "w": round(w, 3),
                    "geo": round(geo, 4),
                    "sal": round(salience, 3),
                    "dx": round(dx, 2),
                    "dy": round(dy, 2),
                    "dz": round(dz, 2),
                    "euclid": round(euclid, 2),
                    "type": otype,
                    "mounted": mounted,
                    "in_exp": in_exp,
                    "far_valid": far_but_valid,
                }
            )

        rows.sort(key=lambda r: (-r["score"], r["label"]))

        print(
            f"{'rank':<5}{'label':<24}{'score':>8}{'w':>6}{'geo':>8}{'sal':>6}"
            f"{'dx':>7}{'dy':>7}{'dz':>7}{'eucl':>7}{'expected':>9}{'far':>5}  type/mounted"
        )
        for i, r in enumerate(rows, 1):
            tm = f"{r['type']}/{r['mounted']}".strip("/")
            print(
                f"{i:<5}{r['label']:<24}{r['score']:>8.4f}{r['w']:>6.2f}{r['geo']:>8.4f}{r['sal']:>6.2f}"
                f"{r['dx']:>7.2f}{r['dy']:>7.2f}{r['dz']:>7.2f}{r['euclid']:>7.2f}"
                f"{r['in_exp']:>9}{r['far_valid']:>5}  {tm}"
            )

        n_total = len(rows)
        n_expected_in_top6 = sum(1 for r in rows[:6] if r["in_exp"] == "yes")
        n_expected_total = sum(1 for r in rows if r["in_exp"] == "yes")
        unique_scores = sorted({r["score"] for r in rows}, reverse=True)
        print(
            f"\nsummary: {n_total} candidates, "
            f"{n_expected_in_top6}/{n_expected_total} expected in top-6, "
            f"{len(unique_scores)} unique score values out of {n_total}"
        )
        print(f"  unique score values: {unique_scores[:5]}{'...' if len(unique_scores) > 5 else ''}")


if __name__ == "__main__":
    main()

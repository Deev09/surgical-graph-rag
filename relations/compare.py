"""A/B authored vs computed relations.

Per relation type: TP / FP / FN against an authored set.
Edge equality is (source, type, target). Weights are ignored.
ATTACHED_TO is reported separately as `not_emitted_by_compute`.

When the input scene has no authored relations, the TP/FP/FN section is
omitted and only the computed-edge inventory is reported.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from relations.compute import (
    DIRECTIONAL_TYPES,
    SKIPPED_TYPES,
    SYMMETRIC_TYPES,
    compute_relations,
)

REPO = Path(__file__).resolve().parent.parent


def _key(edge: dict) -> tuple[str, str, str]:
    return (edge["source"], edge["type"], edge["target"])


def run(scene_path: Path, out_dir: Path) -> None:
    scene = json.loads(scene_path.read_text())
    objects = scene["objects"]
    authored = scene.get("relations", []) or []
    computed = compute_relations(objects)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scene_graph.json").write_text(
        json.dumps({"scene": scene["scene"], "objects": objects, "relations": computed}, indent=2)
    )

    authored_set = {_key(e) for e in authored}
    computed_set = {_key(e) for e in computed}

    by_type: dict[str, dict[str, list]] = defaultdict(
        lambda: {"true_positives": [], "false_positives": [], "false_negatives": []}
    )
    all_types = set(DIRECTIONAL_TYPES) | set(SYMMETRIC_TYPES) | set(SKIPPED_TYPES)
    all_types |= {k[1] for k in authored_set} | {k[1] for k in computed_set}

    for k in authored_set & computed_set:
        by_type[k[1]]["true_positives"].append(list(k))
    for k in computed_set - authored_set:
        by_type[k[1]]["false_positives"].append(list(k))
    for k in authored_set - computed_set:
        by_type[k[1]]["false_negatives"].append(list(k))

    summary = {}
    for t in sorted(all_types):
        bucket = by_type[t]
        tp = len(bucket["true_positives"])
        fp = len(bucket["false_positives"])
        fn = len(bucket["false_negatives"])
        summary[t] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision_vs_authored": tp / (tp + fp) if (tp + fp) else None,
            "recall_vs_authored": tp / (tp + fn) if (tp + fn) else None,
        }

    diff = {
        "scene": scene["scene"],
        "thresholds": {"min_delta": 0.3, "near_threshold": 1.0},
        "totals": {
            "authored": len(authored_set),
            "computed": len(computed_set),
            "intersection": len(authored_set & computed_set),
        },
        "skipped_by_compute": list(SKIPPED_TYPES),
        "has_authored_relations": bool(authored_set),
        "per_type": summary if authored_set else None,
        "per_type_edges": dict(by_type) if authored_set else None,
        "computed_edge_counts": {
            t: sum(1 for e in computed if e["type"] == t) for t in sorted({e["type"] for e in computed})
        },
    }

    (out_dir / "relation_diff.json").write_text(json.dumps(diff, indent=2))
    headline = {
        "totals": diff["totals"],
        "computed_edge_counts": diff["computed_edge_counts"],
    }
    if authored_set:
        headline["per_type_summary"] = summary
    print(json.dumps(headline, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene-graph", type=Path, default=REPO / "baselines" / "v1" / "scene_graph.json")
    p.add_argument("--out-dir", type=Path, default=REPO / "baselines" / "v1_computed_relations")
    args = p.parse_args()
    run(args.scene_graph, args.out_dir)


if __name__ == "__main__":
    main()

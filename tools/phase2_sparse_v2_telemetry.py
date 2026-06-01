"""Phase 2 P2.08 — sparse_v2 NEAR telemetry on Replica room_0.

Runs both sparse v1 (centroid, byte-frozen) and sparse v2 (aabb_surface)
on the same Replica enriched-v2 EntityArtifacts bundle and emits:

  scenes/replica_room_0/eval/phase2_sparse_v2_telemetry.json

The artifact records edge counts, density ratios, and per-pair deltas so
the human reviewer can decide whether to (a) loosen the sparse-density
guardrail for v2, (b) retune sparse_near_threshold, or (c) accept the
current behavior. **The Phase 1 density guardrail is NOT touched by this
tool** — it only reports.

Skip-on-missing: if the enriched-v2 importer output isn't on disk
(scenes/replica_room_0/enriched/v2/scene_graph.json), this script exits
with a clear message rather than fabricating a synthetic comparison.

Run: python tools/phase2_sparse_v2_telemetry.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.base import ReconstructionConfig
from adapters.oracle_replica import OracleReplicaAdapter, build_replica_capture_bundle
from extractors.base import InstanceExtractorConfig
from extractors.oracle_replica import OracleReplicaExtractor
from graph.builder import SPARSE_DENSITY_LIMIT
from graph.relations.directional import DirectionalConfig, DirectionalExtractor
from graph.relations.proximity import (
    SPARSE_V2_VERSION,
    ProximityConfig,
    ProximityExtractor,
)
from representations.mesh import MeshRepresentation


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
REPLICA_V2_DIR = REPLICA_SCENE_DIR / "enriched" / "v2"
ARTIFACT_PATH = REPLICA_SCENE_DIR / "eval" / "phase2_sparse_v2_telemetry.json"

DEFAULT_THRESHOLD = 1.0


def _pair_key(edge) -> tuple[str, str]:
    return tuple(sorted([edge.source.uid, edge.target.uid]))  # type: ignore[return-value]


def main() -> int:
    if not (REPLICA_V2_DIR / "scene_graph.json").exists():
        print(
            "Refusing to emit telemetry: enriched-v2 importer output is "
            f"missing at {REPLICA_V2_DIR.relative_to(REPO_ROOT)}/scene_graph.json. "
            "Re-run the importer with --enriched-v2 first."
        )
        return 1

    capture = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    repr_bundle = OracleReplicaAdapter().reconstruct(
        capture, ReconstructionConfig(name="oracle_replica", version="0.1", params={}),
    )
    representation = MeshRepresentation(bundle=repr_bundle)
    artifacts = OracleReplicaExtractor(enriched_v2_path=REPLICA_V2_DIR).extract(
        representation, InstanceExtractorConfig(
            name="oracle_replica", version="0.1", params={},
        ),
    )
    entity_count = len(artifacts.entities)

    v1_edges, v1_diag = ProximityExtractor().extract(
        artifacts, ProximityConfig(
            mode="sparse", sparse_near_threshold=DEFAULT_THRESHOLD,
        ),
    )
    v2_edges, v2_diag = ProximityExtractor().extract(
        artifacts, ProximityConfig(
            mode="sparse", sparse_version=2,
            sparse_near_threshold=DEFAULT_THRESHOLD,
        ),
    )
    _directional_edges, directional_diag = DirectionalExtractor().extract(
        artifacts, DirectionalConfig(mode="sparse"),
    )

    v1_pairs = {_pair_key(e): e.evidence.get("distance_m") for e in v1_edges}
    v2_pairs = {_pair_key(e): e.evidence.get("distance_m") for e in v2_edges}

    new_in_v2 = sorted(set(v2_pairs) - set(v1_pairs))
    only_in_v1 = sorted(set(v1_pairs) - set(v2_pairs))
    in_both = sorted(set(v1_pairs) & set(v2_pairs))

    v1_density = v1_diag.logical_edges_total / max(1, entity_count)
    v2_density = v2_diag.logical_edges_total / max(1, entity_count)
    combined_v1_logical = (
        directional_diag.logical_edges_total + v1_diag.logical_edges_total
    )
    combined_v2_logical = (
        directional_diag.logical_edges_total + v2_diag.logical_edges_total
    )
    combined_v1_density = combined_v1_logical / max(1, entity_count)
    combined_v2_density = combined_v2_logical / max(1, entity_count)

    per_pair_distances = sorted([
        {
            "pair": list(p),
            "v1_centroid_distance_m": v1_pairs.get(p),
            "v2_surface_distance_m": v2_pairs.get(p),
            "membership": (
                "both" if p in in_both else (
                    "v2_only" if p in v2_pairs else "v1_only"
                )
            ),
        }
        for p in set(v1_pairs) | set(v2_pairs)
    ], key=lambda d: (d["pair"][0], d["pair"][1]))

    payload = {
        "gate": "phase2_sparse_v2_telemetry",
        "scene_id": artifacts.scene_id,
        "entity_count": entity_count,
        "config": {
            "sparse_near_threshold": DEFAULT_THRESHOLD,
            "v1_distance_metric": "centroid",
            "v1_extractor_version": ProximityExtractor.version,
            "v2_distance_metric": "aabb_surface",
            "v2_extractor_version": SPARSE_V2_VERSION,
            "comment": (
                "Threshold preserved from Phase 1 default. No retuning. "
                "v2 surface distance ≤ v1 centroid distance per A2."
            ),
        },
        "v1": {
            "physical_edges": v1_diag.physical_edges_total,
            "logical_edges": v1_diag.logical_edges_total,
            "density_ratio_per_entity": v1_density,
        },
        "v2": {
            "physical_edges": v2_diag.physical_edges_total,
            "logical_edges": v2_diag.logical_edges_total,
            "density_ratio_per_entity": v2_density,
            "exceeds_phase1_sparse_density_limit": v2_density > SPARSE_DENSITY_LIMIT,
            "phase1_sparse_density_limit": SPARSE_DENSITY_LIMIT,
        },
        "directional_sparse": {
            "physical_edges": directional_diag.physical_edges_total,
            "logical_edges": directional_diag.logical_edges_total,
            "density_ratio_per_entity": (
                directional_diag.logical_edges_total / max(1, entity_count)
            ),
        },
        "combined_sparse_graph_before_near_surface": {
            "v1_logical_edges": combined_v1_logical,
            "v1_density_ratio_per_entity": combined_v1_density,
            "v2_logical_edges": combined_v2_logical,
            "v2_density_ratio_per_entity": combined_v2_density,
            "v2_exceeds_phase1_sparse_density_limit": (
                combined_v2_density > SPARSE_DENSITY_LIMIT
            ),
            "phase1_sparse_density_limit": SPARSE_DENSITY_LIMIT,
        },
        "delta": {
            "edges_in_both": len(in_both),
            "edges_new_in_v2": len(new_in_v2),
            "edges_only_in_v1": len(only_in_v1),
            "edges_new_in_v2_sample": [list(p) for p in new_in_v2[:16]],
            "edges_only_in_v1_sample": [list(p) for p in only_in_v1[:16]],
        },
        "notes": [
            "Per Q6: density recorded as telemetry only. Phase 1 sparse-density "
            "guardrail (<= 14) is NOT touched by this run.",
            "Per A2: surface distance is bounded above by centroid distance, so "
            "v2 NEAR is a superset of v1 NEAR at the same threshold.",
            "The builder guardrail applies to the combined graph, not only the "
            "proximity family. The combined pre-NEAR_SURFACE ratio is reported "
            "separately.",
            "If combined v2 exceeds the Phase 1 density guardrail, decide "
            "explicitly whether to (a) revise the v2 integration policy, "
            "(b) retune the threshold, or (c) accept the denser graph. Do not "
            "silently relax either.",
        ],
        "per_pair_distances": per_pair_distances,
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    rel = ARTIFACT_PATH.relative_to(REPO_ROOT)
    print(f"Wrote {rel}")
    print(f"  entities:        {entity_count}")
    print(f"  v1 logical edges:  {v1_diag.logical_edges_total}  "
          f"(density {v1_density:.3f}/entity)")
    print(f"  v2 logical edges:  {v2_diag.logical_edges_total}  "
          f"(density {v2_density:.3f}/entity)")
    print(f"  pairs new in v2: {len(new_in_v2)}")
    print(f"  pairs only in v1: {len(only_in_v1)}")
    print(f"  pairs in both:    {len(in_both)}")
    print(f"  directional sparse logical edges: {directional_diag.logical_edges_total}")
    print(f"  combined v1 density: {combined_v1_density:.3f}/entity")
    print(f"  combined v2 density: {combined_v2_density:.3f}/entity")
    if combined_v2_density > SPARSE_DENSITY_LIMIT:
        print(
            f"  WARNING: combined v2 density ({combined_v2_density:.3f}) "
            f"exceeds Phase 1 guardrail ({SPARSE_DENSITY_LIMIT}). "
            "Telemetry-only; no auto-loosening."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

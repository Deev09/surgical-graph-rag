"""Phase 6 P6.05 — Router QA eval runner on Replica room_0.

Builds one explicit MIXED eval-candidate graph (directional + NEAR_SURFACE +
ON_SURFACE + CONTACTS_SURFACE + ON_ENTITY_SURFACE), runs the frozen Phase 6
mixed question set through the real Router, and writes:

  scenes/replica_room_0/eval/phase6_router_qa_eval.json

This is a reasoner-native eval track. It is NOT comparable to the v1
benchmark, and it is NOT directly comparable to the P5 6/6 scorecard because
the question set changed (Q3 defer->answer, Q7 new empty row).
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
from eval.router_qa import score_questions
from extractors.base import InstanceExtractorConfig
from extractors.oracle_replica import OracleReplicaExtractor
from graph.builder import ExtractorRun, build_graph
from graph.relations.contacts_surface import ContactsSurfaceConfig, ContactsSurfaceExtractor
from graph.relations.directional import DirectionalConfig, DirectionalExtractor
from graph.relations.on_entity_surface import (
    OnEntitySurfaceConfig,
    OnEntitySurfaceExtractor,
)
from graph.relations.on_surface import OnSurfaceConfig, OnSurfaceExtractor
from graph.relations.surface import SurfaceProximityConfig, SurfaceProximityExtractor
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import Phase6RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer
from representations.mesh import MeshRepresentation


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
REPLICA_V2_DIR = REPLICA_SCENE_DIR / "enriched" / "v2"
QUESTIONS_PATH = REPO_ROOT / "eval" / "questions" / "phase6_mixed_qa.json"
ARTIFACT_PATH = REPLICA_SCENE_DIR / "eval" / "phase6_router_qa_eval.json"


def _build_replica_artifacts():
    capture = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    representation = MeshRepresentation(
        bundle=OracleReplicaAdapter().reconstruct(
            capture, ReconstructionConfig(name="oracle_replica", version="0.1", params={}),
        ),
    )
    return OracleReplicaExtractor(enriched_v2_path=REPLICA_V2_DIR).extract(
        representation,
        InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )


def _eval_runs() -> list[ExtractorRun]:
    """The Phase 6 mixed eval-candidate graph; not a default builder path."""
    return [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse")),
        ExtractorRun(SurfaceProximityExtractor(), SurfaceProximityConfig(use_polygon_clip=True)),
        ExtractorRun(OnSurfaceExtractor(), OnSurfaceConfig()),
        ExtractorRun(ContactsSurfaceExtractor(), ContactsSurfaceConfig()),
        ExtractorRun(OnEntitySurfaceExtractor(), OnEntitySurfaceConfig()),
    ]


def _extractor_configs(diag) -> list[dict]:
    versions = dict(diag.extractor_versions)
    return [
        {"name": "directional", "version": versions.get("directional"),
         "params": {"mode": "sparse"}},
        {"name": "near_surface", "version": versions.get("near_surface"),
         "params": {"use_polygon_clip": True}},
        {"name": "on_surface", "version": versions.get("on_surface"),
         "params": {"contact_threshold_m": 0.02, "penetration_tolerance_m": 0.03}},
        {"name": "contacts_surface", "version": versions.get("contacts_surface"),
         "params": {"contact_threshold_m": 0.02, "penetration_tolerance_m": 0.02}},
        {"name": "on_entity_surface", "version": versions.get("on_entity_surface"),
         "params": {"contact_threshold_m": 0.02, "penetration_tolerance_m": 0.03}},
    ]


def main() -> int:
    if not (REPLICA_V2_DIR / "scene_graph.json").exists():
        print("Refusing: enriched-v2 importer output is missing.")
        return 1

    artifacts = _build_replica_artifacts()
    bundle, diag = build_graph(
        artifacts, _eval_runs(), density_policy="phase2_telemetry_only",
    )
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]

    router = Router(
        compiler=Phase6RulesCompiler(), executor=RulesExecutor(),
        verbalizer=StandardVerbalizer(),
    )
    ctx = ExecutionContext(
        completeness=CompletenessProfile(
            source="oracle", entity_recall_by_class={}, edge_recall_by_type={},
        ),
    )
    scorecard = score_questions(questions, bundle, router, ctx)

    entity_surface_edges = [e for e in bundle.edges if e.type == "ON_ENTITY_SURFACE"]
    table_answer = next(
        r for r in scorecard["per_question"] if r["question_id"] == "Q3"
    )
    chair_empty = next(
        r for r in scorecard["per_question"] if r["question_id"] == "Q7"
    )
    payload = {
        "scene_id": artifacts.scene_id,
        "phase": "P6.05",
        "artifact_kind": "router_qa_eval",
        "schema_version": 1,
        "graph": {
            "bundle_hash": bundle.bundle_hash,
            "edge_count": len(bundle.edges),
            "on_entity_surface_edge_count": len(entity_surface_edges),
            "note": "explicit mixed eval-candidate graph; NOT a default builder path",
            "extractor_configs": _extractor_configs(diag),
        },
        "eval": {
            "question_set": str(QUESTIONS_PATH.relative_to(REPO_ROOT)),
            **scorecard,
        },
        "phase6_eval_definition_change": {
            "q3": "defer -> answer because EntitySurface support exists",
            "q7": "new honest-empty row; answerable relation with zero bindings",
            "not_comparable_to_p5": True,
        },
        "summary": {
            "q3_table_cited_uids": table_answer["cited_uids"],
            "q7_chair_actual_outcome": chair_empty["actual_outcome"],
            "on_entity_surface_pairs": sorted(
                [e.source.uid, e.target.uid] for e in entity_surface_edges
            ),
        },
        "interpretation_limits": [
            "NOT comparable to the v1 benchmark; this is a separate "
            "reasoner-native eval track (scores Router answers/deferrals, "
            "not top-k retrieval).",
            "NOT directly comparable to the P5 6/6 scorecard; the question "
            "set changed (Q3 defer->answer, Q7 new empty row).",
            "single scene (Replica room_0); not generalization evidence.",
            "pot obj_43 is intentionally band-excluded at +0.0349 m float; "
            "support-furniture classes are excluded as supported tabletop answers.",
        ],
        "determinism": {"timestamp_free": True},
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    rel = ARTIFACT_PATH.relative_to(REPO_ROOT)
    agg = scorecard["aggregate"]
    print(f"Wrote {rel}")
    print(f"  bundle_hash:               {bundle.bundle_hash}")
    print(f"  questions:                 {agg['total_questions']}")
    print(f"  category_counts:           {agg['category_counts']}")
    print(f"  false_answer_count:        {agg['false_answer_count']}")
    print(f"  all_expected_outcomes_met: {agg['all_expected_outcomes_met']}")
    for r in scorecard["per_question"]:
        print(f"    {r['question_id']} {r['category']:13s} "
              f"({r['expected_outcome']} -> {r['actual_outcome']}) {r['cited_uids'][:5]}")
    return 0 if agg["all_expected_outcomes_met"] and agg["false_answer_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

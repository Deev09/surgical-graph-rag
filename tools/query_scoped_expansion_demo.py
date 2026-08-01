"""Query-scoped activation of provisional raw-mask proposals.

This is an evaluation-isolated prototype:

1. build the unchanged hard graph at ``accepted_score``;
2. compile one entity-support query and locate its accepted anchor surfaces;
3. use geometry only to select absent raw candidates (score-provisional or
   composition-lost) satisfying the existing rest-contact predicate on anchors;
4. rebuild and answer with hard + selected candidates;
5. compare with an indiscriminate inclusive control and two Replica reference
   importers.

Replica labels and surfaces are injected only after selection, through the
existing C1 evaluation bundle. They are used to interpret results, never to
choose provisional masks. Phase 8 has no promoted human-verified keys yet, so
the A/B reference comparison is diagnostic rather than an accuracy claim.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.question_battery import _runs
from demo.replica_habitat_import import (
    ROOM_0_Z_TRANSLATION,
    _aligned_structural_surfaces,
    _gravity_align_matrix,
    import_habitat_room,
)
from demo.replica_mesh_import import import_mesh_room
from extractors.entity_surfaces import derive_entity_top_surfaces
from geometry.rest_contact import RestContactConfig
from graph.builder import build_graph
from graph.relations.on_entity_surface import OnEntitySurfaceConfig
from reasoner.ast import EdgeConstraint, EntityClassRef
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer
from segmenter.base import (
    SegmentationOutput,
    load_segmentation_output,
    save_segmentation_output,
)
from segmenter.derived import build_c1_eval_bundle
from segmenter.ply import parse_vertices
from segmenter.query_expansion import (
    materialize_query_scoped_assignment,
    raw_proposal_entities,
    select_support_region_candidates,
)
from segmenter.uncertainty_policy import (
    UncertaintyPolicyConfig,
    build_policy_view,
)
from tools.c1_resolve_sweep import load_raw_masks


def _router() -> Router:
    return Router(
        compiler=RulesCompiler(),
        executor=RulesExecutor(),
        verbalizer=StandardVerbalizer(),
    )


def _ctx(source: str) -> ExecutionContext:
    return ExecutionContext(completeness=CompletenessProfile(
        source=source,
        entity_recall_by_class={},
        edge_recall_by_type={},
    ))


def _variant_segmentation(
    source: SegmentationOutput,
    scores: np.ndarray,
    assignment: np.ndarray,
    *,
    variant: str,
    config: UncertaintyPolicyConfig,
) -> SegmentationOutput:
    ids = [int(i) for i in np.unique(assignment) if i >= 0]
    source_config = json.loads(source.config_params_json)
    return SegmentationOutput(
        input_mesh_sha256=source.input_mesh_sha256,
        n_vertices=source.n_vertices,
        segmenter_name=source.segmenter_name,
        segmenter_version=source.segmenter_version,
        config_params_json=json.dumps({
            **source_config,
            "query_scoped_expansion_demo": {
                "variant": variant,
                "accepted_score": config.accepted_score,
                "provisional_score": config.provisional_score,
                "min_vertices": config.min_vertices,
                "score_transform": config.score_transform,
                "note": (
                    "policy comparison over identical saved raw masks; "
                    "not a model change"
                ),
            },
        }, sort_keys=True),
        vertex_instance_ids=np.asarray(assignment, dtype=np.int64),
        instance_confidence={instance_id: float(scores[instance_id])
                             for instance_id in ids},
        runtime_seconds=source.runtime_seconds,
        hardware=source.hardware,
    ).finalize()


def _frame_rotation(
    room_dir: Path,
    *,
    z_translation: float,
) -> tuple[tuple[float, float, float], ...]:
    info = json.loads(
        (room_dir / "habitat" / "info_semantic.json").read_text(encoding="utf-8")
    )
    gravity = info["gravity_dir"]
    initial = _gravity_align_matrix(
        (float(gravity[0]), float(gravity[1]), float(gravity[2]))
    )
    rotation, _surfaces, _diag, _yaw = _aligned_structural_surfaces(
        info, initial, z_translation
    )
    return tuple(tuple(float(v) for v in row) for row in rotation)


def _canonical_mesh_vertices(
    room_dir: Path,
    rotation: tuple[tuple[float, float, float], ...],
    *,
    z_translation: float,
) -> np.ndarray:
    xyz = parse_vertices(room_dir / "mesh.ply")
    xyz = np.einsum("ij,nj->ni", np.asarray(rotation), xyz)
    xyz[:, 2] += z_translation
    if not np.isfinite(xyz).all():
        raise ValueError("non-finite canonical mesh vertices")
    return xyz


def _uid_map(injection: dict) -> dict[str, str]:
    return {
        row["uid"]: f"obj_{row['oracle_id']}"
        for row in injection["oracle_injections"]
    }


def _answer_payload(answer, graph, *, pred_to_oracle: dict[str, str] | None) -> dict:
    labels = {node.id: node.label for node in graph.nodes}
    if pred_to_oracle is None:
        translated = sorted(answer.cited_uids)
    else:
        translated = sorted(
            pred_to_oracle.get(uid, f"pred:{uid}") for uid in answer.cited_uids
        )
    return {
        "outcome": answer.outcome,
        "text": answer.text,
        "cited_uids": sorted(answer.cited_uids),
        "cited_labels": {
            uid: labels.get(uid, "unknown") for uid in sorted(answer.cited_uids)
        },
        "cited_edges": sorted(answer.cited_edges),
        "translated_oracle_uids": translated,
    }


def _build_graph_and_answer(
    artifacts,
    query: str,
    router: Router,
    ctx: ExecutionContext,
):
    graph, diagnostics = build_graph(
        artifacts,
        _runs(),
        density_policy="phase2_telemetry_only",
    )
    answer = router.answer(query, graph, ctx)
    return graph, diagnostics, answer


def _eval_variant(
    room_dir: Path,
    variant_dir: Path,
    scene_id: str,
    query: str,
    router: Router,
    *,
    z_translation: float,
    min_vertices: int,
) -> dict:
    artifacts, injection = build_c1_eval_bundle(
        room_dir,
        variant_dir,
        scene_id,
        z_translation=z_translation,
        min_vertices=min_vertices,
    )
    graph, diagnostics, answer = _build_graph_and_answer(
        artifacts, query, router, _ctx("unknown")
    )
    return {
        "artifacts": artifacts,
        "graph": graph,
        "injection": injection,
        "answer": _answer_payload(
            answer, graph, pred_to_oracle=_uid_map(injection)
        ),
        "graph_summary": {
            "bundle_hash": graph.bundle_hash,
            "n_nodes": len(graph.nodes),
            "n_edges": len(graph.edges),
            "n_on_entity_surface_edges": (
                diagnostics.edges_emitted_per_type.get("ON_ENTITY_SURFACE", 0)
            ),
        },
    }


def _support_class(query: str, graph, router: Router) -> tuple[str, str]:
    compiled = router.compiler.compile(query, graph)
    if compiled.outcome != "compiled" or compiled.ast is None:
        raise ValueError(
            f"query must compile through RulesCompiler; got "
            f"{compiled.outcome}: {compiled.notes}"
        )
    constraints = [
        constraint for constraint in compiled.ast.where
        if isinstance(constraint, EdgeConstraint)
        and constraint.type == "SUPPORTS"
        and isinstance(constraint.source, EntityClassRef)
    ]
    if len(constraints) != 1:
        raise ValueError(
            "query-scoped demo requires exactly one entity-class SUPPORTS "
            f"constraint; found {len(constraints)}"
        )
    return constraints[0].source.entity_class, compiled.notes


def _reference_answer(artifacts, query: str, router: Router) -> dict:
    graph, _diagnostics, answer = _build_graph_and_answer(
        artifacts, query, router, _ctx("oracle")
    )
    return _answer_payload(answer, graph, pred_to_oracle=None)


def _reference_comparison(answer: dict, reference: dict) -> dict:
    cited = set(answer["translated_oracle_uids"])
    expected = set(reference["translated_oracle_uids"])
    hits = cited & expected
    return {
        "reference_uids": sorted(expected),
        "matching_uids": sorted(hits),
        "missed_uids": sorted(expected - cited),
        "extra_uids": sorted(cited - expected),
        "diagnostic_recall": len(hits) / len(expected) if expected else None,
        "diagnostic_precision": len(hits) / len(cited) if cited else None,
    }


def _decision_payload(decision, evidence_by_id: dict[int, object]) -> dict:
    proposal_id = int(decision.candidate_uid.split("_", 1)[1])
    evidence = evidence_by_id[proposal_id]
    return {
        "proposal_id": proposal_id,
        "candidate_uid": decision.candidate_uid,
        "selected": decision.selected,
        "matching_anchor_uids": decision.matching_anchor_uids,
        "backend_score": evidence.backend_score,
        "association_confidence": evidence.association_confidence,
        "assigned_vertices": evidence.assigned_vertices,
        "anchor_checks": [
            {
                "anchor_uid": check.anchor_uid,
                "on_surface": check.on_surface,
                "failed_clauses": check.failed_clauses,
                "bottom_gap_m": check.bottom_gap_m,
                "in_plane_gap_m": check.in_plane_gap_m,
            }
            for check in decision.checks
        ],
    }


def run_experiment(
    room_dir: Path,
    bundle_dir: Path,
    scene_id: str,
    query: str,
    config: UncertaintyPolicyConfig,
    *,
    z_translation: float = ROOM_0_Z_TRANSLATION,
) -> dict:
    config.validate()
    source = load_segmentation_output(bundle_dir)
    masks, scores = load_raw_masks(bundle_dir)
    if masks.shape[1] != source.n_vertices:
        raise ValueError(
            f"raw masks have {masks.shape[1]} vertices; "
            f"source sidecar has {source.n_vertices}"
        )
    view = build_policy_view(masks, scores, config)
    router = _router()

    with tempfile.TemporaryDirectory() as td:
        temp_root = Path(td)
        hard_dir = temp_root / "hard"
        hard_seg = _variant_segmentation(
            source, scores, view.hard_assignment,
            variant="hard", config=config,
        )
        save_segmentation_output(hard_seg, hard_dir)
        hard = _eval_variant(
            room_dir, hard_dir, scene_id, query, router,
            z_translation=z_translation, min_vertices=config.min_vertices,
        )

        support_class, compiler_notes = _support_class(
            query, hard["graph"], router
        )
        relation_config = OnEntitySurfaceConfig()
        anchor_surfaces = [
            surface for surface in derive_entity_top_surfaces(
                hard["artifacts"].entities,
                frame=hard["artifacts"].frame,
                support_class_allowlist=relation_config.support_class_allowlist,
            )
            if surface.owner_class == support_class
        ]

        activation_ids = {
            proposal.proposal_id
            for proposal in view.proposals
            if proposal.raw_activation_candidate
        }
        rotation = _frame_rotation(room_dir, z_translation=z_translation)
        raw_candidates = raw_proposal_entities(
            _canonical_mesh_vertices(
                room_dir, rotation, z_translation=z_translation
            ),
            masks,
            proposal_ids=activation_ids,
        )
        activation_uids = {f"obj_{proposal_id}" for proposal_id in activation_ids}
        rest_config = RestContactConfig(
            contact_threshold_m=relation_config.contact_threshold_m,
            penetration_tolerance_m=relation_config.penetration_tolerance_m,
            max_tilt_deg=relation_config.max_tilt_deg,
            footprint_tolerance_m=relation_config.footprint_tolerance_m,
        )
        decisions = select_support_region_candidates(
            raw_candidates,
            candidate_uids=activation_uids,
            anchors=anchor_surfaces,
            gravity=hard["artifacts"].frame.gravity,
            rest_config=rest_config,
        )
        selected_ids = {
            int(decision.candidate_uid.split("_", 1)[1])
            for decision in decisions if decision.selected
        }
        protected_anchor_ids = {
            int(surface.owner_entity_uid.split("_", 1)[1])
            for surface in anchor_surfaces
        }

        scoped_dir = temp_root / "query_scoped"
        scoped_seg = _variant_segmentation(
            source,
            scores,
            materialize_query_scoped_assignment(
                view.hard_assignment,
                masks,
                scores,
                selected_ids=selected_ids,
                protected_ids=protected_anchor_ids,
                min_vertices=config.min_vertices,
            ),
            variant="query_scoped",
            config=config,
        )
        scoped_instance_ids = set(scoped_seg.instance_ids())
        hard_instance_ids = set(hard_seg.instance_ids())
        save_segmentation_output(scoped_seg, scoped_dir)
        scoped = _eval_variant(
            room_dir, scoped_dir, scene_id, query, router,
            z_translation=z_translation, min_vertices=config.min_vertices,
        )

        inclusive_dir = temp_root / "inclusive"
        inclusive_seg = _variant_segmentation(
            source, scores, view.inclusive_assignment,
            variant="inclusive", config=config,
        )
        save_segmentation_output(inclusive_seg, inclusive_dir)
        inclusive = _eval_variant(
            room_dir, inclusive_dir, scene_id, query, router,
            z_translation=z_translation, min_vertices=config.min_vertices,
        )

        reference_a = _reference_answer(
            import_habitat_room(
                room_dir, scene_id, z_translation=z_translation
            ),
            query,
            router,
        )
        reference_b = _reference_answer(
            import_mesh_room(
                room_dir, scene_id, z_translation=z_translation
            ),
            query,
            router,
        )

        evidence_by_id = {
            proposal.proposal_id: proposal for proposal in view.proposals
        }
        hard_answer = hard["answer"]
        scoped_answer = scoped["answer"]
        inclusive_answer = inclusive["answer"]
        hard_uids = set(hard_answer["translated_oracle_uids"])
        scoped_uids = set(scoped_answer["translated_oracle_uids"])
        ref_a_uids = set(reference_a["translated_oracle_uids"])
        ref_b_uids = set(reference_b["translated_oracle_uids"])
        references_agree = ref_a_uids == ref_b_uids
        stable_reference = ref_a_uids if references_agree else set()
        new_scoped = scoped_uids - hard_uids

        return {
            "schema": "query_scoped_expansion_demo_v2",
            "scene_id": scene_id,
            "query": query,
            "hypothesis": (
                "query-local activation can recover support answers while "
                "avoiding the extra citations of global inclusion"
            ),
            "interpretation": {
                "answer_key_type": (
                    "dual_replica_reference_diagnostic_not_human_verified"
                ),
                "labels_and_surfaces": (
                    "oracle-injected after geometry-only activation for C1 "
                    "instance-boundary isolation"
                ),
                "score_semantics": (
                    "thresholds use native backend scores; existence evidence "
                    f"uses {config.score_transform}"
                ),
                "accuracy_claim_authorized": False,
            },
            "config": {
                "accepted_score": config.accepted_score,
                "provisional_score": config.provisional_score,
                "min_vertices": config.min_vertices,
                "score_transform": config.score_transform,
                "support_class": support_class,
                "compiler_notes": compiler_notes,
                "rest_contact": {
                    "contact_threshold_m": rest_config.contact_threshold_m,
                    "penetration_tolerance_m": (
                        rest_config.penetration_tolerance_m
                    ),
                    "max_tilt_deg": rest_config.max_tilt_deg,
                    "footprint_tolerance_m": (
                        rest_config.footprint_tolerance_m
                    ),
                },
            },
            "activation": {
                "n_hard_anchors": len(anchor_surfaces),
                "hard_anchor_uids": [
                    surface.owner_entity_uid for surface in anchor_surfaces
                ],
                "protected_anchor_proposal_ids": sorted(protected_anchor_ids),
                "n_raw_activation_candidates": len(activation_uids),
                "n_score_provisional_candidates": sum(
                    proposal.state == "provisional"
                    and proposal.raw_activation_candidate
                    for proposal in view.proposals
                ),
                "n_accepted_composition_lost_candidates": sum(
                    proposal.state == "accepted"
                    and proposal.raw_activation_candidate
                    for proposal in view.proposals
                ),
                "n_empty_scored_masks_excluded": sum(
                    proposal.raw_vertices == 0
                    and proposal.state != "discarded"
                    for proposal in view.proposals
                ),
                "n_selected": len(selected_ids),
                "selected_proposal_ids": sorted(selected_ids),
                "selected_proposal_ids_materialized": sorted(
                    selected_ids & scoped_instance_ids
                ),
                "selected_proposal_ids_dropped_after_recomposition": sorted(
                    selected_ids - scoped_instance_ids
                ),
                "accepted_instance_ids_displaced": sorted(
                    hard_instance_ids - scoped_instance_ids
                ),
                "decisions": [
                    _decision_payload(decision, evidence_by_id)
                    for decision in decisions
                ],
            },
            "references": {
                "a_habitat_boxes": reference_a,
                "b_semantic_mesh_boxes": reference_b,
                "a_b_answer_sets_agree": references_agree,
                "stable_reference_uids": sorted(stable_reference),
            },
            "variants": {
                "hard": {
                    "answer": hard_answer,
                    "graph": hard["graph_summary"],
                    "vs_reference_a": _reference_comparison(
                        hard_answer, reference_a
                    ),
                    "vs_reference_b": _reference_comparison(
                        hard_answer, reference_b
                    ),
                },
                "query_scoped": {
                    "answer": scoped_answer,
                    "graph": scoped["graph_summary"],
                    "vs_reference_a": _reference_comparison(
                        scoped_answer, reference_a
                    ),
                    "vs_reference_b": _reference_comparison(
                        scoped_answer, reference_b
                    ),
                },
                "inclusive": {
                    "answer": inclusive_answer,
                    "graph": inclusive["graph_summary"],
                    "vs_reference_a": _reference_comparison(
                        inclusive_answer, reference_a
                    ),
                    "vs_reference_b": _reference_comparison(
                        inclusive_answer, reference_b
                    ),
                },
            },
            "delta": {
                "raw_prediction_answer_set_changed": (
                    set(scoped_answer["cited_uids"])
                    != set(hard_answer["cited_uids"])
                ),
                "new_prediction_uids": sorted(
                    set(scoped_answer["cited_uids"])
                    - set(hard_answer["cited_uids"])
                ),
                "removed_prediction_uids": sorted(
                    set(hard_answer["cited_uids"])
                    - set(scoped_answer["cited_uids"])
                ),
                "translated_answer_set_changed": bool(new_scoped) or bool(
                    hard_uids - scoped_uids
                ),
                "new_query_scoped_uids": sorted(new_scoped),
                "recovered_stable_reference_uids": sorted(
                    new_scoped & stable_reference
                ),
                "new_query_scoped_nonreference_uids": sorted(
                    new_scoped - stable_reference
                ) if references_agree else [],
                "inclusive_only_uids": sorted(
                    set(inclusive_answer["translated_oracle_uids"]) - scoped_uids
                ),
            },
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("room_dir", type=Path)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("scene_id")
    parser.add_argument("--query", default="what is on the table?")
    parser.add_argument("--accepted-score", type=float, default=0.2)
    parser.add_argument("--provisional-score", type=float, default=0.05)
    parser.add_argument("--min-vertices", type=int, default=20)
    parser.add_argument(
        "--score-transform",
        choices=("identity_probability", "sigmoid_logit"),
        default="identity_probability",
    )
    parser.add_argument("--z-translation", type=float, default=ROOM_0_Z_TRANSLATION)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    config = UncertaintyPolicyConfig(
        accepted_score=args.accepted_score,
        provisional_score=args.provisional_score,
        min_vertices=args.min_vertices,
        score_transform=args.score_transform,
    )
    try:
        report = run_experiment(
            args.room_dir,
            args.bundle_dir,
            args.scene_id,
            args.query,
            config,
            z_translation=args.z_translation,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"[query_scoped_expansion_demo] HARD FAIL: {exc}")
        return 1

    support_class = report["config"]["support_class"].replace("-", "_")
    out = args.out or (
        REPO_ROOT / "runs" / "phase8_c1"
        / f"{args.scene_id}_{support_class}_query_expansion.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    activation = report["activation"]
    delta = report["delta"]
    print(
        f"[query_scoped_expansion_demo] {args.scene_id} {args.query!r}: "
        f"anchors={activation['n_hard_anchors']} "
        f"raw_candidates={activation['n_raw_activation_candidates']} "
        f"selected={activation['selected_proposal_ids']}"
    )
    for name in ("hard", "query_scoped", "inclusive"):
        answer = report["variants"][name]["answer"]
        print(
            f"  {name:12} {answer['outcome']:8} "
            f"{answer['translated_oracle_uids']}"
        )
    print(
        "  recovered stable reference: "
        f"{delta['recovered_stable_reference_uids']}"
    )
    try:
        out_label = out.relative_to(REPO_ROOT)
    except ValueError:
        out_label = out
    print(f"[query_scoped_expansion_demo] report -> {out_label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

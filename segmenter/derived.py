"""C1.03 — evaluation-only derived bundle: anonymous candidate + oracle.

    anonymous candidate (segmenter/candidate.py)
      + held-out mesh_semantic.ply / info_semantic.json
      -> exact vertex-index correspondence (tools/c1_exact_eval.py)
      -> oracle labels injected on matched segments
      -> variant A's structural surfaces injected verbatim
      -> C1 EVALUATION bundle (distinct hash from the candidate)

This module is the ONLY place oracle content meets segmenter output. Rules:
  - matched segments get the oracle class as display_label, provenance
    recorded per entity in notes["oracle_injections"];
  - matched segments whose oracle class is structural (floor/wall/ceiling)
    or dropped (undefined/non-plane/plane) are REMOVED — mirroring how A/B
    route those classes to surfaces / drop them — and recorded in
    notes["removed_structural_or_dropped"], never silently;
  - unmatched predictions KEEP their anonymous segment_<id> label and stay
    in the bundle (they are the segmenter's false positives; dropping them
    would hide errors);
  - the frame passed to the candidate builder is A's (shared helper), so
    C1 boxes are directly comparable to A/B boxes.

Reports built on this bundle must state that labels and surfaces were
injected for isolation (see docs/mesh_pipeline_contract.md).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from demo.replica_habitat_import import (
    ROOM_0_Z_TRANSLATION,
    STRUCTURAL_CLASSES,
    _aligned_structural_surfaces,
    _gravity_align_matrix,
    import_habitat_room,
)
from extractors.base import EntityArtifacts
from segmenter.base import load_segmentation_output
from segmenter.candidate import build_candidate_artifacts
from tools.c1_exact_eval import evaluate

DROP_CLASSES = ("undefined", "non-plane", "plane")


def build_c1_eval_bundle(
    room_dir: Path,
    bundle_dir: Path,
    scene_id: str,
    *,
    z_translation: float = ROOM_0_Z_TRANSLATION,
    min_vertices: int = 20,
) -> tuple[EntityArtifacts, dict]:
    """Returns (enriched EntityArtifacts, correspondence/injection report)."""
    report = evaluate(room_dir, bundle_dir)          # hard-checks G1 + hashes
    seg = load_segmentation_output(bundle_dir)

    info = json.loads((room_dir / "habitat" / "info_semantic.json").read_text())
    g = info["gravity_dir"]
    R0 = _gravity_align_matrix((float(g[0]), float(g[1]), float(g[2])))
    R, _, _, _ = _aligned_structural_surfaces(info, R0, z_translation)

    candidate = build_candidate_artifacts(
        room_dir / "mesh.ply", seg, scene_id,
        rotation=R, z_translation=z_translation,
        bundle_dir=bundle_dir, min_vertices=min_vertices)

    A = import_habitat_room(room_dir, scene_id, z_translation=z_translation)

    pred_to_match = {m["pred_id"]: m for m in report["matches"]}
    entities = []
    injections = []
    removed = []
    unmatched = []
    for e in candidate.entities:
        pred_id = int(e.identity.source_instance_ref.split(":")[1])
        m = pred_to_match.get(pred_id)
        if m is None or not m["oracle_class"]:
            unmatched.append(e.identity.object_uid)
            entities.append(e)                        # stays anonymous
            continue
        label = m["oracle_class"]
        if label in STRUCTURAL_CLASSES or label in DROP_CLASSES:
            removed.append({"uid": e.identity.object_uid, "oracle_id": m["oracle_id"],
                            "oracle_class": label})
            continue
        entities.append(dataclasses.replace(
            e,
            identity=dataclasses.replace(e.identity, display_label=label),
        ))
        injections.append({"uid": e.identity.object_uid,
                           "oracle_id": m["oracle_id"],
                           "oracle_class": label,
                           "iou": m["iou"]})

    injection_report = {
        "provenance": "oracle_correspondence",
        "n_injected_labels": len(injections),
        "n_unmatched_kept_anonymous": len(unmatched),
        "n_removed_structural_or_dropped": len(removed),
        "oracle_injections": injections,
        "unmatched_kept_anonymous": unmatched,
        "removed_structural_or_dropped": removed,
        "surfaces_injected_from": "replica_habitat_import (variant A, verbatim)",
        "exact_eval": {k: report[k] for k in (
            "n_matched", "n_unmatched_predictions", "n_unmatched_oracle",
            "recall_at_iou", "support_owner")},
    }

    arts = dataclasses.replace(
        candidate,
        bundle_hash=f"c1eval_{seg.output_sha256[:16]}",   # distinct from candidate
        extractor_name="c1_eval_bundle",
        entities=entities,
        structural_surfaces=A.structural_surfaces,        # verbatim from A
        diagnostics=dataclasses.replace(
            candidate.diagnostics,
            n_entities=len(entities),
            n_structural_surfaces=len(A.structural_surfaces),
            notes=(f"C1 eval bundle: {len(injections)} oracle labels injected, "
                   f"{len(unmatched)} anonymous, {len(removed)} structural removed; "
                   f"surfaces from variant A"),
        ),
        notes={
            **candidate.notes,
            "semantic_source": "oracle_correspondence",
            "surface_source": "variant_A_verbatim",
            "oracle_injection": injection_report,
            "isolation_statement": (
                "labels and structural surfaces were INJECTED from oracle "
                "data for C1 isolation; only instance boundaries are learned"),
        },
    )
    return arts, injection_report

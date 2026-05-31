"""Replica oracle InstanceExtractor.

Reads a SceneRepresentation whose bundle.notes['semantic_export_path']
points at a pre-imported Replica scene_graph.json (produced by
importers/replica.py), and emits an EntityArtifacts bundle with stable
per-instance object_uids.

Boundary discipline (per Phase 1 batch instructions):
  - This module owns Habitat semantic instance enumeration, immutable
    object_uid assignment, semantic hypotheses, and the initial entity
    bundle.
  - It does NOT touch capture metadata, mesh handles, or frame
    normalization — those are the adapter's territory. The extractor
    inherits the SceneFrame from the SceneRepresentationBundle verbatim.

Phase 1 oracle path does NOT emit structural surfaces (Phase 2 work per
phase0_design.md §11). The extractor returns an empty list and records
this status in diagnostics so downstream code knows floor / wall /
ceiling refs will be empty.

object_uid is preserved verbatim from the pre-imported scene_graph.json
(e.g. "obj_93"), which itself derived from the Habitat instance_id. This
keeps identity stable across reruns of the same imported scene and
across processes.

Display labels and aliases:
  - display_label = obj["label"]   (e.g. "table_1", suffixed by importer)
  - aliases includes the base label stripped of "_<int>" suffix if
    present, so a query for "table" can match table_1 and table_2.
  - source_instance_ref strips the "obj_" prefix to recover the Habitat
    instance_id as a string.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from extractors.base import (
    EntityArtifact, EntityArtifacts, EntityIdentity, ExtractionDiagnostics,
    InstanceExtractorCapabilities, InstanceExtractorConfig, SemanticHypothesis,
)
from extractors.serde import CURRENT_SCHEMA_VERSION
from representations.base import Channel, SceneRepresentation


_EXTRACTOR_NAME = "oracle_replica"
_EXTRACTOR_VERSION = "0.1"

_SUFFIX_PATTERN = re.compile(r"^(.+?)_(\d+)$")


def _base_label(display_label: str) -> str | None:
    """Recover the base label from a suffixed display label per the legacy
    importer convention ('<base>_<int>' → '<base>'). Returns None if the
    label is not suffixed."""
    m = _SUFFIX_PATTERN.match(display_label)
    return m.group(1) if m else None


def _source_instance_ref_from_id(object_id: str) -> str:
    """Strip the importer's 'obj_' prefix to recover the source Habitat
    instance_id as a string."""
    if object_id.startswith("obj_"):
        return object_id[len("obj_"):]
    return object_id


def _bundle_hash(
    representation_hash: str,
    extractor_name: str,
    extractor_version: str,
    config_params: dict[str, Any],
) -> str:
    payload = json.dumps(
        {
            "representation_hash": representation_hash,
            "extractor_name": extractor_name,
            "extractor_version": extractor_version,
            "config_params": config_params,
        },
        sort_keys=True,
    )
    return f"ent_{extractor_name}_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _build_entity(obj: dict[str, Any]) -> EntityArtifact:
    object_uid = str(obj["id"])
    display_label = str(obj["label"])
    base = _base_label(display_label)
    aliases: list[str] = []
    if base is not None and base != display_label:
        aliases.append(base)

    xyz = obj["xyz"]
    centroid = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    sizes = obj.get("attributes", {}).get("bbox_sizes", [0.0, 0.0, 0.0])
    sx, sy, sz = float(sizes[0]), float(sizes[1]), float(sizes[2])
    half = (sx / 2.0, sy / 2.0, sz / 2.0)
    bbox_aabb = (
        (centroid[0] - half[0], centroid[1] - half[1], centroid[2] - half[2]),
        (centroid[0] + half[0], centroid[1] + half[1], centroid[2] + half[2]),
    )

    return EntityArtifact(
        identity=EntityIdentity(
            object_uid=object_uid,
            display_label=display_label,
            aliases=aliases,
            source_instance_ref=_source_instance_ref_from_id(object_uid),
        ),
        bbox_aabb=bbox_aabb,
        bbox_obb=None,
        centroid=centroid,
        geometry_handle=None,
        semantic_hypotheses=[
            SemanticHypothesis(
                label=base if base is not None else display_label,
                confidence=1.0,
                source="habitat_oracle",
            ),
        ],
        embedding=None,
        extraction_diagnostics={
            "bbox_aabb_is_approximate": True,
            "bbox_aabb_note": (
                "sizes are in object-local frame; true world-frame AABB "
                "requires OBB application (Phase 2 work)"
            ),
        },
    )


@dataclass(frozen=True)
class OracleReplicaExtractor:
    """InstanceExtractor that emits one EntityArtifact per object in a
    pre-imported Replica scene_graph.json. Structural surfaces are
    intentionally empty (Phase 2 work)."""
    name: str = _EXTRACTOR_NAME
    version: str = _EXTRACTOR_VERSION
    required_channels: frozenset[Channel] = field(default_factory=frozenset)

    def extract(
        self,
        representation: SceneRepresentation,
        config: InstanceExtractorConfig,
    ) -> EntityArtifacts:
        start = time.perf_counter()
        bundle = representation.bundle
        sg_path_str = bundle.notes.get("semantic_export_path")
        if not sg_path_str:
            raise ValueError(
                "OracleReplicaExtractor requires bundle.notes['semantic_export_path']; "
                "ensure the upstream adapter populated it."
            )
        sg_path = Path(sg_path_str)
        scene_graph = json.loads(sg_path.read_text(encoding="utf-8"))
        objects = scene_graph.get("objects", [])

        entities = [_build_entity(o) for o in objects]
        uids = [e.identity.object_uid for e in entities]
        if len(set(uids)) != len(uids):
            duplicates = [u for u in uids if uids.count(u) > 1]
            raise ValueError(
                f"object_uid collision in extractor input {sg_path}: "
                f"duplicates={sorted(set(duplicates))}"
            )

        runtime = time.perf_counter() - start

        return EntityArtifacts(
            schema_version=CURRENT_SCHEMA_VERSION,
            bundle_hash=_bundle_hash(
                bundle.representation_hash, self.name, self.version, dict(config.params),
            ),
            scene_id=bundle.scene_id,
            frame=bundle.frame,
            representation_hash=bundle.representation_hash,
            extractor_name=self.name,
            extractor_version=self.version,
            entities=entities,
            structural_surfaces=[],
            geometry_store_path=None,
            diagnostics=ExtractionDiagnostics(
                n_entities=len(entities),
                n_structural_surfaces=0,
                runtime_seconds=runtime,
                coverage_score=1.0,
                notes=(
                    "Oracle wrapper around pre-imported scene_graph.json. "
                    "Structural surfaces (floor / wall / ceiling) deferred to "
                    "Phase 2 per phase0_design.md §11."
                ),
            ),
            notes={
                "semantic_export_path": str(sg_path),
                "z_translation_already_applied": True,
                "structural_surfaces_status": "deferred_to_phase_2",
            },
        )

    def capabilities(self) -> InstanceExtractorCapabilities:
        return InstanceExtractorCapabilities(
            label_vocab=None,
            provides_embeddings=False,
            provides_oriented_bboxes=False,
            provides_structural_surfaces=False,
            extractor_class_hint="all",
        )

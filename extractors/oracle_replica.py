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

from common.types import OrientedBBox, Plane, Vec3
from extractors.base import (
    EntityArtifact, EntityArtifacts, EntityIdentity, ExtractionDiagnostics,
    InstanceExtractorCapabilities, InstanceExtractorConfig, SemanticHypothesis,
    StructuralSurface,
)
from extractors.serde import CURRENT_SCHEMA_VERSION
from representations.base import Channel, SceneRepresentation


_VALID_SURFACE_TYPES = ("floor", "wall", "ceiling")
_VALID_SURFACE_SOURCES = ("habitat_label", "mesh_ransac", "synth_bbox_fallback")


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
    enriched_v2_content_hash: str | None = None,
) -> str:
    """Hash payload preserves Phase 1 keys when no v2 content hash is supplied,
    so default-mode artifacts retain their historical bundle_hash. When
    enriched-v2 content is supplied, its digest joins the payload so the
    v2 bundle is content-addressed rather than filesystem-path-addressed."""
    payload_dict: dict[str, Any] = {
        "representation_hash": representation_hash,
        "extractor_name": extractor_name,
        "extractor_version": extractor_version,
        "config_params": config_params,
    }
    if enriched_v2_content_hash is not None:
        payload_dict["enriched_v2_content_hash"] = enriched_v2_content_hash
    payload = json.dumps(payload_dict, sort_keys=True)
    return f"ent_{extractor_name}_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _identity_for(obj: dict[str, Any]) -> tuple[EntityIdentity, str | None]:
    object_uid = str(obj["id"])
    display_label = str(obj["label"])
    base = _base_label(display_label)
    aliases: list[str] = []
    if base is not None and base != display_label:
        aliases.append(base)
    identity = EntityIdentity(
        object_uid=object_uid,
        display_label=display_label,
        aliases=aliases,
        source_instance_ref=_source_instance_ref_from_id(object_uid),
    )
    return identity, base


def _build_entity(obj: dict[str, Any]) -> EntityArtifact:
    """Phase 1 entity build: AABB derived from local-frame sizes (approximate).
    Behavior frozen so bundle_hash and downstream comparisons remain stable."""
    identity, base = _identity_for(obj)

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
        identity=identity,
        bbox_aabb=bbox_aabb,
        bbox_obb=None,
        centroid=centroid,
        geometry_handle=None,
        semantic_hypotheses=[
            SemanticHypothesis(
                label=base if base is not None else identity.display_label,
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


def _build_entity_v2(obj: dict[str, Any]) -> EntityArtifact:
    """Enriched-v2 entity build: tight world-frame AABB from the importer
    plus the rotated OBB. No 'bbox_aabb_is_approximate' diagnostic."""
    identity, base = _identity_for(obj)

    xyz = obj["xyz"]
    centroid = (float(xyz[0]), float(xyz[1]), float(xyz[2]))

    bbox_aabb_raw = obj.get("bbox_aabb")
    if bbox_aabb_raw is None:
        raise ValueError(
            f"enriched-v2 object {obj.get('id')!r} missing required 'bbox_aabb'; "
            f"upstream importer must emit schema_version 2"
        )
    bbox_aabb: tuple[Vec3, Vec3] = (
        (float(bbox_aabb_raw[0][0]), float(bbox_aabb_raw[0][1]), float(bbox_aabb_raw[0][2])),
        (float(bbox_aabb_raw[1][0]), float(bbox_aabb_raw[1][1]), float(bbox_aabb_raw[1][2])),
    )

    obb_raw = obj.get("bbox_obb")
    if obb_raw is None:
        raise ValueError(
            f"enriched-v2 object {obj.get('id')!r} missing required 'bbox_obb'"
        )
    bbox_obb = OrientedBBox(
        center=(float(obb_raw["center"][0]), float(obb_raw["center"][1]), float(obb_raw["center"][2])),
        extents=(float(obb_raw["extents"][0]), float(obb_raw["extents"][1]), float(obb_raw["extents"][2])),
        rotation_quat=(
            float(obb_raw["rotation_quat"][0]),
            float(obb_raw["rotation_quat"][1]),
            float(obb_raw["rotation_quat"][2]),
            float(obb_raw["rotation_quat"][3]),
        ),
    )

    return EntityArtifact(
        identity=identity,
        bbox_aabb=bbox_aabb,
        bbox_obb=bbox_obb,
        centroid=centroid,
        geometry_handle=None,
        semantic_hypotheses=[
            SemanticHypothesis(
                label=base if base is not None else identity.display_label,
                confidence=1.0,
                source="habitat_oracle",
            ),
        ],
        embedding=None,
        extraction_diagnostics={},
    )


def _build_structural_surface(s: dict[str, Any]) -> StructuralSurface:
    """Convert one importer-emitted structural-surface JSON record into
    a StructuralSurface dataclass. Validates surface_type and source."""
    stype = s["surface_type"]
    if stype not in _VALID_SURFACE_TYPES:
        raise ValueError(f"unknown surface_type {stype!r}")
    src = s.get("source")
    if src not in _VALID_SURFACE_SOURCES:
        raise ValueError(
            f"unknown source {src!r} on surface {s.get('surface_uid')!r}; "
            f"allowed: {_VALID_SURFACE_SOURCES}"
        )
    plane_d = s["plane"]
    normal = plane_d["normal"]
    plane = Plane(
        a=float(normal[0]), b=float(normal[1]), c=float(normal[2]),
        d=float(plane_d["d"]),
    )
    poly_raw = s.get("polygon")
    polygon: list[Vec3] | None
    if poly_raw is None:
        polygon = None
    else:
        polygon = [
            (float(p[0]), float(p[1]), float(p[2])) for p in poly_raw
        ]
    return StructuralSurface(
        surface_uid=str(s["surface_uid"]),
        surface_type=stype,
        plane=plane,
        polygon=polygon,
        confidence=float(s.get("confidence", 1.0)),
        source=src,
    )


@dataclass(frozen=True)
class OracleReplicaExtractor:
    """InstanceExtractor that emits one EntityArtifact per object in a
    pre-imported Replica scene_graph.json.

    Default mode (`enriched_v2_path is None`) preserves the Phase 1
    behavior exactly: AABB derived from local-frame sizes, no OBB,
    empty structural_surfaces, capability flags False, bundle_hash
    bit-identical to the historical hash.

    Enriched-v2 mode (`enriched_v2_path` set) reads from the importer's
    schema-v2 output (scenes/<scene>/enriched/v2/scene_graph.json):
    tight world-frame AABBs from the rotated OBB, populated bbox_obb on
    every entity, structural_surfaces with `source` provenance, both
    capability flags True. The v2 path is opt-in only; Phase 1 callers
    are unaffected."""
    name: str = _EXTRACTOR_NAME
    version: str = _EXTRACTOR_VERSION
    required_channels: frozenset[Channel] = field(default_factory=frozenset)
    enriched_v2_path: Path | None = None

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

        if self.enriched_v2_path is None:
            return self._extract_phase1(bundle, config, Path(sg_path_str), start)
        return self._extract_v2(bundle, config, Path(sg_path_str), start)

    def _extract_phase1(
        self, bundle, config, sg_path: Path, start: float,
    ) -> EntityArtifacts:
        scene_graph = json.loads(sg_path.read_text(encoding="utf-8"))
        objects = scene_graph.get("objects", [])
        entities = [_build_entity(o) for o in objects]
        _assert_unique_uids(entities, sg_path)
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

    def _extract_v2(
        self, bundle, config, phase1_sg_path: Path, start: float,
    ) -> EntityArtifacts:
        v2_sg_path = self.enriched_v2_path / "scene_graph.json"  # type: ignore[union-attr]
        v2_scene_graph_bytes = v2_sg_path.read_bytes()
        v2_scene_graph = json.loads(v2_scene_graph_bytes)
        if v2_scene_graph.get("schema_version") != 2:
            raise ValueError(
                f"enriched-v2 scene_graph at {v2_sg_path} has schema_version="
                f"{v2_scene_graph.get('schema_version')!r}; expected 2"
            )
        if v2_scene_graph.get("scene") != bundle.scene_id:
            raise ValueError(
                f"enriched-v2 scene_graph at {v2_sg_path} has scene="
                f"{v2_scene_graph.get('scene')!r}; expected {bundle.scene_id!r}"
            )
        objects = v2_scene_graph.get("objects", [])
        entities = [_build_entity_v2(o) for o in objects]
        _assert_unique_uids(entities, v2_sg_path)

        raw_surfaces = v2_scene_graph.get("structural_surfaces", [])
        surfaces = [_build_structural_surface(s) for s in raw_surfaces]
        surface_uids = [s.surface_uid for s in surfaces]
        if len(set(surface_uids)) != len(surface_uids):
            dupes = [u for u in surface_uids if surface_uids.count(u) > 1]
            raise ValueError(
                f"surface_uid collision in {v2_sg_path}: duplicates={sorted(set(dupes))}"
            )

        runtime = time.perf_counter() - start
        return EntityArtifacts(
            schema_version=CURRENT_SCHEMA_VERSION,
            bundle_hash=_bundle_hash(
                bundle.representation_hash, self.name, self.version, dict(config.params),
                enriched_v2_content_hash=hashlib.sha256(v2_scene_graph_bytes).hexdigest(),
            ),
            scene_id=bundle.scene_id,
            frame=bundle.frame,
            representation_hash=bundle.representation_hash,
            extractor_name=self.name,
            extractor_version=self.version,
            entities=entities,
            structural_surfaces=surfaces,
            geometry_store_path=None,
            diagnostics=ExtractionDiagnostics(
                n_entities=len(entities),
                n_structural_surfaces=len(surfaces),
                runtime_seconds=runtime,
                coverage_score=1.0,
                notes=(
                    "Oracle wrapper around enriched-v2 importer output; "
                    "tight world-frame AABBs and structural surfaces populated."
                ),
            ),
            notes={
                "semantic_export_path": str(phase1_sg_path),
                "enriched_v2_path": str(self.enriched_v2_path),
                "z_translation_already_applied": True,
                "structural_surfaces_status": "habitat_label_path",
            },
        )

    def capabilities(self) -> InstanceExtractorCapabilities:
        use_v2 = self.enriched_v2_path is not None
        return InstanceExtractorCapabilities(
            label_vocab=None,
            provides_embeddings=False,
            provides_oriented_bboxes=use_v2,
            provides_structural_surfaces=use_v2,
            extractor_class_hint="all",
        )


def _assert_unique_uids(entities: list[EntityArtifact], src_path: Path) -> None:
    uids = [e.identity.object_uid for e in entities]
    if len(set(uids)) != len(uids):
        duplicates = [u for u in uids if uids.count(u) > 1]
        raise ValueError(
            f"object_uid collision in extractor input {src_path}: "
            f"duplicates={sorted(set(duplicates))}"
        )

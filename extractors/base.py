"""InstanceExtractor Protocol, EntityArtifacts, StructuralSurface — §5.3.

Extractor diagnostics are observational. They do NOT authorize the executor
to call missing results "empty"; that decision uses an externally
calibrated CompletenessProfile (reasoner/base.py) passed via
ExecutionContext.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from common.types import JSON, OrientedBBox, Plane, SceneFrame, Vec3
from representations.base import Channel, SceneRepresentation


@dataclass(frozen=True)
class EntityIdentity:
    """object_uid is immutable WITHIN this EntityArtifacts bundle only.
    Cross-bundle identity requires an explicit BundleCorrespondence."""
    object_uid: str
    display_label: str
    aliases: list[str] = field(default_factory=list)
    source_instance_ref: str = ""


@dataclass(frozen=True)
class SemanticHypothesis:
    label: str
    confidence: float
    source: str


@dataclass(frozen=True)
class StructuralSurface:
    """First-class non-entity surface (floor / wall / ceiling). Edges may
    target a StructuralSurface via GraphRef(kind="surface", uid=surface_uid).
    surface_uid is stable within this EntityArtifacts bundle only.

    `source` (P2.06 / A7) records how the surface was produced. Canonical
    Phase 2 emits 'habitat_label' for floor/wall/ceiling instances drawn
    from the Habitat semantic JSON. 'mesh_ransac' is reserved for future
    RANSAC-fit fallback. 'synth_bbox_fallback' marks the labeled
    experiment-only path; it is excluded from the canonical exit gates.
    'mesh_region_fit' is the C3.0-S deterministic raw-mesh estimator."""
    surface_uid: str
    surface_type: Literal["floor", "wall", "ceiling"]
    plane: Plane
    polygon: list[Vec3] | None
    confidence: float
    source: Literal[
        "habitat_label", "mesh_ransac", "mesh_region_fit",
        "synth_bbox_fallback",
    ]


@dataclass(frozen=True)
class EntityArtifact:
    identity: EntityIdentity
    bbox_aabb: tuple[Vec3, Vec3]
    bbox_obb: OrientedBBox | None
    centroid: Vec3
    geometry_handle: str | None
    semantic_hypotheses: list[SemanticHypothesis] = field(default_factory=list)
    embedding: np.ndarray | None = None
    extraction_diagnostics: dict[str, JSON] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionDiagnostics:
    n_entities: int
    n_structural_surfaces: int
    runtime_seconds: float
    coverage_score: float | None
    notes: str


@dataclass(frozen=True)
class InstanceExtractorCapabilities:
    label_vocab: list[str] | None
    provides_embeddings: bool
    provides_oriented_bboxes: bool
    provides_structural_surfaces: bool
    extractor_class_hint: Literal["furniture_only", "small_objects", "all"]
    # extractor_class_hint is observational. The executor's empty-vs-unknown
    # decision uses CompletenessProfile passed via ExecutionContext.


@dataclass(frozen=True)
class InstanceExtractorConfig:
    name: str
    version: str
    params: dict[str, JSON] = field(default_factory=dict)


@dataclass(frozen=True)
class EntityArtifacts:
    schema_version: int            # bump on breaking changes; loader checks
    bundle_hash: str
    scene_id: str
    frame: SceneFrame
    representation_hash: str
    extractor_name: str
    extractor_version: str
    entities: list[EntityArtifact]
    structural_surfaces: list[StructuralSurface]
    geometry_store_path: Path | None
    diagnostics: ExtractionDiagnostics
    notes: dict[str, JSON] = field(default_factory=dict)


@dataclass(frozen=True)
class BundleCorrespondence:
    """Maps entities and surfaces across two EntityArtifacts bundles.
    Produced by tools in eval/, never by extractors."""
    source_bundle_hash: str
    target_bundle_hash: str
    entity_pairs: list[tuple[str, str]]
    surface_pairs: list[tuple[str, str]]
    method: str
    score: dict[str, float]
    unmatched_source_entities: list[str]
    unmatched_target_entities: list[str]
    unmatched_source_surfaces: list[str]
    unmatched_target_surfaces: list[str]


class InstanceExtractor(Protocol):
    name: str
    version: str
    required_channels: frozenset[Channel]

    def extract(
        self,
        representation: SceneRepresentation,
        config: InstanceExtractorConfig,
    ) -> EntityArtifacts: ...

    def capabilities(self) -> InstanceExtractorCapabilities: ...

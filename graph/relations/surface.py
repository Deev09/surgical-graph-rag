"""Surface-proximity relation extractor — NEAR_SURFACE.

Emits NEAR_SURFACE(entity, surface) edges where `surface` is a
StructuralSurface from the EntityArtifacts bundle and the entity's
AABB-to-plane distance is within a per-surface-type threshold.

Edge semantics (A4 + P2.07):
  - distance = bbox_to_plane(entity.bbox_aabb, surface.plane), non-negative,
    0 when the plane intersects the box,
  - emit iff distance <= threshold_for(surface.surface_type).

Provenance on each edge (analogous to sparse_v2):
  - extractor = "near_surface", version = "0.1",
  - target = GraphRef(kind="surface", uid=surface_uid),
  - evidence["distance_metric"] = "bbox_to_plane",
  - evidence["distance_m"] = computed distance,
  - evidence["surface_type"] = "floor"|"wall"|"ceiling",
  - evidence["threshold_m"] = threshold used,
  - evidence["source"] = surface.source.

Canonical policy (A3, A7): the extractor refuses to emit NEAR_SURFACE
against surfaces whose `source == "synth_bbox_fallback"` unless
`config.include_synth_fallback` is True. Skipped surfaces are recorded
as EdgeRejection entries with rejected_reason="surface_source_excluded".

Isolation from global builder density policy (P2.09 sign-off): this
module ships the extractor and tests, but is NOT yet wired into any
default GraphBuilder run. P2.10 makes the version-aware density-cap
decision before integration; the combined sparse-v2 graph already
exceeds 14/entity per the phase2_sparse_v2_telemetry artifact.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

from extractors.base import EntityArtifact, EntityArtifacts, StructuralSurface
from geometry.surface_distance import bbox_to_plane
from graph.relations.base import (
    RelationExtractorConfig, RelationExtractorDiagnostics,
    count_logical_edges, make_edge_id, make_entity_ref, make_surface_ref,
)
from graph.schema import Edge, EdgeRejection, EdgeType


NEAR_SURFACE_TYPES: frozenset[EdgeType] = frozenset({"NEAR_SURFACE"})

DEFAULT_FLOOR_THRESHOLD_M = 0.05
DEFAULT_WALL_THRESHOLD_M = 0.30
DEFAULT_CEILING_THRESHOLD_M = 0.10


@dataclass(frozen=True)
class SurfaceProximityConfig:
    """Per-surface-type thresholds; provisional Replica-calibrated defaults
    per Q4. include_synth_fallback gates the synth-bbox-fallback policy
    (A3); leave False on the canonical path."""
    mode: Literal["sparse"] = "sparse"
    floor_threshold_m: float = DEFAULT_FLOOR_THRESHOLD_M
    wall_threshold_m: float = DEFAULT_WALL_THRESHOLD_M
    ceiling_threshold_m: float = DEFAULT_CEILING_THRESHOLD_M
    include_synth_fallback: bool = False


def _threshold_for(config: SurfaceProximityConfig, surface_type: str) -> float:
    if surface_type == "floor":
        return config.floor_threshold_m
    if surface_type == "wall":
        return config.wall_threshold_m
    if surface_type == "ceiling":
        return config.ceiling_threshold_m
    raise ValueError(f"unknown surface_type {surface_type!r}")


def _validate_config(config: SurfaceProximityConfig) -> None:
    if config.mode != "sparse":
        raise ValueError(f"unknown mode {config.mode!r}; supported: 'sparse'")
    for name in (
        "floor_threshold_m",
        "wall_threshold_m",
        "ceiling_threshold_m",
    ):
        value = getattr(config, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")


def _build_near_surface_edge(
    entity: EntityArtifact,
    surface: StructuralSurface,
    distance: float,
    threshold: float,
    *,
    extractor: str,
    version: str,
) -> Edge:
    source = make_entity_ref(entity.identity.object_uid)
    target = make_surface_ref(surface.surface_uid)
    return Edge(
        edge_id=make_edge_id(extractor, version, source, "NEAR_SURFACE", target),
        source=source,
        type="NEAR_SURFACE",
        target=target,
        frame="world",
        weight=1.0,
        confidence=1.0,
        extractor=extractor,
        extractor_version=version,
        evidence={
            "distance_m": distance,
            "distance_metric": "bbox_to_plane",
            "threshold_m": threshold,
            "surface_type": surface.surface_type,
            "source": surface.source,
        },
    )


class SurfaceProximityExtractor:
    name: str = "near_surface"
    version: str = "0.1"
    edge_types: frozenset[EdgeType] = NEAR_SURFACE_TYPES

    def extract(
        self,
        entities: EntityArtifacts,
        config: RelationExtractorConfig,
    ) -> tuple[list[Edge], RelationExtractorDiagnostics]:
        if not isinstance(config, SurfaceProximityConfig):
            raise TypeError(
                f"SurfaceProximityExtractor requires SurfaceProximityConfig, "
                f"got {type(config).__name__}"
            )
        _validate_config(config)
        start = time.perf_counter()
        edges: list[Edge] = []
        rejections: list[EdgeRejection] = []
        rejection_counts: dict[EdgeType, int] = {}
        max_rejection_samples = 64

        active_surfaces: list[StructuralSurface] = []
        excluded_surfaces: list[StructuralSurface] = []
        for surface in entities.structural_surfaces:
            if surface.source == "synth_bbox_fallback" and not config.include_synth_fallback:
                excluded_surfaces.append(surface)
                continue
            active_surfaces.append(surface)

        for entity in entities.entities:
            for surface in excluded_surfaces:
                rejection_counts["NEAR_SURFACE"] = (
                    rejection_counts.get("NEAR_SURFACE", 0) + 1
                )
                if len(rejections) < max_rejection_samples:
                    rejections.append(EdgeRejection(
                        source=make_entity_ref(entity.identity.object_uid),
                        type="NEAR_SURFACE",
                        target=make_surface_ref(surface.surface_uid),
                        extractor=self.name,
                        rejected_reason="surface_source_excluded",
                        evidence={
                            "source": surface.source,
                            "include_synth_fallback": False,
                            "policy": (
                                "canonical_extractor_excludes_synth_bbox_fallback"
                            ),
                        },
                    ))
            for surface in active_surfaces:
                threshold = _threshold_for(config, surface.surface_type)
                distance = bbox_to_plane(entity.bbox_aabb, surface.plane)
                if distance <= threshold:
                    edges.append(_build_near_surface_edge(
                        entity, surface, distance, threshold,
                        extractor=self.name, version=self.version,
                    ))
                else:
                    rejection_counts["NEAR_SURFACE"] = (
                        rejection_counts.get("NEAR_SURFACE", 0) + 1
                    )
                    if len(rejections) < max_rejection_samples:
                        rejections.append(EdgeRejection(
                            source=make_entity_ref(entity.identity.object_uid),
                            type="NEAR_SURFACE",
                            target=make_surface_ref(surface.surface_uid),
                            extractor=self.name,
                            rejected_reason="distance_exceeds_surface_threshold",
                            evidence={
                                "distance_m": distance,
                                "distance_metric": "bbox_to_plane",
                                "threshold_m": threshold,
                                "surface_type": surface.surface_type,
                                "source": surface.source,
                            },
                        ))

        counts: dict[EdgeType, int] = {"NEAR_SURFACE": len(edges)}
        runtime_ms = int((time.perf_counter() - start) * 1000)
        return edges, RelationExtractorDiagnostics(
            extractor=self.name,
            version=self.version,
            mode=config.mode,
            physical_edges_per_type=counts,
            physical_edges_total=len(edges),
            logical_edges_total=count_logical_edges(edges),
            rejections_per_type=rejection_counts,
            rejection_samples=rejections,
            runtime_ms=runtime_ms,
        )

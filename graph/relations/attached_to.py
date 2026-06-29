"""ATTACHED_TO relation extractor — wall-mounted objects (P7.01).

A gated view over two existing Phase 4/5 predicates, inventing no new thresholds:

    ATTACHED_TO(entity -> wall)  iff
        wall_contact(entity, wall)            # reuse CONTACTS_SURFACE geometry
        AND NOT (entity rests on any floor)   # reuse ON_SURFACE geometry

"Attached" = held by the wall, not merely touching it at floor level. The
floor-rest disqualifier is what separates a wall-mounted sconce/vent/sink from a
floor cabinet shoved against the wall. "Elevated" is therefore emergent (= not
floor-resting), recorded as evidence, never a gate constant.

Self-contained: the extractor runs BOTH predicates itself (it has the wall and
floor surfaces + the entity box), so it is order-independent — it does not depend
on ON_SURFACE / CONTACTS_SURFACE having run first.

Edge semantics:
  - source = entity ref, target = wall surface ref (same direction as
    CONTACTS_SURFACE),
  - emit iff wall_contact AND not floor-supported.

Evidence: the full wall_contact evidence (wall_capable, on_interior_side,
sd_centroid_m, wall_gap_m, ..., up) plus floor_supported=false,
floor_rest_best_bottom_gap_m (nearest floor rest-gap that still failed), and the
inherited thresholds. `up` is normalized tuple -> list (the P4.06 serde lesson).

Isolation: ships extractor + tests, NOT wired into any default GraphBuilder run
(same as ON_SURFACE / CONTACTS_SURFACE / ON_ENTITY_SURFACE).

Known limits (see docs/phase7_plan.md): flush-mounted pictures are elevated but
not wall-contacting (missed); v1 disqualifies floor-rest only, so a tabletop
object pushed flush against a wall could read as attached (rare).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from extractors.base import EntityArtifact, EntityArtifacts, StructuralSurface
from geometry.rest_contact import RestContactConfig, rest_contact
from geometry.wall_contact import WallContactConfig, wall_contact
from graph.relations import contacts_surface as _cs
from graph.relations import on_surface as _os
from graph.relations.base import (
    RelationExtractorConfig, RelationExtractorDiagnostics,
    count_logical_edges, make_edge_id, make_entity_ref, make_surface_ref,
)
from graph.schema import Edge, EdgeRejection, EdgeType


ATTACHED_TO_TYPES: frozenset[EdgeType] = frozenset({"ATTACHED_TO"})

ATTACHED_TO_VERSION = "0.1"


@dataclass(frozen=True)
class AttachedToConfig:
    """Wall-contact gate + floor-rest disqualifier. All thresholds are inherited
    verbatim from CONTACTS_SURFACE (wall) and ON_SURFACE (floor); Phase 7 invents
    none. Field sanity is delegated to the two geometry configs."""
    mode: Literal["sparse"] = "sparse"
    # wall-contact gate (CONTACTS_SURFACE defaults)
    contact_threshold_m: float = _cs.DEFAULT_CONTACT_THRESHOLD_M
    penetration_tolerance_m: float = _cs.DEFAULT_PENETRATION_TOLERANCE_M
    max_wall_tilt_deg: float = _cs.DEFAULT_MAX_WALL_TILT_DEG
    footprint_tolerance_m: float = _cs.DEFAULT_FOOTPRINT_TOLERANCE_M
    # floor-rest disqualifier (ON_SURFACE defaults)
    floor_contact_threshold_m: float = _os.DEFAULT_CONTACT_THRESHOLD_M
    floor_penetration_tolerance_m: float = _os.DEFAULT_PENETRATION_TOLERANCE_M
    floor_max_tilt_deg: float = _os.DEFAULT_MAX_TILT_DEG
    floor_footprint_tolerance_m: float = _os.DEFAULT_FOOTPRINT_TOLERANCE_M
    include_synth_fallback: bool = False

    def __post_init__(self) -> None:
        # Delegate field sanity to the two geometry configs (raise on bad values).
        self._wall_config()
        self._rest_config()
        if self.mode != "sparse":
            raise ValueError(f"unknown mode {self.mode!r}; supported: 'sparse'")

    def _wall_config(self) -> WallContactConfig:
        return WallContactConfig(
            contact_threshold_m=self.contact_threshold_m,
            penetration_tolerance_m=self.penetration_tolerance_m,
            max_wall_tilt_deg=self.max_wall_tilt_deg,
            footprint_tolerance_m=self.footprint_tolerance_m,
        )

    def _rest_config(self) -> RestContactConfig:
        return RestContactConfig(
            contact_threshold_m=self.floor_contact_threshold_m,
            penetration_tolerance_m=self.floor_penetration_tolerance_m,
            max_tilt_deg=self.floor_max_tilt_deg,
            footprint_tolerance_m=self.floor_footprint_tolerance_m,
        )


def _active(surfaces, surface_type, include_synth) -> list[StructuralSurface]:
    out = []
    for s in surfaces:
        if s.surface_type != surface_type:
            continue
        if s.source == "synth_bbox_fallback" and not include_synth:
            continue
        if s.polygon is None:
            continue
        out.append(s)
    return out


def _build_edge(entity, surface, evidence, *, extractor, version) -> Edge:
    source = make_entity_ref(entity.identity.object_uid)
    target = make_surface_ref(surface.surface_uid)
    return Edge(
        edge_id=make_edge_id(extractor, version, source, "ATTACHED_TO", target),
        source=source,
        type="ATTACHED_TO",
        target=target,
        frame="world",
        weight=1.0,
        confidence=1.0,
        extractor=extractor,
        extractor_version=version,
        evidence=evidence,
    )


class AttachedToExtractor:
    name: str = "attached_to"
    version: str = ATTACHED_TO_VERSION
    edge_types: frozenset[EdgeType] = ATTACHED_TO_TYPES

    def extract(
        self,
        entities: EntityArtifacts,
        config: RelationExtractorConfig,
    ) -> tuple[list[Edge], RelationExtractorDiagnostics]:
        if not isinstance(config, AttachedToConfig):
            raise TypeError(
                f"AttachedToExtractor requires AttachedToConfig, "
                f"got {type(config).__name__}"
            )
        start = time.perf_counter()
        wall_cfg = config._wall_config()
        rest_cfg = config._rest_config()
        gravity = entities.frame.gravity
        edges: list[Edge] = []
        rejections: list[EdgeRejection] = []
        rejection_counts: dict[EdgeType, int] = {}
        max_rejection_samples = 64

        def record_rejection(entity, surface, reason: str, evidence: dict) -> None:
            rejection_counts["ATTACHED_TO"] = rejection_counts.get("ATTACHED_TO", 0) + 1
            if len(rejections) < max_rejection_samples:
                rejections.append(EdgeRejection(
                    source=make_entity_ref(entity.identity.object_uid),
                    type="ATTACHED_TO",
                    target=make_surface_ref(surface.surface_uid),
                    extractor=self.name,
                    rejected_reason=reason,
                    evidence=evidence,
                ))

        walls = _active(entities.structural_surfaces, "wall", config.include_synth_fallback)
        floors = _active(entities.structural_surfaces, "floor", config.include_synth_fallback)

        for entity in entities.entities:
            # Floor-rest disqualifier: is the entity STANDING on any floor?
            # This is rest_contact's on_surface conjunction with the penetration
            # floor removed: an object can sink arbitrarily far through the floor
            # plane (a floor lamp whose bbox dips below z=0) and still be
            # floor-standing, not wall-attached. We require it to be above the
            # floor (centroid_on_support_side), over it (footprint_ok), and with
            # its bottom no higher than the floor contact threshold. The elevated
            # sconce (bottom far above the floor) fails the last clause and is NOT
            # disqualified -- that is what makes it attachable.
            floor_supported = False
            best_floor_gap: float | None = None  # bottom height above nearest floor it sits over
            for f in floors:
                fr = rest_contact(
                    entity.bbox_aabb, entity.centroid,
                    f.plane, f.polygon, gravity, rest_cfg,
                )
                ev = fr.evidence
                gap = ev.get("bottom_gap_m")
                # only floors the object is above (centroid on support side) and
                # over (footprint overlap) are relevant -- those are floors below it
                above_and_over = (
                    bool(ev.get("support_capable"))
                    and bool(ev.get("centroid_on_support_side"))
                    and bool(ev.get("footprint_ok"))
                    and gap is not None
                )
                if above_and_over:
                    if best_floor_gap is None or float(gap) < best_floor_gap:
                        best_floor_gap = float(gap)
                    if float(gap) <= rest_cfg.contact_threshold_m:
                        floor_supported = True

            for surface in walls:
                result = wall_contact(
                    entity.bbox_aabb, entity.centroid,
                    surface.plane, surface.polygon, gravity, wall_cfg,
                )
                evidence = dict(result.evidence)
                if isinstance(evidence.get("up"), tuple):
                    evidence["up"] = list(evidence["up"])
                evidence["surface_type"] = surface.surface_type
                evidence["source"] = surface.source
                evidence["polygon_clip_required"] = True
                evidence["floor_supported"] = floor_supported
                evidence["floor_rest_best_bottom_gap_m"] = best_floor_gap
                evidence["bottom_elevation_m"] = best_floor_gap
                evidence["contact_threshold_m"] = config.contact_threshold_m
                evidence["penetration_tolerance_m"] = config.penetration_tolerance_m
                evidence["max_wall_tilt_deg"] = config.max_wall_tilt_deg
                evidence["footprint_tolerance_m"] = config.footprint_tolerance_m

                if result.contacts_surface and not floor_supported:
                    edges.append(_build_edge(
                        entity, surface, evidence,
                        extractor=self.name, version=self.version,
                    ))
                elif result.contacts_surface and floor_supported:
                    rej = dict(evidence)
                    rej["disqualified_reason"] = "floor_supported"
                    record_rejection(entity, surface, "attached_to_floor_supported", rej)
                else:
                    rej = dict(evidence)
                    rej["failed_clauses"] = result.failed_clauses
                    record_rejection(entity, surface, "wall_contact_clauses_failed", rej)

        counts: dict[EdgeType, int] = {"ATTACHED_TO": len(edges)}
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

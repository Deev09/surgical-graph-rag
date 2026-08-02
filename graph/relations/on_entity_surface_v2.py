"""ON_ENTITY_SURFACE v2 — semantics_v2 track support (D2 + D3).

Protocol: docs/semantics_v2_track_protocol.md (signed off 2026-08-02;
constants FROZEN). Opt-in; the frozen v1 `on_entity_surface.py` is
untouched and remains the default everywhere.

D3 — allowlist v2: the v1 allowlist plus cabinet, nightstand, bed.

D2 — two disjuncts:
  1. TOP-REST: the unchanged v1 extractor logic (delegated verbatim),
     run with the v2 allowlist.
  2. CONTAINED-REST: entity E is on supporter S iff
       - S's class is in the v2 allowlist and S != E;
       - E's XY footprint CENTER lies inside S's XY footprint;
       - E's footprint area <= 0.5 x S's footprint area;
       - S.bottom <= E.bottom <= S.top + band (band = the v1 top-rest
         contact threshold, inherited);
       - E is NOT floor-supported (frozen ON_SURFACE rest-contact
         geometry, as attached_to v1 uses it; declared limitation:
         floor-touching draped textiles remain misses).
     Supported-class policy (frozen at review): E may be ANY
     non-structural entity including allowlist classes; supporters are
     allowlist entities only; chains are permitted; E is assigned to
     the SMALLEST-footprint qualifying supporter (tie: lower uid); a
     contained-rest edge is skipped when the top-rest pass already
     emitted the same (E, S) edge.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from extractors.base import EntityArtifact, EntityArtifacts
from extractors.entity_surfaces import (
    DEFAULT_SUPPORT_CLASS_ALLOWLIST, support_class_for_entity,
)
from geometry.rest_contact import RestContactConfig, rest_contact
from graph.relations import on_surface as _os
from graph.relations.attached_to import _active
from graph.relations.base import (
    RelationExtractorConfig, RelationExtractorDiagnostics,
    count_logical_edges, make_edge_id, make_entity_ref,
)
from graph.relations.on_entity_surface import (
    OnEntitySurfaceConfig, OnEntitySurfaceExtractor,
)
from graph.schema import Edge, EdgeType

ON_ENTITY_SURFACE_V2_VERSION = "2.0"
V2_SUPPORT_CLASS_ALLOWLIST: tuple[str, ...] = (
    DEFAULT_SUPPORT_CLASS_ALLOWLIST + ("cabinet", "nightstand", "bed"))


@dataclass(frozen=True)
class OnEntitySurfaceV2Config:
    """D2/D3 frozen constants. Top-rest thresholds are the v1 defaults;
    contained-rest adds only the declared footprint fraction."""
    mode: Literal["sparse"] = "sparse"
    support_class_allowlist: tuple[str, ...] = field(
        default_factory=lambda: V2_SUPPORT_CLASS_ALLOWLIST)
    footprint_area_max_frac: float = 0.5      # D2 contained-rest
    # floor-support disqualifier (frozen ON_SURFACE geometry defaults)
    floor_contact_threshold_m: float = _os.DEFAULT_CONTACT_THRESHOLD_M
    floor_penetration_tolerance_m: float = _os.DEFAULT_PENETRATION_TOLERANCE_M
    floor_max_tilt_deg: float = _os.DEFAULT_MAX_TILT_DEG
    floor_footprint_tolerance_m: float = _os.DEFAULT_FOOTPRINT_TOLERANCE_M

    def __post_init__(self) -> None:
        if not (0 < self.footprint_area_max_frac <= 1):
            raise ValueError("footprint_area_max_frac must be in (0, 1]")
        self._v1_config()
        self._rest_config()

    def _v1_config(self) -> OnEntitySurfaceConfig:
        return OnEntitySurfaceConfig(
            support_class_allowlist=self.support_class_allowlist)

    def _rest_config(self) -> RestContactConfig:
        return RestContactConfig(
            contact_threshold_m=self.floor_contact_threshold_m,
            penetration_tolerance_m=self.floor_penetration_tolerance_m,
            max_tilt_deg=self.floor_max_tilt_deg,
            footprint_tolerance_m=self.floor_footprint_tolerance_m,
        )


def _footprint_area(e: EntityArtifact) -> float:
    lo, hi = e.bbox_aabb
    return max(hi[0] - lo[0], 0.0) * max(hi[1] - lo[1], 0.0)


def _center_inside_xy(e: EntityArtifact, s: EntityArtifact) -> bool:
    cx, cy = e.centroid[0], e.centroid[1]
    lo, hi = s.bbox_aabb
    return lo[0] <= cx <= hi[0] and lo[1] <= cy <= hi[1]


class OnEntitySurfaceV2Extractor:
    name: str = "on_entity_surface_v2"
    version: str = ON_ENTITY_SURFACE_V2_VERSION
    edge_types: frozenset[EdgeType] = frozenset({"ON_ENTITY_SURFACE"})

    def extract(
        self,
        entities: EntityArtifacts,
        config: RelationExtractorConfig,
    ) -> tuple[list[Edge], RelationExtractorDiagnostics]:
        if not isinstance(config, OnEntitySurfaceV2Config):
            raise TypeError(
                "OnEntitySurfaceV2Extractor requires OnEntitySurfaceV2Config")
        start = time.perf_counter()

        # disjunct 1: the unchanged v1 extractor with the v2 allowlist
        v1_edges, v1_diag = OnEntitySurfaceExtractor().extract(
            entities, config._v1_config())
        existing_pairs = {(e.source.uid, e.target.uid) for e in v1_edges}

        # floor-support disqualifier inputs
        rest_cfg = config._rest_config()
        gravity = entities.frame.gravity
        floors = _active(entities.structural_surfaces, "floor", False)

        def floor_supported(e: EntityArtifact) -> bool:
            for f in floors:
                ev = rest_contact(e.bbox_aabb, e.centroid, f.plane,
                                  f.polygon, gravity, rest_cfg).evidence
                gap = ev.get("bottom_gap_m")
                if (bool(ev.get("support_capable"))
                        and bool(ev.get("centroid_on_support_side"))
                        and bool(ev.get("footprint_ok"))
                        and gap is not None
                        and float(gap) <= rest_cfg.contact_threshold_m):
                    return True
            return False

        supporters = [
            (s, support_class_for_entity(s, config.support_class_allowlist))
            for s in entities.entities]
        supporters = [(s, cls) for s, cls in supporters if cls is not None]
        # top-rest contact band inherited from the v1 config
        band = config._v1_config().contact_threshold_m

        contained_edges: list[Edge] = []
        for e in entities.entities:
            if floor_supported(e):
                continue
            e_area = _footprint_area(e)
            e_bottom = e.bbox_aabb[0][2]
            candidates = []
            for s, cls in supporters:
                if s.identity.object_uid == e.identity.object_uid:
                    continue
                s_lo, s_hi = s.bbox_aabb
                if not _center_inside_xy(e, s):
                    continue
                s_area = _footprint_area(s)
                if s_area <= 0 or e_area > config.footprint_area_max_frac * s_area:
                    continue
                if not (s_lo[2] <= e_bottom <= s_hi[2] + band):
                    continue
                candidates.append((s_area, s.identity.object_uid, s, cls))
            if not candidates:
                continue
            candidates.sort(key=lambda c: (c[0], c[1]))
            s_area, _, s, cls = candidates[0]
            pair = (e.identity.object_uid, s.identity.object_uid)
            if pair in existing_pairs:
                continue
            source = make_entity_ref(e.identity.object_uid)
            target = make_entity_ref(s.identity.object_uid)
            contained_edges.append(Edge(
                edge_id=make_edge_id(self.name, self.version, source,
                                     "ON_ENTITY_SURFACE", target),
                source=source, type="ON_ENTITY_SURFACE", target=target,
                frame="world", weight=1.0, confidence=1.0,
                extractor=self.name, extractor_version=self.version,
                evidence={
                    "semantics_track": "v2",
                    "disjunct": "contained_rest",
                    "owner_entity_uid": s.identity.object_uid,
                    "owner_class": cls,
                    "supporter_footprint_area_m2": round(s_area, 4),
                    "entity_footprint_area_m2": round(e_area, 4),
                    "footprint_area_max_frac": config.footprint_area_max_frac,
                    "band_m": band,
                    "n_qualifying_supporters": len(candidates),
                },
            ))

        edges = v1_edges + contained_edges
        runtime_ms = int((time.perf_counter() - start) * 1000)
        return edges, RelationExtractorDiagnostics(
            extractor=self.name, version=self.version, mode=config.mode,
            physical_edges_per_type={"ON_ENTITY_SURFACE": len(edges)},
            physical_edges_total=len(edges),
            logical_edges_total=count_logical_edges(edges),
            rejections_per_type=dict(v1_diag.rejections_per_type),
            rejection_samples=list(v1_diag.rejection_samples),
            runtime_ms=runtime_ms,
        )

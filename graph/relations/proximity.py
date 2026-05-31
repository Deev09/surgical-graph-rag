"""Proximity relation extractor (compat + sparse).

Compat: faithful port of relations/compute.py NEAR logic.
  - LEGACY_NEAR_THRESHOLD = 1.0 (centroid distance, strict less-than)
  - Each unordered pair within threshold contributes TWO edges:
    NEAR(a, b) and NEAR(b, a). This duplication is preserved exactly
    for legacy reproduction.

Sparse: emits NEAR ONCE per unordered pair within threshold, with
evidence flag symmetric=True. Centroid distance only — surface-to-
surface NEAR depends on world-frame OBBs that arrive in Phase 2
per phase0_design.md §11.1.

FAR is intentionally not emitted by either mode. FAR is a query-time
operator over a centroid index (phase0_design.md §6).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from math import sqrt
from typing import Literal

from extractors.base import EntityArtifact, EntityArtifacts
from graph.relations.base import (
    RelationExtractorConfig, RelationExtractorDiagnostics,
    count_logical_edges, make_edge_id, make_entity_ref,
)
from graph.schema import Edge, EdgeRejection, EdgeType


LEGACY_NEAR_THRESHOLD = 1.0   # frozen for compat; do not change

PROXIMITY_TYPES: frozenset[EdgeType] = frozenset({"NEAR"})


@dataclass(frozen=True)
class ProximityConfig:
    mode: Literal["compat", "sparse"]
    sparse_near_threshold: float = 1.0   # centroid distance, strict less-than


def _euclid_3d(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _build_near(
    src: EntityArtifact,
    tgt: EntityArtifact,
    *,
    extractor: str,
    version: str,
    evidence: dict,
) -> Edge:
    source = make_entity_ref(src.identity.object_uid)
    target = make_entity_ref(tgt.identity.object_uid)
    return Edge(
        edge_id=make_edge_id(extractor, version, source, "NEAR", target),
        source=source,
        type="NEAR",
        target=target,
        frame="world",
        weight=1.0,
        confidence=1.0,
        extractor=extractor,
        extractor_version=version,
        evidence=evidence,
    )


# ----- compat path -----

def extract_compat(
    entities: list[EntityArtifact],
    *,
    extractor: str,
    version: str,
) -> tuple[list[Edge], dict[EdgeType, int]]:
    """Legacy NEAR. Emits both NEAR(a,b) and NEAR(b,a) per unordered
    pair within LEGACY_NEAR_THRESHOLD. Iteration order matches the
    legacy compute_relations loop."""
    edges: list[Edge] = []
    counts: dict[EdgeType, int] = {"NEAR": 0}
    n = len(entities)
    for i in range(n):
        for j in range(i + 1, n):
            a = entities[i]
            b = entities[j]
            distance = _euclid_3d(a.centroid, b.centroid)
            if distance < LEGACY_NEAR_THRESHOLD:
                evidence = {"distance_m": distance}
                edges.append(_build_near(a, b, extractor=extractor, version=version, evidence=evidence))
                edges.append(_build_near(b, a, extractor=extractor, version=version, evidence=evidence))
                counts["NEAR"] += 2
    return edges, counts


# ----- sparse path -----

def extract_sparse(
    entities: list[EntityArtifact],
    *,
    extractor: str,
    version: str,
    sparse_near_threshold: float,
) -> tuple[list[Edge], dict[EdgeType, int], list[EdgeRejection], dict[EdgeType, int]]:
    """Sparse NEAR. Emits ONCE per unordered pair within threshold with
    evidence symmetric=True."""
    edges: list[Edge] = []
    counts: dict[EdgeType, int] = {"NEAR": 0}
    rejections: list[EdgeRejection] = []
    rejection_counts: dict[EdgeType, int] = {}
    max_rejection_samples = 64
    n = len(entities)
    for i in range(n):
        for j in range(i + 1, n):
            a = entities[i]
            b = entities[j]
            distance = _euclid_3d(a.centroid, b.centroid)
            if distance < sparse_near_threshold:
                edges.append(_build_near(
                    a, b, extractor=extractor, version=version,
                    evidence={"distance_m": distance, "symmetric": True},
                ))
                counts["NEAR"] += 1
            else:
                rejection_counts["NEAR"] = rejection_counts.get("NEAR", 0) + 1
                if len(rejections) < max_rejection_samples:
                    rejections.append(EdgeRejection(
                        source=make_entity_ref(a.identity.object_uid),
                        type="NEAR",
                        target=make_entity_ref(b.identity.object_uid),
                        extractor=extractor,
                        rejected_reason="distance_exceeds_sparse_near_threshold",
                        evidence={
                            "distance_m": distance,
                            "sparse_near_threshold": sparse_near_threshold,
                        },
                    ))
    return edges, counts, rejections, rejection_counts


class ProximityExtractor:
    name: str = "proximity"
    version: str = "0.1"
    edge_types: frozenset[EdgeType] = PROXIMITY_TYPES

    def extract(
        self,
        entities: EntityArtifacts,
        config: RelationExtractorConfig,
    ) -> tuple[list[Edge], RelationExtractorDiagnostics]:
        if not isinstance(config, ProximityConfig):
            raise TypeError(
                f"ProximityExtractor requires ProximityConfig, "
                f"got {type(config).__name__}"
            )
        start = time.perf_counter()
        if config.mode == "compat":
            edges, counts = extract_compat(
                entities.entities, extractor=self.name, version=self.version,
            )
            rejections: list[EdgeRejection] = []
            rejection_counts: dict[EdgeType, int] = {}
        elif config.mode == "sparse":
            edges, counts, rejections, rejection_counts = extract_sparse(
                entities.entities, extractor=self.name, version=self.version,
                sparse_near_threshold=config.sparse_near_threshold,
            )
        else:
            raise ValueError(f"unknown mode {config.mode!r}")
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

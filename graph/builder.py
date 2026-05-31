"""GraphBuilder — orchestrates RelationExtractors over an EntityArtifacts
bundle.

Boring by design. Iterates extractor configs in the order the caller
provides, runs each extractor with its matching config, validates the
aggregated edge set, and assembles SceneGraphBundle + BuildDiagnostics.

Does NOT:
  - Score, rank, or evaluate (eval/ Phase 3 territory)
  - Pick which extractors to run (the caller decides)
  - Run two extractors with the same name (rejected)

DOES enforce:
  - Single mode across all runs: compat XOR sparse. Mixed is rejected.
  - No duplicate edge_id values across the aggregated set.
  - No duplicate edge keys (source, type, target) across the aggregated
    set. Identical-duplicate facts are rejected upfront per the initial
    policy choice; a future dedup policy can relax this with explicit
    tests.
  - Sparse-mode density guardrail:
      logical_edges_total / len(entities.entities) <= SPARSE_DENSITY_LIMIT
    (skipped vacuously when entity_count == 0).

Determinism:
  - Iteration order = caller-provided run order.
  - bundle_hash includes entity_bundle_hash, mode, and the ordered list
    of (extractor_name, extractor_version, config) tuples.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass

from extractors.base import EntityArtifact, EntityArtifacts
from graph.relations.base import (
    RelationExtractor, RelationExtractorConfig,
    RelationExtractorDiagnostics, count_logical_edges, edge_key,
)
from graph.schema import (
    BuildDiagnostics, Edge, EdgeRejection, EdgeType, Node, SceneGraphBundle,
)
from graph.serde import CURRENT_SCHEMA_VERSION


SPARSE_DENSITY_LIMIT = 14   # logical_edges / entity_count


class GraphBuildError(ValueError):
    """Raised when the builder cannot produce a valid SceneGraphBundle."""


@dataclass(frozen=True)
class ExtractorRun:
    """One extractor + its config. Run order in the caller's list is
    preserved by the builder."""
    extractor: RelationExtractor
    config: RelationExtractorConfig


def _node_from_entity(entity: EntityArtifact, *, entity_bundle: EntityArtifacts) -> Node:
    """Project an EntityArtifact onto a graph Node. The top semantic
    hypothesis (if any) provides label + confidence; the display label
    and aliases land in attributes for query-time lookup."""
    top = entity.semantic_hypotheses[0] if entity.semantic_hypotheses else None
    label = top.label if top else entity.identity.display_label
    label_conf = top.confidence if top else 1.0
    return Node(
        id=entity.identity.object_uid,
        label=label,
        label_confidence=label_conf,
        centroid=entity.centroid,
        bbox_aabb=entity.bbox_aabb,
        bbox_obb=entity.bbox_obb,
        embedding_ref=None,
        attributes={
            "display_label": entity.identity.display_label,
            "aliases": list(entity.identity.aliases),
        },
        provenance={
            "entity_extractor": entity_bundle.extractor_name,
            "entity_extractor_version": entity_bundle.extractor_version,
            "source_instance_ref": entity.identity.source_instance_ref,
        },
    )


def _validate_single_mode(runs: list[ExtractorRun]) -> str:
    if not runs:
        raise GraphBuildError("GraphBuilder requires at least one extractor run")
    modes = {r.config.mode for r in runs}
    if len(modes) > 1:
        raise GraphBuildError(
            f"GraphBuilder refuses mixed-mode runs; got modes={sorted(modes)}"
        )
    mode = next(iter(modes))
    if mode not in ("compat", "sparse"):
        raise GraphBuildError(f"unknown mode {mode!r} in extractor config")
    return mode


def _validate_unique_extractor_names(runs: list[ExtractorRun]) -> None:
    seen: set[str] = set()
    for r in runs:
        if r.extractor.name in seen:
            raise GraphBuildError(
                f"extractor {r.extractor.name!r} appears more than once in runs"
            )
        seen.add(r.extractor.name)


def _validate_no_duplicates(edges: list[Edge]) -> None:
    """Reject duplicate edge_id AND duplicate edge keys.

    Initial policy: any duplicate (whether identical or conflicting) is
    a configuration mistake and is rejected upfront. A relaxed dedup
    policy can replace this later with explicit tests."""
    seen_ids: dict[str, Edge] = {}
    seen_keys: dict[tuple, Edge] = {}
    for e in edges:
        if e.edge_id in seen_ids:
            prev = seen_ids[e.edge_id]
            raise GraphBuildError(
                f"duplicate edge_id {e.edge_id!r}: "
                f"emitted by {prev.extractor!r} and {e.extractor!r}"
            )
        seen_ids[e.edge_id] = e
        key = edge_key(e)
        if key in seen_keys:
            prev = seen_keys[key]
            raise GraphBuildError(
                f"duplicate edge key {key!r}: "
                f"emitted by {prev.extractor!r} (id {prev.edge_id!r}) and "
                f"{e.extractor!r} (id {e.edge_id!r})"
            )
        seen_keys[key] = e


def _enforce_sparse_density(
    logical_total: int,
    entity_count: int,
    limit: float = SPARSE_DENSITY_LIMIT,
) -> None:
    if entity_count == 0:
        return
    ratio = logical_total / entity_count
    if ratio > limit:
        raise GraphBuildError(
            f"sparse density guardrail exceeded: "
            f"logical_edges={logical_total} / entity_count={entity_count} "
            f"= {ratio:.3f} > {limit}"
        )


def _build_bundle_hash(
    entity_bundle_hash: str,
    mode: str,
    runs: list[ExtractorRun],
) -> str:
    """Hash includes entity bundle, mode, and the ordered list of
    (extractor_name, extractor_version, config) tuples. Run order is
    significant: a different order produces a different hash, which
    truthfully reflects that the inputs differ."""
    runs_payload = [
        {
            "extractor_name": r.extractor.name,
            "extractor_version": r.extractor.version,
            "config": asdict(r.config),
        }
        for r in runs
    ]
    payload = json.dumps(
        {
            "entity_bundle_hash": entity_bundle_hash,
            "mode": mode,
            "runs": runs_payload,
        },
        sort_keys=True,
    )
    return f"graph_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def build_graph(
    entities: EntityArtifacts,
    runs: list[ExtractorRun],
) -> tuple[SceneGraphBundle, BuildDiagnostics]:
    """Orchestrate extractors over an EntityArtifacts bundle.

    Validation order:
      1. At least one run; single mode; unique extractor names.
      2. Run extractors in caller-provided order.
      3. Reject duplicate edge_ids and duplicate edge keys.
      4. Enforce sparse density guardrail (sparse mode only).
      5. Assemble nodes and edges into a SceneGraphBundle.
    """
    mode = _validate_single_mode(runs)
    _validate_unique_extractor_names(runs)

    all_edges: list[Edge] = []
    per_extractor: list[RelationExtractorDiagnostics] = []
    runtime_ms_per_extractor: dict[str, int] = {}
    extractor_versions: dict[str, str] = {}
    edges_emitted_per_type: dict[EdgeType, int] = {}
    rejections_per_type: dict[EdgeType, int] = {}
    rejection_samples: list[EdgeRejection] = []

    for run in runs:
        extractor_versions[run.extractor.name] = run.extractor.version
        family_edges, family_diag = run.extractor.extract(entities, run.config)
        per_extractor.append(family_diag)
        runtime_ms_per_extractor[family_diag.extractor] = family_diag.runtime_ms
        for t, c in family_diag.physical_edges_per_type.items():
            edges_emitted_per_type[t] = edges_emitted_per_type.get(t, 0) + c
        for t, c in family_diag.rejections_per_type.items():
            rejections_per_type[t] = rejections_per_type.get(t, 0) + c
        rejection_samples.extend(family_diag.rejection_samples)
        all_edges.extend(family_edges)

    _validate_no_duplicates(all_edges)

    physical_total = len(all_edges)
    logical_total = count_logical_edges(all_edges)

    if mode == "sparse":
        _enforce_sparse_density(logical_total, len(entities.entities))

    nodes = [_node_from_entity(e, entity_bundle=entities) for e in entities.entities]
    surface_uids = sorted({
        ref.uid
        for e in all_edges
        for ref in (e.source, e.target)
        if ref.kind == "surface"
    })

    bundle = SceneGraphBundle(
        schema_version=CURRENT_SCHEMA_VERSION,
        bundle_hash=_build_bundle_hash(entities.bundle_hash, mode, runs),
        scene_id=entities.scene_id,
        frame=entities.frame,
        entity_bundle_hash=entities.bundle_hash,
        nodes=nodes,
        edges=all_edges,
        structural_surface_refs=surface_uids,
    )
    diagnostics = BuildDiagnostics(
        extractor_versions=extractor_versions,
        edges_emitted_per_type=edges_emitted_per_type,
        rejections_per_type=rejections_per_type,
        rejection_samples=rejection_samples,
        runtime_ms_per_extractor=runtime_ms_per_extractor,
        per_extractor=per_extractor,
        physical_edges_total=physical_total,
        logical_edges_total=logical_total,
        mode=mode,
    )
    return bundle, diagnostics

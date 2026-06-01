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
from dataclasses import MISSING, asdict, dataclass, fields
from typing import Literal

from extractors.base import EntityArtifact, EntityArtifacts, StructuralSurface
from graph.relations.base import (
    RelationExtractor, RelationExtractorConfig,
    RelationExtractorDiagnostics, count_logical_edges, edge_key,
)
from graph.schema import (
    BuildDiagnostics, Edge, EdgeRejection, EdgeType, Node, SceneGraphBundle,
    SurfaceRecord,
)
from graph.serde import CURRENT_SCHEMA_VERSION


SPARSE_DENSITY_LIMIT = 14   # Phase 1 sparse-v1 only — telemetry-only for Phase 2 candidates

DensityPolicy = Literal["phase1_block", "phase2_telemetry_only"]
# - "phase1_block": sparse-mode build raises GraphBuildError when
#   logical_edges / entity_count > SPARSE_DENSITY_LIMIT. Default; matches
#   Phase 1 behavior.
# - "phase2_telemetry_only": sparse-mode build records the ratio in
#   BuildDiagnostics.density_ratio but does NOT raise. Used for Phase 2
#   candidates (sparse-v2 and/or with NEAR_SURFACE) per the P2.10
#   sign-off. Caller opts in explicitly; the policy is recorded on the
#   diagnostics so it is visible in artifacts.


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


def _validate_unique_bundle_uids(
    entities: EntityArtifacts,
    surface_records: list[SurfaceRecord],
) -> None:
    """Reject ambiguous graph identity before projecting refs into sets."""
    entity_uids = [e.identity.object_uid for e in entities.entities]
    surface_uids = [s.uid for s in surface_records]
    if len(entity_uids) != len(set(entity_uids)):
        raise GraphBuildError("entity bundle contains duplicate entity uid values")
    if len(surface_uids) != len(set(surface_uids)):
        raise GraphBuildError("entity bundle contains duplicate surface uid values")


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
    *,
    effective_versions: dict[str, str],
) -> str:
    """Hash includes entity bundle, mode, and the ordered list of
    (extractor_name, effective_extractor_version, config) tuples. Run order
    is significant: a different order produces a different hash, which
    truthfully reflects that the inputs differ."""
    runs_payload = [
        {
            "extractor_name": r.extractor.name,
            "extractor_version": effective_versions[r.extractor.name],
            "config": _config_hash_payload(r.config),
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


def _config_hash_payload(config: RelationExtractorConfig) -> dict:
    """Serialize config for hashing while preserving legacy defaults.

    New opt-in config fields may omit their default value from the hash when
    that default means the pre-existing behavior. Non-default values always
    remain hash inputs.
    """
    payload = asdict(config)
    for config_field in fields(config):
        if not config_field.metadata.get("hash_omit_if_default"):
            continue
        if config_field.default is MISSING:
            continue
        if getattr(config, config_field.name) == config_field.default:
            payload.pop(config_field.name, None)
    return payload


def _project_surface_record(s: StructuralSurface) -> SurfaceRecord:
    """Project an EntityArtifacts.StructuralSurface onto a graph-level
    SurfaceRecord. All geometry + provenance carried verbatim (C1)."""
    return SurfaceRecord(
        uid=s.surface_uid,
        surface_type=s.surface_type,
        plane=s.plane,
        polygon=list(s.polygon) if s.polygon is not None else None,
        source=s.source,
        confidence=s.confidence,
    )


def _validate_edge_refs(
    edges: list[Edge],
    entity_uids: set[str],
    surface_uids: set[str],
) -> None:
    """G7: reject edges referencing entity or surface UIDs that are not
    in the entity bundle. Either source or target may reference either
    kind; the check is symmetric."""
    for e in edges:
        for ref in (e.source, e.target):
            if ref.kind == "entity" and ref.uid not in entity_uids:
                raise GraphBuildError(
                    f"edge {e.edge_id!r} references unknown entity "
                    f"uid {ref.uid!r} (extractor={e.extractor})"
                )
            if ref.kind == "surface" and ref.uid not in surface_uids:
                raise GraphBuildError(
                    f"edge {e.edge_id!r} references unknown surface "
                    f"uid {ref.uid!r} (extractor={e.extractor})"
                )
            if ref.kind not in ("entity", "surface"):
                raise GraphBuildError(
                    f"edge {e.edge_id!r} references unknown GraphRef kind "
                    f"{ref.kind!r} (extractor={e.extractor})"
                )


def build_graph(
    entities: EntityArtifacts,
    runs: list[ExtractorRun],
    *,
    density_policy: DensityPolicy = "phase1_block",
) -> tuple[SceneGraphBundle, BuildDiagnostics]:
    """Orchestrate extractors over an EntityArtifacts bundle.

    Validation order:
      1. At least one run; single mode; unique extractor names.
      2. Run extractors in caller-provided order.
      3. Reject duplicate edge_ids and duplicate edge keys.
      4. Reject edges referencing unknown entity / surface UIDs (G7).
      5. Sparse density check per `density_policy`:
           - phase1_block (default): raise on ratio > SPARSE_DENSITY_LIMIT.
           - phase2_telemetry_only: record ratio; do not raise.
      6. Assemble nodes, edges, and full structural_surfaces into a
         SceneGraphBundle (C1).
    """
    if density_policy not in ("phase1_block", "phase2_telemetry_only"):
        raise GraphBuildError(f"unknown density_policy {density_policy!r}")

    mode = _validate_single_mode(runs)
    _validate_unique_extractor_names(runs)

    # C1 identity validation happens before extractors run so malformed
    # bundles cannot leak ambiguity into relation generation.
    surface_records = [_project_surface_record(s) for s in entities.structural_surfaces]
    _validate_unique_bundle_uids(entities, surface_records)

    all_edges: list[Edge] = []
    per_extractor: list[RelationExtractorDiagnostics] = []
    runtime_ms_per_extractor: dict[str, int] = {}
    extractor_versions: dict[str, str] = {}
    edges_emitted_per_type: dict[EdgeType, int] = {}
    rejections_per_type: dict[EdgeType, int] = {}
    rejection_samples: list[EdgeRejection] = []

    for run in runs:
        family_edges, family_diag = run.extractor.extract(entities, run.config)
        extractor_versions[run.extractor.name] = family_diag.version
        per_extractor.append(family_diag)
        runtime_ms_per_extractor[family_diag.extractor] = family_diag.runtime_ms
        for t, c in family_diag.physical_edges_per_type.items():
            edges_emitted_per_type[t] = edges_emitted_per_type.get(t, 0) + c
        for t, c in family_diag.rejections_per_type.items():
            rejections_per_type[t] = rejections_per_type.get(t, 0) + c
        rejection_samples.extend(family_diag.rejection_samples)
        all_edges.extend(family_edges)

    _validate_no_duplicates(all_edges)

    # C1: retain all surfaces from the entity bundle (not only those
    # referenced by edges).
    entity_uids = {e.identity.object_uid for e in entities.entities}
    surface_uids = {s.uid for s in surface_records}

    # G7: reject edges with unknown entity/surface UIDs.
    _validate_edge_refs(all_edges, entity_uids, surface_uids)

    physical_total = len(all_edges)
    logical_total = count_logical_edges(all_edges)
    entity_count = len(entities.entities)
    density_ratio = (
        logical_total / entity_count if entity_count > 0 else None
    )

    if mode == "sparse" and density_policy == "phase1_block":
        _enforce_sparse_density(logical_total, entity_count)

    nodes = [_node_from_entity(e, entity_bundle=entities) for e in entities.entities]
    bundle = SceneGraphBundle(
        schema_version=CURRENT_SCHEMA_VERSION,
        bundle_hash=_build_bundle_hash(
            entities.bundle_hash,
            mode,
            runs,
            effective_versions=extractor_versions,
        ),
        scene_id=entities.scene_id,
        frame=entities.frame,
        entity_bundle_hash=entities.bundle_hash,
        nodes=nodes,
        edges=all_edges,
        structural_surface_refs=[s.uid for s in surface_records],
        structural_surfaces=surface_records,
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
        density_policy=density_policy,
        density_ratio=density_ratio,
        sparse_density_limit=SPARSE_DENSITY_LIMIT,
    )
    return bundle, diagnostics

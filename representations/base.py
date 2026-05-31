"""SceneRepresentationBundle (immutable artifact) and SceneRepresentation
Protocol (runtime wrapper) — phase0_design.md §5.2.

The bundle is content-addressed, serializable, and has no methods. The
Protocol is a runtime object that wraps a bundle and exposes capabilities
like render_view and query_geometry. Methods do not live on the on-disk
artifact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from common.types import CameraPose, JSON, SceneFrame


Channel = Literal["rgb", "depth", "normals", "semantic_features", "instance_features"]


@dataclass(frozen=True)
class GeometryHandle:
    kind: Literal[
        "mesh_file", "pointcloud_file", "splat_file", "nerf_checkpoint",
        "in_memory",
        # oracle_passthrough: the geometry "source" is an upstream system
        # (e.g. Habitat) that already references it; this bundle does not
        # load a geometry blob. Used by oracle adapters.
        "oracle_passthrough",
    ]
    uri: str
    notes: dict[str, JSON] = field(default_factory=dict)


@dataclass(frozen=True)
class ReconstructionDiagnostics:
    loss: float | None
    coverage: float | None
    pose_rmse: float | None
    runtime_seconds: float
    notes: str


@dataclass(frozen=True)
class RepresentationCapabilities:
    renderable_channels: frozenset[Channel]
    supports_arbitrary_pose: bool
    deterministic: bool
    typical_render_ms: int


@dataclass(frozen=True)
class SceneRepresentationBundle:
    schema_version: int            # bump on breaking changes; loader checks
    representation_hash: str
    scene_id: str
    frame: SceneFrame
    capabilities: RepresentationCapabilities
    geometry_handle: GeometryHandle
    poses: list[CameraPose]
    diagnostics: ReconstructionDiagnostics
    notes: dict[str, JSON] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderRequest:
    request_hash: str
    camera: CameraPose
    width: int
    height: int
    channels: frozenset[Channel]
    feature_extractor: str | None


@dataclass(frozen=True)
class ViewBundle:
    request: RenderRequest
    camera: CameraPose
    rgb: np.ndarray | None
    depth: np.ndarray | None
    normals: np.ndarray | None
    semantic_features: np.ndarray | None
    instance_features: np.ndarray | None
    feature_extractor: str | None
    cache_key: str


@dataclass(frozen=True)
class GeometryQuery:
    kind: str
    params: dict[str, JSON] = field(default_factory=dict)


@dataclass(frozen=True)
class GeometryResult:
    kind: str
    payload: dict[str, JSON]


class SceneRepresentation(Protocol):
    bundle: SceneRepresentationBundle

    def render_view(self, request: RenderRequest) -> ViewBundle: ...

    def query_geometry(self, query: GeometryQuery) -> GeometryResult: ...

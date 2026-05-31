"""ReconstructionAdapter Protocol and CaptureBundle — phase0_design.md §5.1.

Stage 1 of the pipeline. Reads raw capture (images, mesh, semantic export),
produces an immutable SceneRepresentationBundle.

Adapters do not score themselves. Diagnostics go on the bundle; pass/fail
metrics live in eval/.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from common.types import CameraPose, JSON
from representations.base import SceneRepresentationBundle


@dataclass(frozen=True)
class CaptureBundle:
    """Tagged-union-style input. Each adapter declares which fields it
    accepts via its capabilities; unsupported fields should be None."""
    bundle_hash: str
    scene_id: str
    images_dir: Path | None
    poses: list[CameraPose] | None
    rgbd_dir: Path | None
    mesh_path: Path | None
    semantic_export: Path | None
    notes: dict[str, JSON] = field(default_factory=dict)


@dataclass(frozen=True)
class ReconstructionConfig:
    name: str
    version: str
    params: dict[str, JSON] = field(default_factory=dict)


@dataclass(frozen=True)
class ReconstructionCapabilities:
    produces_mesh: bool
    produces_pointcloud: bool
    produces_gaussian_splat: bool
    produces_nerf_field: bool
    estimates_poses: bool
    requires_gpu: bool
    typical_runtime_minutes: int


class ReconstructionAdapter(Protocol):
    name: str
    version: str

    def reconstruct(
        self,
        capture: CaptureBundle,
        config: ReconstructionConfig,
    ) -> SceneRepresentationBundle: ...

    def capabilities(self) -> ReconstructionCapabilities: ...

"""Runtime mesh-backed SceneRepresentation.

Wraps a SceneRepresentationBundle whose geometry handle is either a mesh
file or an oracle passthrough. Phase 1 oracle path does NOT render; the
capability flags declare this and render_view raises NotImplementedError
when called. query_geometry supports a "describe_handle" probe so callers
can inspect the geometry source without loading blobs.

Real mesh loading and rendering are Phase 2+ work and would replace this
module's behavior contract while keeping the same Protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from representations.base import (
    GeometryQuery, GeometryResult, RenderRequest, SceneRepresentationBundle,
    ViewBundle,
)
from representations.serde import load_scene_repr_bundle


@dataclass(frozen=True)
class MeshRepresentation:
    """Implements the SceneRepresentation Protocol for mesh-backed and
    oracle-passthrough bundles. Phase 1 behavior:

      - render_view: raises NotImplementedError. The bundle's
        capabilities.renderable_channels is empty, so this should never
        be called by code that respects the capability contract.
      - query_geometry: only "describe_handle" is supported. It returns
        the GeometryHandle's metadata so downstream code can inspect the
        source without dereferencing it.
    """
    bundle: SceneRepresentationBundle

    def render_view(self, request: RenderRequest) -> ViewBundle:
        raise NotImplementedError(
            "MeshRepresentation does not render in Phase 1; "
            f"requested channels: {sorted(request.channels)}. "
            "Check bundle.capabilities.renderable_channels before calling."
        )

    def query_geometry(self, query: GeometryQuery) -> GeometryResult:
        if query.kind != "describe_handle":
            raise NotImplementedError(
                "Phase 1 MeshRepresentation only supports "
                f"query.kind='describe_handle'; got {query.kind!r}"
            )
        h = self.bundle.geometry_handle
        return GeometryResult(
            kind="handle_description",
            payload={
                "geometry_kind": h.kind,
                "uri": h.uri,
                "notes": dict(h.notes),
            },
        )


def load_mesh_representation(bundle_dir: Path) -> MeshRepresentation:
    """Convenience loader: read a SceneRepresentationBundle from disk and
    wrap it as a runtime MeshRepresentation."""
    return MeshRepresentation(bundle=load_scene_repr_bundle(bundle_dir))

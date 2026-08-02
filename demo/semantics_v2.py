"""semantics_v2 track battery configuration (opt-in; S1 deliverable).

Protocol: docs/semantics_v2_track_protocol.md (signed off 2026-08-02).
This module is the ONLY place the v2 track is assembled: the frozen v1
`demo/question_battery._runs()` and every default gate path are
untouched (guarded by tests/tools/test_semantics_v2_guards.py).

  runs_v2()          — the v1 extractor stack with ATTACHED_TO and
                       ON_ENTITY_SURFACE swapped to their v2 versions
                       (D1/D2/D3); everything else byte-identical.
  make_v2_compiler() — RulesCompiler with the D3 anchor additions
                       (cabinet / nightstand / bed).

No scene is scored by importing this module; S2 (variant A first) runs
only under its own owner authorization.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph.builder import ExtractorRun
from graph.relations.attached_to_v2 import AttachedToV2Config, AttachedToV2Extractor
from graph.relations.contacts_surface import ContactsSurfaceConfig, ContactsSurfaceExtractor
from graph.relations.directional import DirectionalConfig, DirectionalExtractor
from graph.relations.on_entity_surface_v2 import (
    OnEntitySurfaceV2Config, OnEntitySurfaceV2Extractor,
)
from graph.relations.on_surface import OnSurfaceConfig, OnSurfaceExtractor
from graph.relations.surface import SurfaceProximityConfig, SurfaceProximityExtractor
from reasoner.compiler_rules import RulesCompiler

V2_ON_ANCHORS = {"cabinet": "cabinet", "nightstand": "nightstand",
                 "bed": "bed"}


def runs_v2() -> list[ExtractorRun]:
    """v1 battery stack with the two v2 relation swaps (D1/D2/D3)."""
    return [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse")),
        ExtractorRun(SurfaceProximityExtractor(),
                     SurfaceProximityConfig(use_polygon_clip=True,
                                            exclude_room_scale_flat=True)),
        ExtractorRun(OnSurfaceExtractor(), OnSurfaceConfig()),
        ExtractorRun(ContactsSurfaceExtractor(),
                     ContactsSurfaceConfig(exclude_room_scale_flat=True)),
        ExtractorRun(OnEntitySurfaceV2Extractor(), OnEntitySurfaceV2Config()),
        ExtractorRun(AttachedToV2Extractor(), AttachedToV2Config()),
    ]


def make_v2_compiler() -> RulesCompiler:
    return RulesCompiler(extra_on_classes=dict(V2_ON_ANCHORS))

"""Scene viewpoint manifest loader.

A views manifest describes the photos / rendered viewpoints used to drive a
VLM or hybrid runner against a scene. Shape:

  {
    "scene_id": "replica_room_0",
    "source": "rendered_from_mesh" | "real_capture" | "mixed",
    "viewpoints": [
      {
        "id": "vp_north_doorway",
        "path": "photos/vp_north_doorway.png",   # relative to manifest dir
        "note": "looking south from doorway",
        "camera": {                                # optional: pose for hybrid runner
          "position": [x, y, z],
          "look_at": [x, y, z],
          "up": [x, y, z]
        }
      }
    ]
  }

This module does NOT render or capture anything. If a scene has no manifest, the
caller should error with a clear hint (eval_vlm.py does this). Adding photos is
a separate slice — render from mesh, capture with a phone, etc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Viewpoint:
    id: str
    path: Path
    note: str = ""
    camera: dict[str, Any] | None = None


@dataclass
class SceneViews:
    scene_id: str
    source: str
    viewpoints: list[Viewpoint] = field(default_factory=list)


def load_views(manifest_path: Path | str) -> SceneViews:
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No views manifest at {manifest_path}.\n"
            "Expected shape:\n"
            '  {"scene_id": "...", "source": "rendered_from_mesh", "viewpoints": [\n'
            '    {"id": "vp_1", "path": "photos/vp_1.png", "note": "..."}\n'
            "  ]}"
        )
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    vps: list[Viewpoint] = []
    for v in raw.get("viewpoints", []):
        p = Path(v["path"])
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"viewpoint image not found: {p} (referenced by {manifest_path})")
        vps.append(Viewpoint(
            id=v["id"],
            path=p,
            note=v.get("note", ""),
            camera=v.get("camera"),
        ))
    if not vps:
        raise ValueError(f"{manifest_path} has zero viewpoints")
    return SceneViews(
        scene_id=raw["scene_id"],
        source=raw.get("source", "unknown"),
        viewpoints=vps,
    )

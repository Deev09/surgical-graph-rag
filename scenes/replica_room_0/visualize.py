"""Interactive overlay: Replica mesh + our scene_graph.json centroids.

Uses trimesh's pyglet viewer. Each scene_graph object becomes a colored
sphere placed at its xyz on top of the raw mesh. A printed legend maps
each color to label + xyz so you can identify spheres in the viewer.

Usage (from repo root):
    .venv/bin/python scenes/replica_room_0/visualize.py \
        --mesh ~/datasets/replica/room_0/mesh.ply \
        --scene scenes/replica_room_0/scene_graph.json

Coordinate alignment:
    The importer translated z so the floor sits at 0. The Replica mesh's
    raw z origin sits ~1.5m below the floor. We undo that translation
    when drawing centroids back onto the raw mesh.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import os
from pathlib import Path

import numpy as np
import trimesh


def _color_for(label: str) -> tuple[int, int, int, int]:
    h = (hash(label) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255), 255)


def _radius_for(sizes: list[float]) -> float:
    vol = max(sizes[0] * sizes[1] * sizes[2], 1e-6)
    return float(max(0.06, min(0.25, vol ** (1 / 3) * 0.35)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True, type=Path)
    p.add_argument("--scene", required=True, type=Path)
    p.add_argument("--meta", type=Path, default=None)
    p.add_argument("--no-show", action="store_true", help="just print legend, don't open viewer")
    args = p.parse_args()

    args.mesh = Path(os.path.expanduser(str(args.mesh)))

    scene_data = json.loads(args.scene.read_text())
    meta_path = args.meta or (args.scene.parent / "capture_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    z_translation = float(meta.get("import_notes", {}).get("z_translation_applied", 0.0))

    print(f"Loading mesh: {args.mesh}")
    mesh = trimesh.load(str(args.mesh), process=False)
    print(f"Mesh has {len(mesh.vertices)} vertices, {len(mesh.faces)} faces.")

    geoms: list = [mesh]
    legend_rows: list[tuple[str, str, tuple[int, int, int]]] = []
    for obj in scene_data["objects"]:
        x, y, z = obj["xyz"]
        z_raw = z - z_translation
        sphere = trimesh.creation.icosphere(
            subdivisions=2,
            radius=_radius_for(obj["attributes"].get("bbox_sizes", [0.2, 0.2, 0.2])),
        )
        sphere.apply_translation([x, y, z_raw])
        rgba = _color_for(obj["label"])
        sphere.visual.face_colors = np.tile(rgba, (len(sphere.faces), 1))
        geoms.append(sphere)
        legend_rows.append(
            (obj["label"], f"({x:.2f}, {y:.2f}, {z:.2f}) [raw z={z_raw:.2f}]", rgba[:3])
        )

    print()
    print(f"{'label':<28} {'centroid':<40} color (R,G,B)")
    print("-" * 90)
    for label, pos, (r, g, b) in legend_rows:
        print(f"{label:<28} {pos:<40} ({r:3d},{g:3d},{b:3d})")
    print()
    print(f"Total: {len(legend_rows)} centroids on mesh with {len(mesh.vertices)} vertices.")
    print("Mouse: rotate. Right-drag: pan. Scroll: zoom. Q to quit.")

    if args.no_show:
        return

    scn = trimesh.Scene(geoms)
    scn.show()


if __name__ == "__main__":
    main()

"""Render 4 canonical viewpoints of the Replica room_0 mesh.

Outputs:
  scenes/replica_room_0/photos/vp_<id>.png      (rendered RGB images)
  scenes/replica_room_0/views.json              (manifest matching benchmark/views.py)

The 4 viewpoints are deterministic corner cameras, eye-height 1.5m above floor,
each looking at the room center. Frame-coordinate convention in views.json is
the scene_graph (post-import) frame, NOT the raw mesh frame. The renderer
internally subtracts the z_translation_applied from capture_meta.json to map
into the raw mesh frame.

Requires (dev tooling, not in requirements.txt):
    pip install pyrender PyOpenGL Pillow

Usage (from repo root):
    .venv/bin/python scenes/replica_room_0/render_views.py \\
        --mesh ~/Desktop/datasets/replica/room_0/mesh.ply
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import trimesh

# pyrender's offscreen renderer dispatches on PYOPENGL_PLATFORM. macOS works
# fine with the variable unset (default NSGL-backed context). If it's set to an
# empty string for some reason, pyrender errors out, so unset it explicitly.
if os.environ.get("PYOPENGL_PLATFORM") == "":
    del os.environ["PYOPENGL_PLATFORM"]
import pyrender  # noqa: E402
from PIL import Image  # noqa: E402


VIEWPOINTS = [
    ("vp_sw", "south-west corner looking NE"),
    ("vp_se", "south-east corner looking NW"),
    ("vp_nw", "north-west corner looking SE"),
    ("vp_ne", "north-east corner looking SW"),
]

WIDTH = 1024
HEIGHT = 768
YFOV_DEG = 75.0
EYE_HEIGHT_ABOVE_FLOOR_M = 1.5
WALL_INSET_M = 0.4


def look_at_pose(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Camera-to-world transform. pyrender convention: -z forward, +y up, +x right."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    new_up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = new_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def corner_eye_positions(
    room_bbox: list[list[float]],
    floor_z_raw: float,
    eye_height: float,
    inset: float,
) -> dict[str, np.ndarray]:
    (x_lo, y_lo, _), (x_hi, y_hi, _) = room_bbox
    x_lo += inset; x_hi -= inset
    y_lo += inset; y_hi -= inset
    z = floor_z_raw + eye_height
    return {
        "vp_sw": np.array([x_lo, y_lo, z]),
        "vp_se": np.array([x_hi, y_lo, z]),
        "vp_nw": np.array([x_lo, y_hi, z]),
        "vp_ne": np.array([x_hi, y_hi, z]),
    }


def build_scene(mesh: trimesh.Trimesh) -> pyrender.Scene:
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene(
        bg_color=[0.05, 0.05, 0.08, 1.0],
        ambient_light=[0.4, 0.4, 0.4],
    )
    scene.add(pr_mesh)
    return scene


def render_one(
    scene: pyrender.Scene,
    renderer: pyrender.OffscreenRenderer,
    eye_raw: np.ndarray,
    target_raw: np.ndarray,
    yfov_rad: float,
) -> np.ndarray:
    up = np.array([0.0, 0.0, 1.0])
    cam_pose = look_at_pose(eye_raw, target_raw, up)
    cam = pyrender.PerspectiveCamera(yfov=yfov_rad, aspectRatio=WIDTH / HEIGHT)
    cam_node = scene.add(cam, pose=cam_pose)

    # Headlamp: directional light pointing forward from camera.
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.5)
    light_node = scene.add(light, pose=cam_pose)

    color, _ = renderer.render(scene)
    scene.remove_node(cam_node)
    scene.remove_node(light_node)
    return color


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, type=Path)
    ap.add_argument("--meta", type=Path, default=None,
                    help="capture_meta.json (defaults to sibling of this script)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output dir for photos/ and views.json (defaults to sibling of this script)")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    meta_path = args.meta or (script_dir / "capture_meta.json")
    out_dir = args.out_dir or script_dir
    photos_dir = out_dir / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = Path(os.path.expanduser(str(args.mesh))).resolve()
    if not mesh_path.exists():
        raise SystemExit(f"mesh not found: {mesh_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    room_bbox = meta["room_bbox"]
    z_translation = float(meta["import_notes"]["z_translation_applied"])
    scene_id = meta["scene_id"]

    # Floor in scene-graph z is ~0.047 (from room_bbox[0][2]). In raw mesh z:
    floor_z_scene = float(room_bbox[0][2])
    floor_z_raw = floor_z_scene - z_translation

    # Room center: midpoint of xy bbox; z at eye height (so cameras look horizontally).
    cx_scene = 0.5 * (room_bbox[0][0] + room_bbox[1][0])
    cy_scene = 0.5 * (room_bbox[0][1] + room_bbox[1][1])
    target_z_scene = floor_z_scene + EYE_HEIGHT_ABOVE_FLOOR_M
    target_z_raw = target_z_scene - z_translation
    target_raw = np.array([cx_scene, cy_scene, target_z_raw])
    target_scene = np.array([cx_scene, cy_scene, target_z_scene])

    eyes_raw = corner_eye_positions(room_bbox, floor_z_raw, EYE_HEIGHT_ABOVE_FLOOR_M, WALL_INSET_M)

    print(f"loading mesh: {mesh_path}")
    t0 = time.perf_counter()
    raw_mesh = trimesh.load(str(mesh_path), process=False, force="mesh")
    print(f"  -> {len(raw_mesh.vertices)} vertices, {len(raw_mesh.faces)} faces "
          f"({(time.perf_counter() - t0):.1f}s)")

    print("building pyrender scene...")
    t0 = time.perf_counter()
    pr_scene = build_scene(raw_mesh)
    print(f"  -> done ({(time.perf_counter() - t0):.1f}s)")

    print("creating offscreen renderer...")
    renderer = pyrender.OffscreenRenderer(viewport_width=WIDTH, viewport_height=HEIGHT)

    yfov_rad = np.deg2rad(YFOV_DEG)
    viewpoints_out = []
    for vp_id, note in VIEWPOINTS:
        eye_raw = eyes_raw[vp_id]
        t0 = time.perf_counter()
        color = render_one(pr_scene, renderer, eye_raw, target_raw, yfov_rad)
        out_path = photos_dir / f"{vp_id}.png"
        Image.fromarray(color).save(out_path)
        dt = (time.perf_counter() - t0) * 1000
        print(f"  {vp_id}: eye_raw={eye_raw.tolist()} -> {out_path} ({dt:.0f}ms)")

        # Record camera in scene-graph coordinates (eye_raw.z + z_translation).
        eye_scene = eye_raw.copy()
        eye_scene[2] = eye_raw[2] + z_translation
        viewpoints_out.append({
            "id": vp_id,
            "path": f"photos/{vp_id}.png",
            "note": note,
            "camera": {
                "position": eye_scene.tolist(),
                "look_at": target_scene.tolist(),
                "up": [0.0, 0.0, 1.0],
                "yfov_deg": YFOV_DEG,
                "aspect": WIDTH / HEIGHT,
                "resolution": [WIDTH, HEIGHT],
            },
        })
    renderer.delete()

    manifest = {
        "scene_id": scene_id,
        "source": "rendered_from_mesh",
        "mesh_source": str(mesh_path),
        "renderer": "pyrender",
        "z_translation_applied": z_translation,
        "viewpoints": viewpoints_out,
    }
    manifest_path = out_dir / "views.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {manifest_path}")
    print(f"wrote {len(viewpoints_out)} photos in {photos_dir}")


if __name__ == "__main__":
    main()

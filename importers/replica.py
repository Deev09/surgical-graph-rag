"""Convert a Replica scene's Habitat semantic export into our scene_graph.json.

Reads:
    <replica_scene>/habitat/info_semantic.json

Writes:
    scenes/<scene_id>/scene_graph.json
    scenes/<scene_id>/capture_meta.json

What this does:
    - Filters out class_name == "undefined" instances.
    - Optional structural filter drops {"wall", "floor", "ceiling"} unless --keep-structural.
    - Suffixes duplicate labels (window -> window_1, window_2, ...).
    - Translates xyz so the floor sits at z = 0 (min z of kept objects).
    - Emits zone = null. Replica has no zones.
    - Records gravity_dir, original/translated coord origin, and source provenance
      in capture_meta.json so downstream code can verify axis convention.

What this does NOT do:
    - No relations are emitted. relations/compute.py is the only relation source.
    - No semantic merging across instances.
    - No re-orientation if gravity_dir is not approximately +Z up.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

STRUCTURAL_DROP = {"wall", "floor", "ceiling"}
UNDEFINED = "undefined"


def _abs(v: float) -> float:
    return v if v >= 0 else -v


def _gravity_is_neg_z(gravity_dir: list[float]) -> bool:
    return _abs(gravity_dir[0]) < 0.05 and _abs(gravity_dir[1]) < 0.05 and gravity_dir[2] < -0.95


def _quat_rotate(q: list[float], v: list[float]) -> list[float]:
    """Rotate vector v by unit quaternion q=[x,y,z,w]. Replica's abb.center is
    in the object's local pre-rotation frame; multiplying by the orientation
    quaternion places it in the world (mesh) frame."""
    qxyz = np.asarray(q[:3], dtype=float)
    qw = float(q[3])
    v_arr = np.asarray(v, dtype=float)
    t = 2.0 * np.cross(qxyz, v_arr)
    return (v_arr + qw * t + np.cross(qxyz, t)).tolist()


def _suffix_duplicates(records: list[dict]) -> list[dict]:
    counts: dict[str, int] = defaultdict(int)
    label_total: dict[str, int] = defaultdict(int)
    for r in records:
        label_total[r["label"]] += 1
    for r in records:
        if label_total[r["label"]] > 1:
            counts[r["label"]] += 1
            r["label"] = f"{r['label']}_{counts[r['label']]}"
    return records


def import_replica(scene_dir: Path, scene_id: str, out_root: Path, keep_structural: bool) -> dict:
    info = json.loads((scene_dir / "habitat" / "info_semantic.json").read_text())
    objects_raw = info["objects"]
    gravity_dir = info["gravity_dir"]

    if not _gravity_is_neg_z(gravity_dir):
        raise SystemExit(
            f"Refusing to import: gravity_dir={gravity_dir} is not approximately -Z. "
            "This importer assumes +Z up."
        )

    kept = []
    dropped_undefined = 0
    dropped_structural = 0
    for o in objects_raw:
        name = o["class_name"]
        if name == UNDEFINED:
            dropped_undefined += 1
            continue
        if (not keep_structural) and name in STRUCTURAL_DROP:
            dropped_structural += 1
            continue
        center_local = o["oriented_bbox"]["abb"]["center"]
        quat = o["oriented_bbox"]["orientation"]["rotation"]
        center_world = _quat_rotate(quat, center_local)
        sizes = o["oriented_bbox"]["abb"]["sizes"]
        kept.append(
            {
                "instance_id": o["id"],
                "label": name,
                "xyz_raw": [float(center_world[0]), float(center_world[1]), float(center_world[2])],
                "sizes": [float(sizes[0]), float(sizes[1]), float(sizes[2])],
            }
        )

    z_min = min(r["xyz_raw"][2] - r["sizes"][2] / 2 for r in kept)
    objects_out = []
    for r in _suffix_duplicates(kept):
        x, y, z = r["xyz_raw"]
        objects_out.append(
            {
                "id": f"obj_{r['instance_id']}",
                "label": r["label"],
                "zone": None,
                "xyz": [round(x, 3), round(y, 3), round(z - z_min, 3)],
                "attributes": {
                    "type": r["label"],
                    "bbox_sizes": [round(s, 3) for s in r["sizes"]],
                },
            }
        )

    scene_graph = {"scene": scene_id, "objects": objects_out, "relations": []}

    xs = [o["xyz"][0] for o in objects_out]
    ys = [o["xyz"][1] for o in objects_out]
    zs = [o["xyz"][2] for o in objects_out]
    capture_meta = {
        "scene_id": scene_id,
        "source": "replica/room_0",
        "axis_convention": {"up_axis": "+z", "gravity_dir_raw": gravity_dir},
        "units": "meters",
        "room_bbox": [
            [round(min(xs), 3), round(min(ys), 3), round(min(zs), 3)],
            [round(max(xs), 3), round(max(ys), 3), round(max(zs), 3)],
        ],
        "object_count": len(objects_out),
        "authored_relation_count": 0,
        "import_notes": {
            "z_translation_applied": round(-z_min, 3),
            "dropped_undefined": dropped_undefined,
            "dropped_structural": dropped_structural,
            "keep_structural": keep_structural,
            "zone_field": "always_null_for_this_scene",
            "abb_center_rotated_by_orientation_quat": True,
            "bbox_sizes_kept_in_local_frame": True,
        },
    }

    out_dir = out_root / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scene_graph.json").write_text(json.dumps(scene_graph, indent=2))
    (out_dir / "capture_meta.json").write_text(json.dumps(capture_meta, indent=2))
    return capture_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--out-root", default=Path("scenes"), type=Path)
    parser.add_argument("--keep-structural", action="store_true")
    args = parser.parse_args()
    meta = import_replica(args.scene_dir, args.scene_id, args.out_root, args.keep_structural)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

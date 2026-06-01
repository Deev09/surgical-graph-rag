"""Convert a Replica scene's Habitat semantic export into our scene_graph.json.

Reads:
    <replica_scene>/habitat/info_semantic.json

Writes:
    scenes/<scene_id>/scene_graph.json
    scenes/<scene_id>/capture_meta.json
    scenes/<scene_id>/enriched/v2/{scene_graph.json,capture_meta.json}
        when --enriched-v2 is passed

What this does:
    - Filters out class_name == "undefined" instances.
    - Optional structural filter drops {"wall", "floor", "ceiling"} unless --keep-structural.
    - Suffixes duplicate labels (window -> window_1, window_2, ...).
    - Translates xyz so the floor sits at z = 0 (min z of kept objects).
    - Emits zone = null. Replica has no zones.
    - Records gravity_dir, original/translated coord origin, and source provenance
      in capture_meta.json so downstream code can verify axis convention.
    - Optional --enriched-v2 output preserves world-frame OBBs and derives tight
      world-frame AABBs without modifying the Phase 1 replay fixture.

What this does NOT do:
    - No relations are emitted. relations/compute.py is the only relation source.
    - No semantic merging across instances.
    - No re-orientation if gravity_dir is not approximately +Z up.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

STRUCTURAL_DROP = {"wall", "floor", "ceiling"}
STRUCTURAL_REQUIRED = ("floor", "wall", "ceiling")
UNDEFINED = "undefined"
ENRICHED_SCHEMA_VERSION = 2


def _abs(v: float) -> float:
    return v if v >= 0 else -v


def _gravity_is_neg_z(gravity_dir: list[float]) -> bool:
    return _abs(gravity_dir[0]) < 0.05 and _abs(gravity_dir[1]) < 0.05 and gravity_dir[2] < -0.95


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _quat_rotate(q: list[float], v: list[float]) -> list[float]:
    """Rotate vector v by unit quaternion q=[x,y,z,w]. Replica's abb.center is
    in the object's local pre-rotation frame; multiplying by the orientation
    quaternion places it in the world (mesh) frame."""
    qxyz = [float(x) for x in q[:3]]
    qw = float(q[3])
    v_xyz = [float(x) for x in v]
    t = [2.0 * x for x in _cross(qxyz, v_xyz)]
    q_cross_t = _cross(qxyz, t)
    return [v_xyz[i] + qw * t[i] + q_cross_t[i] for i in range(3)]


def _aabb_from_obb(
    center: list[float],
    extents: list[float],
    rotation_quat: list[float],
) -> tuple[list[float], list[float]]:
    """Return the tight world-frame AABB enclosing an oriented bbox."""
    corners: list[list[float]] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local_offset = [
                    sx * extents[0],
                    sy * extents[1],
                    sz * extents[2],
                ]
                rotated = _quat_rotate(rotation_quat, local_offset)
                corners.append([center[i] + rotated[i] for i in range(3)])
    return (
        [min(c[i] for c in corners) for i in range(3)],
        [max(c[i] for c in corners) for i in range(3)],
    )


def _translate_z(v: list[float], dz: float) -> list[float]:
    return [float(v[0]), float(v[1]), float(v[2]) + dz]


def _round_vec(v: list[float], digits: int = 6) -> list[float]:
    return [round(float(x), digits) for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _scale(v: list[float], s: float) -> list[float]:
    return [s * v[0], s * v[1], s * v[2]]


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(_dot(v, v))
    if n == 0.0:
        raise ValueError("cannot unit-normalize zero vector")
    return [v[0] / n, v[1] / n, v[2] / n]


def _quat_axis_in_world(rotation_quat: list[float], local_axis_idx: int) -> list[float]:
    """World-frame direction of the OBB's local-frame unit axis k ∈ {0,1,2}."""
    basis = [0.0, 0.0, 0.0]
    basis[local_axis_idx] = 1.0
    return _quat_rotate(rotation_quat, basis)


def _thin_axis_index(extents: list[float]) -> int:
    return min(range(3), key=lambda i: extents[i])


def _obb_face_corners(
    center_world: list[float],
    extents: list[float],
    rotation_quat: list[float],
    thin_axis_idx: int,
    sign_along_thin: float,
) -> list[list[float]]:
    """4 world-frame corners of the OBB face perpendicular to the thin axis,
    on the side selected by sign_along_thin in {+1,-1}. Initial order is
    fixed (-,-), (+,-), (+,+), (-,+) over the two non-thin local axes;
    callers must run _orient_polygon to enforce the winding convention."""
    others = [i for i in range(3) if i != thin_axis_idx]
    i_idx, j_idx = others[0], others[1]
    e_thin = extents[thin_axis_idx]
    e_i = extents[i_idx]
    e_j = extents[j_idx]
    corners: list[list[float]] = []
    for si, sj in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
        local = [0.0, 0.0, 0.0]
        local[thin_axis_idx] = sign_along_thin * e_thin
        local[i_idx] = si * e_i
        local[j_idx] = sj * e_j
        rotated = _quat_rotate(rotation_quat, local)
        corners.append([center_world[k] + rotated[k] for k in range(3)])
    return corners


def _winding_dot(corners: list[list[float]], normal: list[float]) -> float:
    """dot(cross(p1-p0, p2-p0), normal). Positive means polygon winding
    matches the interior-facing normal (the agreed-on Phase 2 convention)."""
    a = _sub(corners[1], corners[0])
    b = _sub(corners[2], corners[0])
    return _dot(_cross(a, b), normal)


def _orient_polygon(
    corners: list[list[float]], normal: list[float],
) -> list[list[float]]:
    """Return corners reordered so dot(cross(p1-p0, p2-p0), normal) > 0.
    Keeps p0 fixed and reverses the traversal direction if the test fails."""
    if _winding_dot(corners, normal) > 0:
        return corners
    return [corners[0], corners[3], corners[2], corners[1]]


def _orient_tag(normal_world: list[float]) -> str:
    """Dominant-axis tag for an interior-facing wall normal: e.g. xplus,
    yminus. Used only to make wall surface_uids legible."""
    abs_components = [abs(normal_world[0]), abs(normal_world[1]), abs(normal_world[2])]
    dominant = max(range(3), key=lambda k: abs_components[k])
    axis = ("x", "y", "z")[dominant]
    sign = "plus" if normal_world[dominant] > 0 else "minus"
    return f"{axis}{sign}"


def _build_structural_surface_record(
    inst: dict,
    surface_type: str,
    gravity_aligned_interior_hint: list[float] | None,
    interior_ref_world: list[float] | None,
    z_min: float,
) -> tuple[dict, list[float]]:
    """Build one structural-surface record from a Habitat-labeled instance.

    Returns (record_dict, world_frame_face_centroid). The face centroid is
    returned in untranslated world frame so the caller can use the floor's
    face centroid as the interior reference for walls (per P2.03 sign-off
    point (a): orient walls using the floor-face centroid, NOT the
    object-centroid room_bbox).
    """
    instance_id = inst["id"]
    center_local = inst["oriented_bbox"]["abb"]["center"]
    quat = [float(x) for x in inst["oriented_bbox"]["orientation"]["rotation"]]
    sizes = inst["oriented_bbox"]["abb"]["sizes"]
    extents = [float(s) / 2.0 for s in sizes]

    center_world = _quat_rotate(quat, center_local)
    thin_idx = _thin_axis_index(extents)
    thin_world_unit = _unit(_quat_axis_in_world(quat, thin_idx))

    if gravity_aligned_interior_hint is not None:
        ref_dir = gravity_aligned_interior_hint
    else:
        if interior_ref_world is None:
            raise ValueError(
                "wall structural-surface extraction requires interior_ref_world "
                "(floor-face centroid)"
            )
        ref_dir = _sub(interior_ref_world, center_world)

    sign_along_thin = 1.0 if _dot(thin_world_unit, ref_dir) >= 0 else -1.0
    interior_normal = _scale(thin_world_unit, sign_along_thin)

    corners_world = _obb_face_corners(
        center_world, extents, quat, thin_idx, sign_along_thin,
    )
    corners_world = _orient_polygon(corners_world, interior_normal)

    corners_translated = [_translate_z(c, -z_min) for c in corners_world]
    center_translated = _translate_z(center_world, -z_min)

    face_centroid_world = [
        sum(c[k] for c in corners_world) / 4.0 for k in range(3)
    ]
    face_centroid_translated = _translate_z(face_centroid_world, -z_min)

    plane_d = -_dot(interior_normal, face_centroid_translated)

    if surface_type == "floor":
        surface_uid = f"floor_{instance_id}"
    elif surface_type == "ceiling":
        surface_uid = f"ceiling_{instance_id}"
    else:
        tag = _orient_tag(interior_normal)
        surface_uid = f"wall_{instance_id}_{tag}"

    record = {
        "surface_uid": surface_uid,
        "surface_type": surface_type,
        "source": "habitat_label",
        "source_instance_ref": str(instance_id),
        "plane": {
            "normal": _round_vec(interior_normal),
            "d": round(float(plane_d), 6),
        },
        "polygon": [_round_vec(c) for c in corners_translated],
        "source_obb": {
            "center": _round_vec(center_translated),
            "extents": _round_vec(extents),
            "rotation_quat": [round(float(x), 6) for x in quat],
        },
        "confidence": 1.0,
    }
    return record, face_centroid_world


def _extract_structural_surfaces(
    info: dict, z_min: float,
) -> list[dict]:
    """Extract floor / wall / ceiling surfaces from Habitat semantic labels.

    P2.03 scope: habitat_label path only. Fails explicitly if any of
    {floor, wall, ceiling} is absent from the raw info; RANSAC fallback
    is deferred until a scene actually requires it. Walls are oriented
    using the first floor's face centroid as the interior reference,
    NOT the object-centroid room_bbox.
    """
    grouped: dict[str, list[dict]] = {k: [] for k in STRUCTURAL_REQUIRED}
    for o in info["objects"]:
        name = o.get("class_name", "")
        if name in grouped:
            grouped[name].append(o)

    missing = [k for k in STRUCTURAL_REQUIRED if not grouped[k]]
    if missing:
        raise SystemExit(
            "Refusing to emit enriched-v2 structural surfaces: required "
            f"Habitat classes are absent: {sorted(missing)}. P2.03 implements "
            "the habitat_label path only; RANSAC is out of scope until a "
            "scene needs it."
        )

    for k in grouped:
        grouped[k].sort(key=lambda o: o["id"])

    floor_interior_hint = [0.0, 0.0, 1.0]
    ceiling_interior_hint = [0.0, 0.0, -1.0]

    surfaces: list[dict] = []
    floor_face_centroid_world: list[float] | None = None

    for inst in grouped["floor"]:
        rec, face_centroid = _build_structural_surface_record(
            inst, "floor", floor_interior_hint, None, z_min,
        )
        surfaces.append(rec)
        if floor_face_centroid_world is None:
            floor_face_centroid_world = face_centroid

    for inst in grouped["ceiling"]:
        rec, _ = _build_structural_surface_record(
            inst, "ceiling", ceiling_interior_hint, None, z_min,
        )
        surfaces.append(rec)

    for inst in grouped["wall"]:
        rec, _ = _build_structural_surface_record(
            inst, "wall", None, floor_face_centroid_world, z_min,
        )
        surfaces.append(rec)

    return surfaces


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


def import_replica(
    scene_dir: Path,
    scene_id: str,
    out_root: Path,
    keep_structural: bool,
    enriched_v2: bool = False,
) -> dict:
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
        extents = [float(s) / 2.0 for s in sizes]
        bbox_aabb_raw = _aabb_from_obb(center_world, extents, quat)
        kept.append(
            {
                "instance_id": o["id"],
                "label": name,
                "xyz_raw": [float(center_world[0]), float(center_world[1]), float(center_world[2])],
                "sizes": [float(sizes[0]), float(sizes[1]), float(sizes[2])],
                "bbox_obb_extents": extents,
                "bbox_obb_rotation_quat": [float(x) for x in quat],
                "bbox_aabb_raw": bbox_aabb_raw,
            }
        )

    if enriched_v2:
        z_min = min(r["bbox_aabb_raw"][0][2] for r in kept)
    else:
        # Legacy behavior: local-frame sizes approximate the world-frame AABB.
        z_min = min(r["xyz_raw"][2] - r["sizes"][2] / 2 for r in kept)
    objects_out = []
    for r in _suffix_duplicates(kept):
        x, y, z = r["xyz_raw"]
        centroid = [float(x), float(y), float(z - z_min)]
        obj = {
            "id": f"obj_{r['instance_id']}",
            "label": r["label"],
            "zone": None,
            "xyz": [round(v, 3) for v in centroid],
            "attributes": {
                "type": r["label"],
                "bbox_sizes": [round(s, 3) for s in r["sizes"]],
            },
        }
        if enriched_v2:
            bbox_min_raw, bbox_max_raw = r["bbox_aabb_raw"]
            obj["bbox_obb"] = {
                "center": _round_vec(centroid),
                "extents": _round_vec(r["bbox_obb_extents"]),
                "rotation_quat": _round_vec(r["bbox_obb_rotation_quat"]),
            }
            obj["bbox_aabb"] = [
                _round_vec(_translate_z(bbox_min_raw, -z_min)),
                _round_vec(_translate_z(bbox_max_raw, -z_min)),
            ]
        objects_out.append(obj)

    scene_graph = {"scene": scene_id, "objects": objects_out, "relations": []}
    structural_surfaces: list[dict] = []
    if enriched_v2:
        scene_graph["schema_version"] = ENRICHED_SCHEMA_VERSION
        structural_surfaces = _extract_structural_surfaces(info, z_min)
        scene_graph["structural_surfaces"] = structural_surfaces

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
    if enriched_v2:
        capture_meta["schema_version"] = ENRICHED_SCHEMA_VERSION
        per_type: dict[str, int] = {}
        for s in structural_surfaces:
            per_type[s["surface_type"]] = per_type.get(s["surface_type"], 0) + 1
        capture_meta["import_notes"].update(
            {
                "output_mode": "enriched_v2",
                "bbox_aabb_derived_from_rotated_obb": True,
                "structural_surface_count": len(structural_surfaces),
                "structural_surface_count_per_type": per_type,
                "structural_surface_sources": sorted({s["source"] for s in structural_surfaces}),
            }
        )

    out_dir = out_root / scene_id
    if enriched_v2:
        out_dir = out_dir / "enriched" / "v2"
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
    parser.add_argument(
        "--enriched-v2",
        action="store_true",
        help="Write schema-v2 OBB output under <scene>/enriched/v2/ without touching Phase 1 files.",
    )
    args = parser.parse_args()
    meta = import_replica(
        args.scene_dir,
        args.scene_id,
        args.out_root,
        args.keep_structural,
        enriched_v2=args.enriched_v2,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

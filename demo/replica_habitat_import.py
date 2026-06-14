"""Reusable importer: raw Replica `habitat/info_semantic.json` -> EntityArtifacts.

This is the real plug seam for a NEW Replica room. It reads the object-level
oriented bounding boxes Replica ships in `habitat/info_semantic.json` and produces
an `EntityArtifacts` bundle the Phase 6 pipeline consumes directly -- object boxes,
class labels, and gravity. No mesh or binary-segmentation decoding required.

Transform (reverse-engineered from room_0 and validated to 2 microns against its
committed enriched_v2 AABBs):

    world_corner = R(obj.orientation.rotation) @ (abb.center +- sizes/2) + obj.translation
    world_corner.z += z_translation          # floor offset; room_0 = 1.52665

The object's AABB is the axis-aligned bound of its 8 world OBB corners. The z
offset is cosmetic for furniture-top support (the relation is invariant to a global
vertical shift -- only relative gaps matter), so a new room imports correctly at any
consistent z_translation; room_0 uses 1.52665 only to byte-match enriched_v2.

Structural surfaces (floor/wall/ceiling) ARE imported here from the Habitat
`floor`/`wall`/`ceiling` semantic classes, so the full QA track -- "what is on
the floor?", "what is against the wall?" -- runs on a new room too, not just the
furniture-top support questions. The OBB->plane+polygon geometry is reused verbatim
from the canonical `importers/replica.py` so the surface shape matches the
committed enriched_v2 records; we only re-express it in this importer's
gravity-aligned frame (the same R_align + z_translation applied to the objects)
so surfaces and objects share one coordinate frame.

Multi-room note: the canonical importer orients every wall by the FIRST floor's
face centroid (it targets single rooms). A real apartment has many rooms, so here
each wall is oriented by its NEAREST floor's face centroid. Walls far from every
floor can still flip; that is a recall-only error (a missed against-the-wall
object), never a false positive, and it is reported, not hidden.

Usage:
    arts = import_habitat_room(Path("/.../datasets/replica/room_1"), "replica_room_1")
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

from extractors.base import (
    EntityArtifact,
    EntityArtifacts,
    EntityIdentity,
    ExtractionDiagnostics,
    StructuralSurface,
    SceneFrame,
)

# Reuse the canonical Habitat-OBB surface geometry so floor/wall/ceiling records
# match the committed enriched_v2 shape (single source of truth for the math).
from importers.replica import (
    _dot,
    _obb_face_corners,
    _orient_polygon,
    _orient_tag,
    _quat_axis_in_world,
    _quat_rotate,
    _scale,
    _sub,
    _thin_axis_index,
    _unit,
)
from common.types import Plane

ROOM_0_Z_TRANSLATION = 1.52665  # exact floor offset that reproduces room_0 enriched_v2
STRUCTURAL_CLASSES = ("floor", "wall", "ceiling")  # routed to surfaces, not entities
# A habitat instance labeled "wall" whose interior normal is not roughly
# horizontal is geometrically a slab/ledge, not a vertical wall. Emitting it as
# a surface lets on_surface treat it as a pseudo-shelf (an object "resting on a
# wall"). Drop such degenerate walls and count them.
WALL_MAX_VERTICAL_NORMAL = 0.3


def _gravity_align_matrix(gravity):
    """Rotation R mapping physical up (=-gravity) onto +z, so the imported
    scene is gravity-canonical (up = +z exactly). Different captures have
    slightly different gravity tilt; the Phase 6 AABB-top derivation requires
    axis-aligned up, so we align here rather than loosen the predicate.
    Rodrigues' formula; identity when already aligned, 180-deg flip handled."""
    gx, gy, gz = gravity
    m = math.sqrt(gx * gx + gy * gy + gz * gz) or 1.0
    ux, uy, uz = -gx / m, -gy / m, -gz / m          # up
    vx, vy, vz = uy * 1.0 - uz * 0.0, uz * 0.0 - ux * 1.0, ux * 0.0 - uy * 0.0  # up x z
    # up x zhat = (uy, -ux, 0); cos = up . zhat = uz
    vx, vy, vz = uy, -ux, 0.0
    c = uz
    if c > 1 - 1e-12:
        return ((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0))
    if c < -1 + 1e-12:                                # up points at -z: flip about x
        return ((1.0, 0, 0), (0, -1.0, 0), (0, 0, -1.0))
    k = 1.0 / (1.0 + c)
    # R = I + [v]x + [v]x^2 * k
    return (
        (1 + (-(vz * vz) - vy * vy) * k, (-vz) + (vx * vy) * k, (vy) + (vx * vz) * k),
        ((vz) + (vx * vy) * k, 1 + (-(vz * vz) - vx * vx) * k, (-vx) + (vy * vz) * k),
        ((-vy) + (vx * vz) * k, (vx) + (vy * vz) * k, 1 + (-(vy * vy) - vx * vx) * k),
    )


def _matvec(R, p):
    return (
        R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2],
        R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2],
        R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2],
    )


def _quat_to_matrix(q):
    """Rotation matrix from an [x, y, z, w] quaternion."""
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _world_aabb(obj, z_translation, R_align):
    """Axis-aligned bound of an object's oriented bbox, in the gravity-aligned
    world frame (R_align maps physical up onto +z)."""
    ob = obj["oriented_bbox"]
    cx, cy, cz = ob["abb"]["center"]
    hx, hy, hz = (s / 2.0 for s in ob["abb"]["sizes"])
    R = _quat_to_matrix(ob["orientation"]["rotation"])
    tx, ty, tz = ob["orientation"]["translation"]
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for sx, sy, sz in itertools.product((-1, 1), repeat=3):
        lx, ly, lz = cx + sx * hx, cy + sy * hy, cz + sz * hz
        w = (R[0][0] * lx + R[0][1] * ly + R[0][2] * lz + tx,
             R[1][0] * lx + R[1][1] * ly + R[1][2] * lz + ty,
             R[2][0] * lx + R[2][1] * ly + R[2][2] * lz + tz)
        w = _matvec(R_align, w)              # gravity-canonicalize
        w = (w[0], w[1], w[2] + z_translation)
        for i, v in enumerate(w):
            lo[i] = min(lo[i], v)
            hi[i] = max(hi[i], v)
    return (tuple(lo), tuple(hi))


def _surface_geometry_world(inst, interior_hint, interior_ref_world):
    """Interior-facing normal, 4 face corners, and face centroid for one
    structural instance, all in the raw Habitat world frame (pre gravity-align).

    Mirrors importers.replica._build_structural_surface_record's geometry: the
    surface is the OBB face perpendicular to the thin axis; the interior normal's
    sign is picked by an explicit hint (floor/ceiling) or by pointing toward an
    interior reference point (walls -> nearest floor-face centroid)."""
    ob = inst["oriented_bbox"]
    center_local = ob["abb"]["center"]
    quat = [float(x) for x in ob["orientation"]["rotation"]]
    extents = [float(s) / 2.0 for s in ob["abb"]["sizes"]]
    center_world = _quat_rotate(quat, center_local)
    thin_idx = _thin_axis_index(extents)
    thin_world_unit = _unit(_quat_axis_in_world(quat, thin_idx))
    if interior_hint is not None:
        ref_dir = interior_hint
    else:
        ref_dir = _sub(interior_ref_world, center_world)
    sign = 1.0 if _dot(thin_world_unit, ref_dir) >= 0 else -1.0
    interior_normal = _scale(thin_world_unit, sign)
    corners = _obb_face_corners(center_world, extents, quat, thin_idx, sign)
    corners = _orient_polygon(corners, interior_normal)
    face_centroid = [sum(c[k] for c in corners) / 4.0 for k in range(3)]
    return interior_normal, corners, face_centroid


def _to_final_frame_point(p, R_align, z_translation):
    w = _matvec(R_align, p)
    return (w[0], w[1], w[2] + z_translation)


def _structural_surfaces(info, R_align, z_translation):
    """Build StructuralSurface records (floor/wall/ceiling) in this importer's
    gravity-aligned + z-translated frame, the SAME frame the object AABBs use.

    Returns (surfaces, diagnostics_dict). Walls are oriented by the nearest
    floor-face centroid (multi-room robust); the count of walls with no floor
    in the scene is reported so a flip risk is never silent."""
    grouped = {k: [] for k in STRUCTURAL_CLASSES}
    for o in info["objects"]:
        name = str(o.get("class_name", "")).strip()
        if name in grouped:
            grouped[name].append(o)
    for k in grouped:
        grouped[k].sort(key=lambda o: o["id"])

    floor_centroids_world: list[list[float]] = []
    pending: list[tuple[dict, str, list[float] | None, list[float] | None]] = []

    for inst in grouped["floor"]:
        normal, corners, fc = _surface_geometry_world(inst, [0.0, 0.0, 1.0], None)
        floor_centroids_world.append(fc)
        pending.append((inst, "floor", normal, corners))
    for inst in grouped["ceiling"]:
        normal, corners, _ = _surface_geometry_world(inst, [0.0, 0.0, -1.0], None)
        pending.append((inst, "ceiling", normal, corners))

    walls_without_floor = 0
    for inst in grouped["wall"]:
        ob = inst["oriented_bbox"]
        wc = _quat_rotate(
            [float(x) for x in ob["orientation"]["rotation"]], ob["abb"]["center"]
        )
        if floor_centroids_world:
            ref = min(
                floor_centroids_world,
                key=lambda fc: sum((fc[i] - wc[i]) ** 2 for i in range(3)),
            )
        else:
            walls_without_floor += 1
            ref = [wc[0], wc[1], wc[2] + 1.0]  # last-resort: assume up is interior
        normal, corners, _ = _surface_geometry_world(inst, None, ref)
        pending.append((inst, "wall", normal, corners))

    surfaces: list[StructuralSurface] = []
    walls_dropped_non_vertical = 0
    for inst, stype, normal_w, corners_w in pending:
        n = _matvec(R_align, normal_w)  # direction: rotate only, no translation
        nmag = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) or 1.0
        n = (n[0] / nmag, n[1] / nmag, n[2] / nmag)
        if stype == "wall" and abs(n[2]) > WALL_MAX_VERTICAL_NORMAL:
            walls_dropped_non_vertical += 1  # degenerate "wall" (horizontal slab)
            continue
        poly = [_to_final_frame_point(c, R_align, z_translation) for c in corners_w]
        fc = tuple(sum(p[k] for p in poly) / 4.0 for k in range(3))
        d = -(n[0] * fc[0] + n[1] * fc[1] + n[2] * fc[2])
        iid = inst["id"]
        if stype == "floor":
            uid = f"floor_{iid}"
        elif stype == "ceiling":
            uid = f"ceiling_{iid}"
        else:
            uid = f"wall_{iid}_{_orient_tag(list(n))}"
        surfaces.append(StructuralSurface(
            surface_uid=uid,
            surface_type=stype,
            plane=Plane(a=n[0], b=n[1], c=n[2], d=d),
            polygon=[tuple(p) for p in poly],
            confidence=1.0,
            source="habitat_label",
        ))
    diag = {
        "n_floor": len(grouped["floor"]),
        "n_wall": len(grouped["wall"]),
        "n_ceiling": len(grouped["ceiling"]),
        "walls_without_floor": walls_without_floor,
        "walls_dropped_non_vertical": walls_dropped_non_vertical,
    }
    return surfaces, diag


def import_habitat_room(
    room_dir: Path,
    scene_id: str,
    *,
    z_translation: float = ROOM_0_Z_TRANSLATION,
    import_surfaces: bool = True,
    drop_classes: tuple[str, ...] = ("undefined", "non-plane", "plane"),
) -> EntityArtifacts:
    """Import a Replica room's labeled objects into an EntityArtifacts bundle.

    Reads <room_dir>/habitat/info_semantic.json. Each object becomes an
    EntityArtifact with object_uid='obj_<id>', display_label=class_name, and a
    world-frame AABB. Structural surfaces are left empty (support derives its own).
    """
    info_path = room_dir / "habitat" / "info_semantic.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    g = info["gravity_dir"]
    R_align = _gravity_align_matrix((float(g[0]), float(g[1]), float(g[2])))
    # After alignment the scene is gravity-canonical: up = +z exactly.
    gravity = (0.0, 0.0, -1.0)

    surfaces: list[StructuralSurface] = []
    surf_diag: dict = {"n_floor": 0, "n_wall": 0, "n_ceiling": 0, "walls_without_floor": 0}
    if import_surfaces:
        surfaces, surf_diag = _structural_surfaces(info, R_align, z_translation)

    entities: list[EntityArtifact] = []
    for obj in info["objects"]:
        cls = str(obj.get("class_name", "")).strip()
        if not cls or cls in drop_classes:
            continue
        # floor/wall/ceiling become StructuralSurfaces, never entities
        if import_surfaces and cls in STRUCTURAL_CLASSES:
            continue
        lo, hi = _world_aabb(obj, z_translation, R_align)
        centroid = tuple((lo[i] + hi[i]) / 2.0 for i in range(3))
        uid = f"obj_{obj['id']}"
        entities.append(EntityArtifact(
            identity=EntityIdentity(
                object_uid=uid,
                display_label=cls,
                aliases=[],
                source_instance_ref=f"habitat:{obj.get('node_id', obj['id'])}",
            ),
            bbox_aabb=(lo, hi),
            bbox_obb=None,
            centroid=centroid,
            geometry_handle=None,
            semantic_hypotheses=[],
            extraction_diagnostics={},
        ))

    raw = info_path.read_bytes()
    bundle_hash = "habitat_" + hashlib.sha256(raw).hexdigest()[:16]
    return EntityArtifacts(
        schema_version=2,
        bundle_hash=bundle_hash,
        scene_id=scene_id,
        frame=SceneFrame(gravity=gravity, canonical_forward=None,
                         canonical_right=None, units="meters",
                         notes="imported from Replica habitat/info_semantic.json"),
        representation_hash=bundle_hash,
        extractor_name="replica_habitat_import",
        extractor_version="0.1",
        entities=entities,
        structural_surfaces=surfaces,
        geometry_store_path=None,
        diagnostics=ExtractionDiagnostics(
            n_entities=len(entities),
            n_structural_surfaces=len(surfaces),
            runtime_seconds=0.0,
            coverage_score=1.0,
            notes=(
                f"objects + structural surfaces; floor={surf_diag['n_floor']} "
                f"wall={surf_diag['n_wall']} ceiling={surf_diag['n_ceiling']} "
                f"walls_without_floor={surf_diag['walls_without_floor']} "
                f"walls_dropped_non_vertical={surf_diag['walls_dropped_non_vertical']}"
                if import_surfaces else "objects-only import; surfaces skipped"
            ),
        ),
        notes={
            "source": "habitat/info_semantic.json",
            "z_translation": z_translation,
            "structural_surfaces": surf_diag,
        },
    )

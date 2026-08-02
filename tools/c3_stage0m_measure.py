"""C3 Stage 0m — read-only real-mesh measurement (approved protocol).

  python3 tools/c3_stage0m_measure.py            # full M1-M4 + verdict
  python3 tools/c3_stage0m_measure.py --census   # M1 only

Protocol: docs/c3_stage0m_measurement_protocol.md. Produces NO surface
artifact, evaluates NO gate, freezes NO constant. Frozen estimator code
(geometry/mesh_surfaces.py) is imported READ-ONLY: face geometry,
adjacency, region growth, and the boundary-loop extractor run unchanged;
the only parameter that varies is the residual band, and only inside the
declared measurement grid (a curve is a measurement; a chosen point would
be an estimator constant and is out of scope).

Labeled measurements (M2-M4) touch room_2 ONLY; room_1/office_0/room_0
receive the label-free census (M1) and stay clean for any successor.

Measurement-method choices (declared): largest-component coverage is the
sum of the component's plane-projected face areas divided by the oracle
polygon area (capped at 1; slab membership already requires the face
centroid inside the polygon, so overhang is marginal). Impostor grouping
(M3) reuses the frozen region growth with the residual band set to b.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry.mesh_surfaces import (
    DEFAULT_CONFIG,
    _boundary_loop,
    _face_adjacency,
    _face_geometry,
    _plane_basis,
    _points_in_polygon,
    _region_components,
    load_raw_triangle_mesh,
    transform_mesh,
)
from tools.c3_surface_run import SCENE_TO_SHORT, _load_generation_inputs, _plane_values

BANDS = (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05)
COVERAGE_REQ = {"floor": 0.85, "ceiling": 0.85, "wall": 0.75}
ATTRIB_SLAB_M = 0.05          # measurement choice, declared in the protocol
IMPOSTOR_AREA_M2 = 1.5        # cited from the frozen family's area gate
DEV_SCENE = "replica_room_2"


def pct(a: np.ndarray, q: float) -> float:
    return round(float(np.percentile(a, q)), 5) if len(a) else None


def m1_census(scene_id: str) -> dict:
    mesh_path, _, _ = _load_generation_inputs(scene_id)
    mesh = load_raw_triangle_mesh(mesh_path)
    tri, normals, area, family, good = _face_geometry(mesh, DEFAULT_CONFIG)
    az = np.abs(normals[good, 2])
    return {
        "scene_id": scene_id,
        "n_vertices": int(len(mesh.xyz)),
        "n_triangles": int(len(mesh.faces)),
        "n_source_quads": int(mesh.n_source_quads),
        "n_degenerate_faces": int(np.count_nonzero(~good)),
        "abs_nz_percentiles": {q: pct(az, q) for q in (10, 50, 90)},
        "n_horizontal_family": int(np.count_nonzero(family == 1)),
        "n_vertical_family": int(np.count_nonzero(family == 2)),
        "total_mesh_area_m2": round(float(area.sum()), 3),
    }


def _load_dev_scene():
    from demo.replica_habitat_import import import_habitat_room
    mesh_path, _, frame = _load_generation_inputs(DEV_SCENE)
    mesh = transform_mesh(load_raw_triangle_mesh(mesh_path),
                          np.asarray(frame["world_from_raw_rotation"]),
                          np.asarray(frame["world_from_raw_translation"]))
    lock = json.loads((REPO_ROOT / "tools" / "replica_scenes.lock.json")
                      .read_text())
    root = Path(lock["data_root_relative_to_repo"])
    arts = import_habitat_room(root / SCENE_TO_SHORT[DEV_SCENE], DEV_SCENE)
    return mesh, arts.structural_surfaces


def _subset_components(subset: np.ndarray, offsets: np.ndarray,
                       neighbors: np.ndarray) -> list[np.ndarray]:
    """Connected components of `subset` faces under the frozen adjacency."""
    in_set = np.zeros(len(offsets) - 1, dtype=bool)
    in_set[subset] = True
    seen = np.zeros_like(in_set)
    comps = []
    for seed in subset:
        if seen[seed]:
            continue
        seen[seed] = True
        queue, qpos = [int(seed)], 0
        while qpos < len(queue):
            cur = queue[qpos]
            qpos += 1
            for nb in neighbors[offsets[cur]:offsets[cur + 1]]:
                nb = int(nb)
                if in_set[nb] and not seen[nb]:
                    seen[nb] = True
                    queue.append(nb)
        comps.append(np.asarray(queue, dtype=np.int64))
    comps.sort(key=len, reverse=True)
    return comps


def measure_dev_scene() -> dict:
    mesh, oracle_surfaces = _load_dev_scene()
    tri, normals, area, family, good = _face_geometry(mesh, DEFAULT_CONFIG)
    centroids = tri.mean(axis=1)
    # raw shared-edge adjacency (family-agnostic: pass all-ones family)
    offsets, neighbors, _ = _face_adjacency(
        mesh.faces, np.ones(len(mesh.faces), dtype=np.int8))

    # ---- M2 + M4: per-oracle-surface cohesion curves ----
    m2 = []
    for s in oracle_surfaces:
        n, d = _plane_values(s)
        poly = np.asarray(s.polygon, dtype=np.float64)
        if poly.shape[1] == 2:                       # 2D polygon -> lift
            raise ValueError(f"unexpected 2D oracle polygon: {s.surface_uid}")
        origin = poly[0]
        poly2 = np.column_stack([
            (poly - origin) @ _plane_basis(n)[0],
            (poly - origin) @ _plane_basis(n)[1]])
        poly_area = abs(float(
            0.5 * np.sum(poly2[:, 0] * np.roll(poly2[:, 1], -1)
                         - np.roll(poly2[:, 0], -1) * poly2[:, 1])))
        vdist = np.abs(tri @ n + d)                  # [F,3] vertex distances
        cen2 = np.column_stack([
            (centroids - origin) @ _plane_basis(n)[0],
            (centroids - origin) @ _plane_basis(n)[1]])
        inside = _points_in_polygon(cen2, poly2)
        proj_area = area * np.abs(normals @ n)       # plane-projected areas
        bands = []
        for b in BANDS:
            slab = np.flatnonzero(good & inside & (vdist.max(axis=1) <= b))
            if not len(slab):
                bands.append({"band_m": b, "n_faces": 0})
                continue
            comps = _subset_components(slab, offsets, neighbors)
            largest = comps[0]
            cov = min(1.0, float(proj_area[largest].sum()) / poly_area)
            dists = vdist[slab].ravel()
            ang = np.degrees(np.arccos(np.clip(
                np.abs(normals[slab] @ n), 0, 1)))
            loop = _boundary_loop(mesh, largest, n, DEFAULT_CONFIG)
            bands.append({
                "band_m": b,
                "n_faces": int(len(slab)),
                "n_components": len(comps),
                "largest_component_faces": int(len(largest)),
                "largest_component_coverage": round(cov, 4),
                "coverage_ok": cov >= COVERAGE_REQ[s.surface_type],
                "boundary_loop_ok": loop is not None,
                "residual_p50_p90_p99_m": [pct(dists, 50), pct(dists, 90),
                                           pct(dists, 99)],
                "residual_rms_m": round(float(np.sqrt(np.mean(dists ** 2))), 5),
                "normal_dev_p50_p90_deg": [pct(ang, 50), pct(ang, 90)],
            })
        m2.append({"oracle_uid": s.surface_uid, "type": s.surface_type,
                   "oracle_polygon_area_m2": round(poly_area, 3),
                   "coverage_required": COVERAGE_REQ[s.surface_type],
                   "bands": bands})

    # ---- oracle labels for impostor attribution ----
    from demo.replica_mesh_import import _parse_semantic_ply
    from tools.c1_exact_eval import oracle_vertex_membership
    lock = json.loads((REPO_ROOT / "tools" / "replica_scenes.lock.json")
                      .read_text())
    root = Path(lock["data_root_relative_to_repo"])
    room = root / SCENE_TO_SHORT[DEV_SCENE]
    _, vidx, oid = _parse_semantic_ply(room / "habitat" / "mesh_semantic.ply")
    v_oracle = oracle_vertex_membership(vidx, oid, len(mesh.xyz))
    info = json.loads((room / "habitat" / "info_semantic.json").read_text())
    cls = {int(o["id"]): o.get("class_name", "?") for o in info["objects"]}
    planes = [(_plane_values(s)) for s in oracle_surfaces]

    # frozen family adjacency for region growth (family-aware, as frozen)
    f_off, f_nbr, _ = _face_adjacency(mesh.faces, family)

    def impostors_at(b: float) -> list[dict]:
        cfg = dataclasses.replace(DEFAULT_CONFIG, region_plane_residual_m=b)
        comps = _region_components(mesh, tri, normals, area, family,
                                   f_off, f_nbr, cfg)
        out = []
        for fam, seed, ids in comps:
            a = float(area[ids].sum())
            if a < IMPOSTOR_AREA_M2:
                continue
            w = area[ids] / max(area[ids].sum(), 1e-12)
            cn = normals[ids].T @ w
            cn = cn / max(np.linalg.norm(cn), 1e-12)
            cd = -float(np.dot(cn, (centroids[ids].T @ w)))
            near = False
            for on, od in planes:
                dot = abs(float(np.dot(cn, on)))
                ang = math.degrees(math.acos(min(1.0, dot)))
                off = abs(abs(cd) - abs(od))
                if ang <= 10.0 and off <= ATTRIB_SLAB_M:
                    near = True
                    break
            if near:
                continue
            face_oids = v_oracle[mesh.faces[ids]].ravel()
            face_oids = face_oids[face_oids >= 0]
            top = (int(np.bincount(face_oids).argmax())
                   if len(face_oids) else None)
            out.append({
                "family": "horizontal" if fam == 1 else "vertical",
                "mesh_area_m2": round(a, 3),
                "plane_z_or_offset": round(cd, 3),
                "attributed_object": (f"obj_{top} ({cls.get(top, '?')})"
                                      if top is not None else "unlabeled"),
            })
        return out

    m3 = {str(b): impostors_at(b) for b in BANDS}

    decision = decide(m2, m3)
    return {"m2_cohesion": m2, "m3_impostors": m3, **decision}


def decide(m2: list[dict], m3: dict) -> dict:
    """The approved decision rule, verbatim from the protocol."""
    per_band_clean = {}
    for i, b in enumerate(BANDS):
        rows = [s["bands"][i] for s in m2]
        viable = all(r.get("coverage_ok") for r in rows)
        loops = all(r.get("boundary_loop_ok") for r in rows)
        n_imp = len(m3[str(b)])
        per_band_clean[str(b)] = {"viable": viable, "boundary_ok": loops,
                                  "n_impostors": n_imp,
                                  "clean": viable and loops and n_imp == 0}
    clean_bands = [b for b in BANDS if per_band_clean[str(b)]["clean"]]
    viable_bands = [b for b in BANDS if per_band_clean[str(b)]["viable"]
                    and per_band_clean[str(b)]["boundary_ok"]]
    verdict = ("CLEAN" if clean_bands else
               "MIXED" if viable_bands else "OVERLAP")
    return {"per_band_decision": per_band_clean,
            "clean_bands": clean_bands, "viable_bands": viable_bands,
            "verdict": verdict}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--census", action="store_true", help="M1 only")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "runs" / "phase8_c3" / "stage0m")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "schema": "c3_stage0m_report_v1",
        "protocol": "docs/c3_stage0m_measurement_protocol.md",
        "read_only": True,
        "m1_census": [m1_census(s) for s in
                      ("replica_room_2", "replica_room_1",
                       "replica_office_0", "replica_room_0")],
    }
    if not args.census:
        report.update(measure_dev_scene())
    out = args.out_dir / "report.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    for c in report["m1_census"]:
        print(f"[M1] {c['scene_id']}: {c['n_triangles']} tris "
              f"({c['n_source_quads']} quads), "
              f"degenerate={c['n_degenerate_faces']}")
    if not args.census:
        for s in report["m2_cohesion"]:
            best = max((b for b in s["bands"] if b.get("n_faces")),
                       key=lambda b: b.get("largest_component_coverage", 0),
                       default=None)
            print(f"[M2] {s['oracle_uid']:<16} best coverage "
                  f"{best['largest_component_coverage'] if best else 0} "
                  f"@ band {best['band_m'] if best else '-'} "
                  f"(need {s['coverage_required']})")
        for b in BANDS:
            d = report["per_band_decision"][str(b)]
            print(f"[b={b:>6}] viable={d['viable']} boundary={d['boundary_ok']} "
                  f"impostors={d['n_impostors']} clean={d['clean']}")
        print(f"VERDICT: {report['verdict']}")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""C3.0-S generation and evaluation under the frozen protocol.

Generation and evaluation are separate commands by design:

  python3 tools/c3_surface_run.py generate --scene replica_room_2
  python3 tools/c3_surface_run.py evaluate --scene replica_room_2

`generate` reads only the raw mesh, the committed raw-input lock, and the
committed numeric frame sidecar.  It finalizes a hash-stamped surface artifact
before `evaluate` is allowed to open oracle surfaces or the human key.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.types import Plane
from extractors.base import EntityArtifacts, StructuralSurface
from geometry.mesh_surfaces import (
    DEFAULT_CONFIG,
    _plane_basis,
    _points_in_polygon,
    estimate_from_ply,
    sha256_file,
)


PROTOCOL = "docs/c3_0_mesh_surfaces_protocol.md"
FRAME_PATH = REPO_ROOT / "eval" / "fixtures" / "c3_0_frames.json"
LOCK_PATH = REPO_ROOT / "tools" / "replica_scenes.lock.json"
KEY_DIR = REPO_ROOT / "eval" / "questions" / "phase8"
DEFAULT_OUT = REPO_ROOT / "runs" / "phase8_c3"
SCENE_TO_SHORT = {
    "replica_room_0": "room_0",
    "replica_room_1": "room_1",
    "replica_room_2": "room_2",
    "replica_office_0": "office_0",
}
DEV_SCENE = "replica_room_2"
TRANSFER_SCENES = ("replica_room_1", "replica_office_0", "replica_room_0")
SURFACE_RELATIONS = ("SUPPORTS_FLOOR", "CONTACTS_SURFACE", "ATTACHED_TO")


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def _surface_to_dict(surface: StructuralSurface) -> dict[str, Any]:
    return {
        "surface_uid": surface.surface_uid,
        "surface_type": surface.surface_type,
        "plane": {"a": surface.plane.a, "b": surface.plane.b,
                  "c": surface.plane.c, "d": surface.plane.d},
        "polygon": ([list(map(float, p)) for p in surface.polygon]
                    if surface.polygon is not None else None),
        "confidence": surface.confidence,
        "source": surface.source,
    }


def _surface_from_dict(row: dict[str, Any]) -> StructuralSurface:
    p = row["plane"]
    return StructuralSurface(
        surface_uid=str(row["surface_uid"]),
        surface_type=str(row["surface_type"]),  # type: ignore[arg-type]
        plane=Plane(a=float(p["a"]), b=float(p["b"]),
                    c=float(p["c"]), d=float(p["d"])),
        polygon=([tuple(map(float, x)) for x in row["polygon"]]
                 if row.get("polygon") is not None else None),
        confidence=float(row["confidence"]),
        source=str(row["source"]),  # type: ignore[arg-type]
    )


def artifact_payload(scene_id: str, mesh_relpath: str, mesh_sha256: str,
                     frame: dict[str, Any], surfaces: list[StructuralSurface],
                     diagnostics: dict[str, Any]) -> dict[str, Any]:
    frame_sha = hashlib.sha256(_canonical_bytes(frame)).hexdigest()
    body = {
        "schema": "c3_0_surface_artifact_v1",
        "protocol": PROTOCOL,
        "scene_id": scene_id,
        "estimator": "mesh_region_fit_v1",
        "input_mesh": {"relpath": mesh_relpath, "sha256": mesh_sha256},
        "frame_input": frame,
        "frame_input_sha256": frame_sha,
        "config": DEFAULT_CONFIG.to_dict(),
        "config_sha256": DEFAULT_CONFIG.sha256(),
        "surfaces": [_surface_to_dict(s) for s in surfaces],
        "diagnostics": diagnostics,
        "isolation": ("Generated from raw mesh.ply plus committed numeric frame "
                      "input only; no semantic mesh, metadata, entities, keys, "
                      "questions, graph edges, or answers were read."),
    }
    body["output_sha256"] = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    return body


def verify_artifact(payload: dict[str, Any]) -> None:
    got = payload.get("output_sha256")
    body = dict(payload)
    body.pop("output_sha256", None)
    expected = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    if got != expected:
        raise ValueError(f"surface artifact hash mismatch: {got} != {expected}")
    if payload.get("estimator") != "mesh_region_fit_v1":
        raise ValueError("surface artifact estimator mismatch")
    if payload.get("config_sha256") != DEFAULT_CONFIG.sha256():
        raise ValueError("surface artifact config is not frozen C3.0-S config")


def _load_generation_inputs(scene_id: str) -> tuple[Path, str, dict[str, Any]]:
    if scene_id not in SCENE_TO_SHORT:
        raise ValueError(f"scene not predeclared: {scene_id}")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    short = SCENE_TO_SHORT[scene_id]
    mesh_rel = f"{short}/mesh.ply"
    rows = {r["relpath"]: r for r in lock["files"]}
    if mesh_rel not in rows:
        raise ValueError(f"raw mesh is not pinned: {mesh_rel}")
    root = Path(lock["data_root_relative_to_repo"])
    mesh_path = root / mesh_rel
    row = rows[mesh_rel]
    if mesh_path.stat().st_size != int(row["size"]):
        raise ValueError(f"raw mesh size mismatch: {mesh_path}")
    actual_sha = sha256_file(mesh_path)
    if actual_sha != row["sha256"]:
        raise ValueError(f"raw mesh hash mismatch: {mesh_path}")
    frames = json.loads(FRAME_PATH.read_text(encoding="utf-8"))["scenes"]
    if scene_id not in frames:
        raise ValueError(f"scene frame is not frozen: {scene_id}")
    return mesh_path, mesh_rel, frames[scene_id]


def generate(scene_id: str, out_dir: Path) -> Path:
    surface_dir = out_dir / "surfaces"
    artifact_path = surface_dir / f"{scene_id}.json"
    run_path = surface_dir / f"{scene_id}_execution.json"
    if artifact_path.exists() or run_path.exists():
        raise FileExistsError(
            f"C3.0-S refuses to overwrite an existing run: {artifact_path}")
    mesh_path, mesh_rel, frame = _load_generation_inputs(scene_id)
    t0 = time.perf_counter()
    result = estimate_from_ply(
        mesh_path, frame["world_from_raw_rotation"],
        frame["world_from_raw_translation"], DEFAULT_CONFIG)
    elapsed = time.perf_counter() - t0
    payload = artifact_payload(scene_id, mesh_rel, sha256_file(mesh_path),
                               frame, result.surfaces, result.diagnostics)
    verify_artifact(payload)
    surface_dir.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(_canonical_bytes(payload))
    execution = {
        "schema": "c3_0_execution_telemetry_v1",
        "scene_id": scene_id,
        "surface_artifact_output_sha256": payload["output_sha256"],
        "runtime_seconds": round(elapsed, 3),
        "note": "Telemetry is outside the deterministic surface-artifact hash.",
    }
    run_path.write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    return artifact_path


def _plane_values(surface: StructuralSurface) -> tuple[np.ndarray, float]:
    p = surface.plane
    n = np.asarray([p.a, p.b, p.c], dtype=np.float64)
    norm = float(np.linalg.norm(n))
    if norm <= 0:
        raise ValueError(f"zero plane normal: {surface.surface_uid}")
    return n / norm, float(p.d) / norm


def _plane_error(pred: StructuralSurface,
                 oracle: StructuralSurface) -> tuple[float, float]:
    pn, pd = _plane_values(pred)
    on, od = _plane_values(oracle)
    dot = float(np.clip(np.dot(pn, on), -1.0, 1.0))
    if dot < 0:
        pn, pd, dot = -pn, -pd, -dot
    angle = math.degrees(math.acos(float(np.clip(dot, -1.0, 1.0))))
    return angle, abs(pd - od)


def _compatible(pred: StructuralSurface, oracle: StructuralSurface) -> bool:
    if pred.surface_type != oracle.surface_type:
        return False
    angle, offset = _plane_error(pred, oracle)
    return angle <= 10.0 and offset <= 0.05


def _poly_grid(poly: np.ndarray, normal: np.ndarray, origin: np.ndarray,
               step: float = 0.01) -> tuple[np.ndarray, np.ndarray, float]:
    u, v = _plane_basis(normal)
    rel = poly - origin[None, :]
    p2 = np.column_stack([rel @ u, rel @ v])
    lo, hi = p2.min(axis=0), p2.max(axis=0)
    nx = max(1, int(math.ceil((hi[0] - lo[0]) / step)))
    ny = max(1, int(math.ceil((hi[1] - lo[1]) / step)))
    if nx * ny > 5_000_000:
        raise ValueError("surface metric grid exceeds safety cap")
    xs = lo[0] + (np.arange(nx) + 0.5) * step
    ys = lo[1] + (np.arange(ny) + 0.5) * step
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points2 = np.column_stack([xx.ravel(), yy.ravel()])
    return points2, p2, step * step


def _project_to_basis(poly: np.ndarray, normal: np.ndarray,
                      origin: np.ndarray) -> np.ndarray:
    u, v = _plane_basis(normal)
    rel = poly - origin[None, :]
    return np.column_stack([rel @ u, rel @ v])


def _coverage(reference: StructuralSurface,
              candidates: list[StructuralSurface]) -> tuple[float, float]:
    if reference.polygon is None:
        return 0.0, 0.0
    rn, _ = _plane_values(reference)
    origin = np.asarray(reference.polygon[0], dtype=np.float64)
    ref3 = np.asarray(reference.polygon, dtype=np.float64)
    points, ref2, cell_area = _poly_grid(ref3, rn, origin)
    inside_ref = _points_in_polygon(points, ref2)
    n_ref = int(np.count_nonzero(inside_ref))
    if n_ref == 0:
        return 0.0, 0.0
    covered = np.zeros(len(points), dtype=bool)
    for candidate in candidates:
        if candidate.polygon is None:
            continue
        c2 = _project_to_basis(np.asarray(candidate.polygon), rn, origin)
        covered |= _points_in_polygon(points, c2)
    n_cov = int(np.count_nonzero(inside_ref & covered))
    return n_cov / n_ref, n_ref * cell_area


def _spill(pred: StructuralSurface,
           candidates: list[StructuralSurface]) -> tuple[float, float]:
    if pred.polygon is None:
        return 1.0, 0.0
    pn, _ = _plane_values(pred)
    origin = np.asarray(pred.polygon[0], dtype=np.float64)
    p3 = np.asarray(pred.polygon, dtype=np.float64)
    points, p2, cell_area = _poly_grid(p3, pn, origin)
    inside_pred = _points_in_polygon(points, p2)
    n_pred = int(np.count_nonzero(inside_pred))
    if n_pred == 0:
        return 1.0, 0.0
    covered = np.zeros(len(points), dtype=bool)
    for candidate in candidates:
        if candidate.polygon is None:
            continue
        c2 = _project_to_basis(np.asarray(candidate.polygon), pn, origin)
        covered |= _points_in_polygon(points, c2)
    n_spill = int(np.count_nonzero(inside_pred & ~covered))
    return n_spill / n_pred, n_pred * cell_area


def surface_geometry_metrics(predicted: list[StructuralSurface],
                             oracle: list[StructuralSurface]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    angle_errors: list[float] = []
    offset_errors: list[float] = []
    type_summary: dict[str, dict[str, float | int | None]] = {}
    for st in ("floor", "wall", "ceiling"):
        ps = [s for s in predicted if s.surface_type == st]
        os = [s for s in oracle if s.surface_type == st]
        cov_num = cov_den = spill_num = spill_den = 0.0
        compatible_oracles = 0
        for o in os:
            compat = [p for p in ps if _compatible(p, o)]
            cov, area = _coverage(o, compat)
            cov_num += cov * area
            cov_den += area
            if compat:
                compatible_oracles += 1
                best = min(compat, key=lambda p: sum(_plane_error(p, o)))
                ae, oe = _plane_error(best, o)
                angle_errors.append(ae)
                offset_errors.append(oe)
            rows.append({"oracle_uid": o.surface_uid,
                         "compatible_predicted": [p.surface_uid for p in compat],
                         "coverage": round(cov, 6)})
        for p in ps:
            compat = [o for o in os if _compatible(p, o)]
            spill, area = _spill(p, compat)
            spill_num += spill * area
            spill_den += area
        type_summary[st] = {
            "n_predicted": len(ps), "n_oracle": len(os),
            "n_compatible_oracle": compatible_oracles,
            "oracle_area_coverage": round(cov_num / cov_den, 6) if cov_den else None,
            "estimated_spill": round(spill_num / spill_den, 6) if spill_den else None,
        }
    return {
        "by_type": type_summary,
        "per_oracle": rows,
        "compatible_plane_angular_error_median_deg":
            round(float(np.median(angle_errors)), 6) if angle_errors else None,
        "compatible_plane_offset_error_median_m":
            round(float(np.median(offset_errors)), 6) if offset_errors else None,
    }


def _candidate_arts(b: EntityArtifacts, surfaces: list[StructuralSurface],
                    artifact_sha: str) -> EntityArtifacts:
    bundle_hash = "c3s_" + hashlib.sha256(
        f"{b.bundle_hash}:{artifact_sha}".encode("utf-8")).hexdigest()[:16]
    return replace(
        b,
        bundle_hash=bundle_hash,
        representation_hash=bundle_hash,
        extractor_name="replica_mesh_import+c3_mesh_region_fit",
        extractor_version="0.2+c3s1",
        structural_surfaces=surfaces,
        diagnostics=replace(
            b.diagnostics, n_structural_surfaces=len(surfaces),
            notes=b.diagnostics.notes + "; structural surfaces=mesh_region_fit_v1"),
        notes={**b.notes,
               "surface_source": "mesh_region_fit_v1",
               "surface_artifact_output_sha256": artifact_sha,
               "c3_stage": "C3.0-S; frame and labels remain injected"},
    )


def _surface_slice(qa: dict[str, Any]) -> dict[str, Any]:
    rows = [qa["per_relation"].get(rel, {}) for rel in SURFACE_RELATIONS]
    hit = sum(int(r.get("n_hit", 0)) for r in rows)
    cited = sum(int(r.get("n_cited", 0)) for r in rows)
    expected = sum(int(r.get("n_expected", 0)) for r in rows)
    return {
        "n_hit": hit, "n_cited": cited, "n_expected": expected,
        "micro_precision": round(hit / cited, 6) if cited else None,
        "micro_recall": round(hit / expected, 6) if expected else None,
    }


def _near_wall_members(bundle) -> set[str]:
    wall_uids = {s.uid for s in bundle.structural_surfaces
                 if s.surface_type == "wall"}
    return {e.source.uid for e in bundle.edges
            if e.type == "NEAR_SURFACE" and e.source.kind == "entity" and
            e.target.kind == "surface" and e.target.uid in wall_uids}


def _set_f1(a: set[str], b: set[str]) -> dict[str, Any]:
    hit = len(a & b)
    precision = hit / len(a) if a else (1.0 if not b else 0.0)
    recall = hit / len(b) if b else (1.0 if not a else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n_candidate": len(a), "n_reference": len(b), "n_hit": hit,
            "precision": round(precision, 6), "recall": round(recall, 6),
            "f1": round(f1, 6)}


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()


def _metric_number(value: float | int | None, missing: float) -> float:
    return missing if value is None else float(value)


def evaluate(scene_id: str, out_dir: Path) -> Path:
    # Keep evaluation-only dependencies behind this boundary.  In particular,
    # importing the runner for `generate` must not import either Replica oracle
    # importer, the graph builder, reasoner, or key scorer.
    from demo.question_battery import _runs
    from demo.replica_habitat_import import import_habitat_room
    from demo.replica_mesh_import import import_mesh_room
    from graph.builder import build_graph
    from reasoner.base import CompletenessProfile, ExecutionContext
    from reasoner.compiler_rules import RulesCompiler
    from reasoner.executor import RulesExecutor
    from reasoner.router import Router
    from reasoner.verbalizer import StandardVerbalizer
    from tools.c1_joint_ceiling import score_against_key

    artifact_path = out_dir / "surfaces" / f"{scene_id}.json"
    if not artifact_path.exists():
        raise FileNotFoundError(f"generate must finalize first: {artifact_path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    verify_artifact(artifact)
    if artifact["scene_id"] != scene_id:
        raise ValueError("surface artifact scene mismatch")
    short = SCENE_TO_SHORT[scene_id]
    root = Path(json.loads(LOCK_PATH.read_text())["data_root_relative_to_repo"])
    room_dir = root / short
    key_path = KEY_DIR / f"{scene_id}_qa.json"
    key = json.loads(key_path.read_text(encoding="utf-8"))
    if key.get("answer_key_type") != "human_verified":
        raise ValueError(f"human_verified key required: {key_path}")

    predicted = [_surface_from_dict(r) for r in artifact["surfaces"]]
    oracle_arts = import_habitat_room(room_dir, scene_id)
    b = import_mesh_room(room_dir, scene_id)
    candidate = _candidate_arts(b, predicted, artifact["output_sha256"])

    b_graph, _ = build_graph(b, _runs(), density_policy="phase2_telemetry_only")
    c_graph, _ = build_graph(candidate, _runs(), density_policy="phase2_telemetry_only")
    router = Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                    verbalizer=StandardVerbalizer())
    ctx = ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))
    b_qa = score_against_key(key, b_graph, router, ctx, None)
    c_qa = score_against_key(key, c_graph, router, ctx, None)
    geometry = surface_geometry_metrics(predicted, oracle_arts.structural_surfaces)
    near_f1 = _set_f1(_near_wall_members(c_graph), _near_wall_members(b_graph))
    b_slice, c_slice = _surface_slice(b_qa), _surface_slice(c_qa)

    by = geometry["by_type"]
    if scene_id == DEV_SCENE:
        gates = {
            "G1": (by["floor"]["n_predicted"] == 1 and
                   by["ceiling"]["n_predicted"] == 1 and
                   by["floor"]["n_compatible_oracle"] == 1 and
                   by["ceiling"]["n_compatible_oracle"] == 1),
            "G2": (_metric_number(by["floor"]["oracle_area_coverage"], 0) >= 0.85 and
                   _metric_number(by["ceiling"]["oracle_area_coverage"], 0) >= 0.85 and
                   _metric_number(by["floor"]["estimated_spill"], 1) <= 0.10 and
                   _metric_number(by["ceiling"]["estimated_spill"], 1) <= 0.10),
            "G3": (_metric_number(by["wall"]["oracle_area_coverage"], 0) >= 0.75 and
                   _metric_number(by["wall"]["estimated_spill"], 1) <= 0.15),
            "G4": (_metric_number(geometry["compatible_plane_angular_error_median_deg"], 999) <= 5.0 and
                   _metric_number(geometry["compatible_plane_offset_error_median_m"], 999) <= 0.03),
            "G5": _metric_number(c_slice["micro_precision"], 0) >= 0.90,
            "G6": _metric_number(c_slice["micro_recall"], 0) >= 0.48,
            "G7": float(near_f1["f1"]) >= 0.85,
        }
    else:
        gates = {
            "H1": (by["floor"]["n_predicted"] == 1 and
                   by["ceiling"]["n_predicted"] == 1 and
                   by["floor"]["n_compatible_oracle"] == 1 and
                   by["ceiling"]["n_compatible_oracle"] == 1),
            "H2": (_metric_number(by["floor"]["oracle_area_coverage"], 0) >= 0.80 and
                   _metric_number(by["ceiling"]["oracle_area_coverage"], 0) >= 0.80 and
                   _metric_number(by["floor"]["estimated_spill"], 1) <= 0.15 and
                   _metric_number(by["ceiling"]["estimated_spill"], 1) <= 0.15),
            "H3": (_metric_number(by["wall"]["oracle_area_coverage"], 0) >= 0.70 and
                   _metric_number(by["wall"]["estimated_spill"], 1) <= 0.20),
            "H4": (c_slice["micro_precision"] is not None and
                   b_slice["micro_precision"] is not None and
                   c_slice["micro_precision"] >= b_slice["micro_precision"] - 0.05),
            "H5": (c_slice["micro_recall"] is not None and
                   b_slice["micro_recall"] is not None and
                   c_slice["micro_recall"] >= b_slice["micro_recall"] - 0.05),
            "H6": float(near_f1["f1"]) >= 0.80,
        }

    report = {
        "schema": "c3_0_surface_report_v1",
        "protocol": PROTOCOL,
        "scene_id": scene_id,
        "git_commit": _git_commit(),
        "surface_artifact_output_sha256": artifact["output_sha256"],
        "isolation": ("B boxes, labels, and frame held fixed; only structural "
                      "surfaces changed to mesh_region_fit_v1."),
        "geometry": geometry,
        "b_surface_slice": b_slice,
        "candidate_surface_slice": c_slice,
        "near_wall_membership_vs_b": near_f1,
        "n_graph_edges": {"B": len(b_graph.edges), "B+S_mesh": len(c_graph.edges)},
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{scene_id}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("generate", "evaluate"))
    ap.add_argument("--scene", required=True, choices=tuple(SCENE_TO_SHORT))
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if args.command == "generate":
        path = generate(args.scene, args.out_dir)
    else:
        path = evaluate(args.scene, args.out_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""C1.01 — model-independent segmenter runner interface + output bundle.

A segmentation backend (local or remote GPU) implements MeshSegmenter and
writes a SegmentationOutput bundle to a directory:

    <out_dir>/
      vertex_instance_ids.npy   int64 [n_vertices]; -1 = unassigned/background
      instance_table.json       [{"instance_id", "n_vertices", "confidence"}]
      meta.json                 input mesh hash, segmenter name/version/config,
                                runtime + hardware diagnostics, output hash

The bundle is immutable evidence: the local graph path only ever LOADS it
(load_segmentation_output), so inference can happen anywhere. Determinism is
checkable via output_sha256 (hash over the dense assignment bytes + the
canonicalized instance table), independent of runtime/hardware fields.

Isolation rule (contract G2/G5): a MeshSegmenter receives ONLY the raw
mesh.ply path. It must not read info_semantic.json or mesh_semantic.ply;
oracle joins happen in evaluation-only code (tools/c1_exact_eval.py and the
derived-bundle builder), never in the segmenter.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class MeshSegmenterConfig:
    """Backend-agnostic config envelope. params_json is a canonical JSON
    string (sorted keys) so configs stay hashable and diffable."""
    params_json: str = "{}"

    def params(self) -> dict:
        return json.loads(self.params_json)

    @staticmethod
    def from_params(params: dict) -> "MeshSegmenterConfig":
        return MeshSegmenterConfig(params_json=json.dumps(params, sort_keys=True))


@dataclass
class SegmentationOutput:
    """Dense per-vertex instance assignment + provenance."""
    input_mesh_sha256: str
    n_vertices: int
    segmenter_name: str
    segmenter_version: str
    config_params_json: str
    vertex_instance_ids: np.ndarray          # int64 [n_vertices], -1 = unassigned
    instance_confidence: dict[int, float] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    hardware: str = ""
    output_sha256: str = ""                  # filled by finalize()

    def instance_ids(self) -> list[int]:
        ids = np.unique(self.vertex_instance_ids)
        return [int(i) for i in ids if i >= 0]

    def finalize(self) -> "SegmentationOutput":
        self.output_sha256 = _output_hash(self)
        return self


class MeshSegmenter(Protocol):
    name: str
    version: str

    def segment(
        self,
        mesh_path: Path,
        config: MeshSegmenterConfig,
        out_dir: Path,
    ) -> SegmentationOutput: ...


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _instance_table(seg: SegmentationOutput) -> list[dict]:
    ids, counts = np.unique(seg.vertex_instance_ids, return_counts=True)
    table = []
    for i, c in zip(ids, counts):
        if i < 0:
            continue
        table.append({
            "instance_id": int(i),
            "n_vertices": int(c),
            "confidence": seg.instance_confidence.get(int(i)),
        })
    return table


def _output_hash(seg: SegmentationOutput) -> str:
    """Deterministic hash over the dense assignment + canonical instance
    table. Excludes runtime/hardware so re-runs are comparable across hosts."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(seg.vertex_instance_ids, dtype=np.int64).tobytes())
    h.update(json.dumps(_instance_table(seg), sort_keys=True).encode())
    h.update(seg.input_mesh_sha256.encode())
    return h.hexdigest()


def save_segmentation_output(seg: SegmentationOutput, out_dir: Path) -> Path:
    """Write the immutable bundle. Validates before writing; finalizes the
    output hash if not already set."""
    _validate(seg)
    if not seg.output_sha256:
        seg.finalize()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "vertex_instance_ids.npy",
            np.ascontiguousarray(seg.vertex_instance_ids, dtype=np.int64))
    (out_dir / "instance_table.json").write_text(
        json.dumps(_instance_table(seg), indent=1, sort_keys=True), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps({
        "input_mesh_sha256": seg.input_mesh_sha256,
        "n_vertices": seg.n_vertices,
        "segmenter_name": seg.segmenter_name,
        "segmenter_version": seg.segmenter_version,
        "config_params_json": seg.config_params_json,
        "runtime_seconds": seg.runtime_seconds,
        "hardware": seg.hardware,
        "output_sha256": seg.output_sha256,
        "written_unix": time.time(),
    }, indent=1, sort_keys=True), encoding="utf-8")
    return out_dir


def load_segmentation_output(out_dir: Path) -> SegmentationOutput:
    """Load + re-validate a bundle. Raises on any tampering/drift: assignment
    length mismatch, out-of-range ids, or output-hash mismatch."""
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    ids = np.load(out_dir / "vertex_instance_ids.npy")
    table = json.loads((out_dir / "instance_table.json").read_text(encoding="utf-8"))
    seg = SegmentationOutput(
        input_mesh_sha256=meta["input_mesh_sha256"],
        n_vertices=int(meta["n_vertices"]),
        segmenter_name=meta["segmenter_name"],
        segmenter_version=meta["segmenter_version"],
        config_params_json=meta["config_params_json"],
        vertex_instance_ids=ids.astype(np.int64),
        instance_confidence={
            int(r["instance_id"]): r["confidence"] for r in table
            if r.get("confidence") is not None
        },
        runtime_seconds=float(meta.get("runtime_seconds", 0.0)),
        hardware=str(meta.get("hardware", "")),
        output_sha256=meta["output_sha256"],
    )
    _validate(seg)
    recomputed = _output_hash(seg)
    if recomputed != seg.output_sha256:
        raise ValueError(
            f"segmentation bundle hash mismatch in {out_dir}: "
            f"meta says {seg.output_sha256[:16]}..., recomputed {recomputed[:16]}...")
    return seg


def _validate(seg: SegmentationOutput) -> None:
    ids = seg.vertex_instance_ids
    if ids.ndim != 1 or len(ids) != seg.n_vertices:
        raise ValueError(
            f"assignment length mismatch: got {ids.shape}, expected ({seg.n_vertices},)")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"vertex_instance_ids must be integer, got {ids.dtype}")
    if ids.min(initial=0) < -1:
        raise ValueError("instance ids below -1 (only -1 marks unassigned)")
    if not seg.input_mesh_sha256:
        raise ValueError("input_mesh_sha256 is required")

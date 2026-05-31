"""Phase 1 exit gates: artifact-emission wrappers around proven logic.

Two gates, both writing artifacts to scenes/replica_room_0/eval/:

  P1.08 compat reproduction gate:
    Runs the oracle pipeline (adapter -> mesh repr -> extractor -> builder)
    in compat mode and asserts the produced edge set equals the legacy
    5,414-edge Replica artifact byte-for-byte by edge key (no tolerance,
    per phase0_design.md §11.1 amendment).
    Emits oracle_adapter_repro_diff.json. The diff lists 'missing' and
    'extra' must both be empty for the gate to pass.

  P1.09 sparse density gate:
    Runs the same pipeline in sparse mode and asserts
    logical_edges_total / entity_count <= 14 with symmetric edges and
    inverse pairs counted once. Per-family counts emitted in the report.
    Emits sparse_density_report.json.

Run as a script:
    python tools/phase1_gates.py

Exit code 0 if both gates pass, 1 otherwise.

The actual gate logic was tested in P1.06 + P1.07. These wrappers exist
to produce the canonical artifacts and provide a single runnable entry
point for CI / manual verification.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.base import ReconstructionConfig
from adapters.oracle_replica import (
    OracleReplicaAdapter, build_replica_capture_bundle,
)
from extractors.base import InstanceExtractorConfig
from extractors.oracle_replica import OracleReplicaExtractor
from graph.builder import (
    ExtractorRun, GraphBuildError, SPARSE_DENSITY_LIMIT, build_graph,
)
from graph.relations.base import edge_key
from graph.relations.directional import DirectionalConfig, DirectionalExtractor
from graph.relations.proximity import ProximityConfig, ProximityExtractor
from representations.mesh import MeshRepresentation


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
LEGACY_REPLICA_RELATIONS = REPLICA_SCENE_DIR / "computed_relations" / "scene_graph.json"
EVAL_DIR = REPLICA_SCENE_DIR / "eval"
COMPAT_DIFF_PATH = EVAL_DIR / "oracle_adapter_repro_diff.json"
SPARSE_REPORT_PATH = EVAL_DIR / "sparse_density_report.json"


def _build_oracle_artifacts():
    capture = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    repr_bundle = OracleReplicaAdapter().reconstruct(
        capture, ReconstructionConfig(name="oracle_replica", version="0.1", params={}),
    )
    representation = MeshRepresentation(bundle=repr_bundle)
    return OracleReplicaExtractor().extract(
        representation,
        InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_compat_reproduction_gate(*, out_path: Path = COMPAT_DIFF_PATH) -> bool:
    """P1.08 gate: compat-mode build reproduces the legacy artifact
    exactly. Writes oracle_adapter_repro_diff.json. Returns True if the
    diff is empty (gate passes)."""
    artifacts = _build_oracle_artifacts()
    runs = [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="compat")),
        ExtractorRun(ProximityExtractor(), ProximityConfig(mode="compat")),
    ]
    bundle, diag = build_graph(artifacts, runs)
    produced_keys = {edge_key(e) for e in bundle.edges}

    legacy = json.loads(LEGACY_REPLICA_RELATIONS.read_text(encoding="utf-8"))
    legacy_keys = {
        ("entity", r["source"], r["type"], "entity", r["target"])
        for r in legacy["relations"]
    }

    missing = sorted(legacy_keys - produced_keys)
    extra = sorted(produced_keys - legacy_keys)
    passed = not missing and not extra

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "gate": "compat_reproduction",
        "exported_at": _now(),
        "scene_id": bundle.scene_id,
        "legacy_artifact": str(LEGACY_REPLICA_RELATIONS.relative_to(REPO_ROOT)),
        "produced_total": len(produced_keys),
        "expected_total": len(legacy_keys),
        "missing": [list(k) for k in missing],
        "extra": [list(k) for k in extra],
        "pass": passed,
        "graph_bundle_hash": bundle.bundle_hash,
        "entity_bundle_hash": artifacts.bundle_hash,
        "extractor_versions": dict(diag.extractor_versions),
        "mode": diag.mode,
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return passed


def run_sparse_density_gate(
    *,
    out_path: Path = SPARSE_REPORT_PATH,
    limit_ratio: float = SPARSE_DENSITY_LIMIT,
) -> bool:
    """P1.09 gate: sparse-mode build satisfies the density guardrail
    (logical_edges / entity_count <= limit_ratio). Writes
    sparse_density_report.json. Returns True if the build succeeded
    AND the ratio is within the limit.

    The GraphBuilder already raises on guardrail violation in sparse
    mode; this wrapper catches the error so a failure report is still
    emitted (with the offending ratio captured from BuildDiagnostics
    where possible, or 'unknown' when the build raised before producing
    diagnostics).
    """
    artifacts = _build_oracle_artifacts()
    runs = [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse")),
        ExtractorRun(ProximityExtractor(), ProximityConfig(mode="sparse")),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    common = {
        "gate": "sparse_density",
        "exported_at": _now(),
        "scene_id": artifacts.scene_id,
        "entity_count": len(artifacts.entities),
        "limit_ratio": limit_ratio,
    }
    try:
        bundle, diag = build_graph(artifacts, runs)
    except GraphBuildError as e:
        report = {
            **common,
            "pass": False,
            "build_error": str(e),
            "physical_edges_total": None,
            "logical_edges_total": None,
            "actual_ratio": None,
            "physical_edges_per_type": {},
            "per_family": [],
        }
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return False

    ratio = diag.logical_edges_total / max(1, len(artifacts.entities))
    per_family = [
        {
            "extractor": d.extractor,
            "version": d.version,
            "mode": d.mode,
            "physical_edges_per_type": dict(d.physical_edges_per_type),
            "physical_edges_total": d.physical_edges_total,
            "logical_edges_total": d.logical_edges_total,
            "rejections_per_type": dict(d.rejections_per_type),
            "runtime_ms": d.runtime_ms,
        }
        for d in diag.per_extractor
    ]
    report = {
        **common,
        "pass": ratio <= limit_ratio,
        "build_error": None,
        "physical_edges_total": diag.physical_edges_total,
        "logical_edges_total": diag.logical_edges_total,
        "actual_ratio": ratio,
        "physical_edges_per_type": dict(diag.edges_emitted_per_type),
        "per_family": per_family,
        "graph_bundle_hash": bundle.bundle_hash,
        "mode": diag.mode,
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return ratio <= limit_ratio


def main() -> int:
    print("=== P1.08 compat reproduction gate ===")
    compat_pass = run_compat_reproduction_gate()
    print(f"  -> {COMPAT_DIFF_PATH.relative_to(REPO_ROOT)}")
    print(f"  pass: {compat_pass}")

    print("\n=== P1.09 sparse density gate ===")
    sparse_pass = run_sparse_density_gate()
    print(f"  -> {SPARSE_REPORT_PATH.relative_to(REPO_ROOT)}")
    print(f"  pass: {sparse_pass}")

    if compat_pass and sparse_pass:
        print("\nBoth gates pass.")
        return 0
    print("\nOne or more gates failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

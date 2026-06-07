"""Phase 4 exit gate (P4.06).

Closeout verifier for the ON_SURFACE rest-contact relation + the SUPPORTS
derived view. Writes ONLY its own report:

  scenes/replica_room_0/eval/phase4_exit_gate_report.json

This gate is a VERIFIER, not a generator. It does NOT invoke any prior
phase's gate/telemetry main() (those have write side-effects). It reads the
committed Phase 2/3 reports and re-derives Phase 4 facts in-memory.

Blocking gates (all on real Replica room_0):
  G1   rest-contact determinism: two OnSurfaceExtractor runs produce
       identical ON_SURFACE edge_id sets AND edge keys.
  G2   subset: every ON_SURFACE (entity, surface) pair is a polygon-mode
       NEAR_SURFACE pair. Violation count 0.
  G3   clean inverse: len(support_facts(bundle)) == len(ON_SURFACE edges).
  G4   no materialized SUPPORTS edges (count == 0).
  G5   P4 smoke fixture: all synthetic cases via the extractor + real F1
       (obj_39 stool / floor_25).
  G6   default path preserved: committed Phase 2/3 exit-gate reports both
       have overall_blocking_pass == true (trusted, NOT re-derived — see
       module note), AND an in-memory default Phase 2/3 candidate build
       contains zero ON_SURFACE edges (ON_SURFACE is in no default path).
  G7   prior artifacts untouched: byte snapshot of git-tracked eval JSON
       artifacts (except this gate's own report) is unchanged before/after
       the gate run.
  G8   threshold-ordering guard enforced: OnSurfaceConfig(contact_threshold_m=0.10)
       raises (hypot(0.10, 0.0) > near_surface_threshold_m 0.05).
  schema  graph serde v3: an ON_SURFACE-bearing bundle round-trips under v3,
       and a manually-downgraded v2 manifest is rejected (strict, no
       migration). Run inline in a temp dir.

Determinism: no timestamp; sorted keys + trailing newline. Re-running
produces a byte-identical report (tested). The committed Phase 1/2/3
artifacts and the P4.05 telemetry are read, never rewritten (tested).

Skip-on-missing: enriched-v2 importer output + the committed Phase 2/3
reports must exist.

Run: python tools/phase4_exit_gate.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.equality import array_aware_equal
from common.serde import SchemaVersionError
from common.types import Plane, SceneFrame
from extractors.base import (
    EntityArtifact, EntityArtifacts, EntityIdentity, ExtractionDiagnostics,
    SemanticHypothesis, StructuralSurface,
)
from extractors.serde import CURRENT_SCHEMA_VERSION as ENT_SCHEMA_VERSION
from graph.builder import ExtractorRun, build_graph
from graph.relations.on_surface import (
    ON_SURFACE_VERSION, OnSurfaceConfig, OnSurfaceExtractor,
)
from graph.relations.surface import SurfaceProximityConfig, SurfaceProximityExtractor
from graph.schema import Edge, GraphRef
from graph.serde import dump_scene_graph_bundle, load_scene_graph_bundle
from graph.views.support import support_facts
from representations.mesh import MeshRepresentation
# Reuse Phase 2 helpers (real artifacts + default candidate runs). Importing
# is read-only; we never call phase2/phase3 gate main().
from tools.phase2_exit_gate import _phase2_runs, _real_replica_artifacts


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
REPLICA_V2_DIR = REPLICA_SCENE_DIR / "enriched" / "v2"
EVAL_DIR = REPLICA_SCENE_DIR / "eval"
ARTIFACT_PATH = EVAL_DIR / "phase4_exit_gate_report.json"
PHASE2_REPORT = EVAL_DIR / "phase2_exit_gate_report.json"
PHASE3_REPORT = EVAL_DIR / "phase3_exit_gate_report.json"
P4_FIXTURE_PATH = REPO_ROOT / "eval" / "questions" / "phase4_on_surface_smoke.json"


def _on_surface_bundle(artifacts):
    bundle, _diag = build_graph(
        artifacts,
        [ExtractorRun(OnSurfaceExtractor(), OnSurfaceConfig())],
        density_policy="phase2_telemetry_only",
    )
    return bundle


# --- G1 determinism ------------------------------------------------------


def _gate_g1(artifacts) -> tuple[bool, dict]:
    extractor = OnSurfaceExtractor()
    e1, _ = extractor.extract(artifacts, OnSurfaceConfig())
    e2, _ = extractor.extract(artifacts, OnSurfaceConfig())
    ids1 = sorted(e.edge_id for e in e1)
    ids2 = sorted(e.edge_id for e in e2)
    keys1 = sorted((e.source.uid, e.type, e.target.uid) for e in e1)
    keys2 = sorted((e.source.uid, e.type, e.target.uid) for e in e2)
    ok = ids1 == ids2 and keys1 == keys2
    return ok, {
        "two_run_edge_ids_match": ids1 == ids2,
        "two_run_edge_keys_match": keys1 == keys2,
        "edge_count": len(e1),
    }


# --- G2 subset vs polygon-mode NEAR_SURFACE ------------------------------


def _gate_g2(artifacts) -> tuple[bool, dict]:
    on_edges, _ = OnSurfaceExtractor().extract(artifacts, OnSurfaceConfig())
    near_edges, _ = SurfaceProximityExtractor().extract(
        artifacts, SurfaceProximityConfig(use_polygon_clip=True),
    )
    on_pairs = {(e.source.uid, e.target.uid) for e in on_edges}
    near_pairs = {(e.source.uid, e.target.uid) for e in near_edges}
    violations = sorted(on_pairs - near_pairs)
    return not violations, {
        "on_surface_pairs": len(on_pairs),
        "near_surface_polygon_pairs": len(near_pairs),
        "violation_count": len(violations),
        "violations": [{"entity_uid": e, "surface_uid": s} for e, s in violations[:10]],
    }


# --- G3 clean inverse / G4 no materialized SUPPORTS ----------------------


def _gate_g3_g4(bundle) -> tuple[bool, bool, dict]:
    on_edges = [e for e in bundle.edges if e.type == "ON_SURFACE"]
    materialized = sum(1 for e in bundle.edges if e.type == "SUPPORTS")
    facts = support_facts(bundle)  # raises if invariant violated
    g3 = len(facts) == len(on_edges)
    g4 = materialized == 0
    return g3, g4, {
        "on_surface_edges": len(on_edges),
        "support_facts": len(facts),
        "materialized_supports": materialized,
    }


# --- G5 P4 smoke fixture -------------------------------------------------


def _scene_frame() -> SceneFrame:
    return SceneFrame(
        gravity=(0.0, 0.0, -1.0), canonical_forward=None,
        canonical_right=None, units="meters", notes="",
    )


def _synth_artifacts(case, surface) -> EntityArtifacts:
    mn = case["entity_aabb"]["min"]
    mx = case["entity_aabb"]["max"]
    entity = EntityArtifact(
        identity=EntityIdentity(
            object_uid=case["id"], display_label=case["id"], aliases=[],
            source_instance_ref=case["id"],
        ),
        bbox_aabb=((mn[0], mn[1], mn[2]), (mx[0], mx[1], mx[2])),
        bbox_obb=None, centroid=tuple(case["entity_centroid"]),
        geometry_handle=None,
        semantic_hypotheses=[SemanticHypothesis(label=case["id"], confidence=1.0, source="fx")],
        embedding=None, extraction_diagnostics={},
    )
    p = surface["plane"]
    sr = StructuralSurface(
        surface_uid=surface["surface_uid"], surface_type=surface["surface_type"],
        plane=Plane(a=p["a"], b=p["b"], c=p["c"], d=p["d"]),
        polygon=[(v[0], v[1], v[2]) for v in surface["polygon"]],
        confidence=1.0, source=surface["source"],
    )
    return EntityArtifacts(
        schema_version=ENT_SCHEMA_VERSION, bundle_hash=f"synth_{case['id']}",
        scene_id=f"synth_{case['id']}", frame=_scene_frame(),
        representation_hash="rep", extractor_name="gate", extractor_version="0.0",
        entities=[entity], structural_surfaces=[sr], geometry_store_path=None,
        diagnostics=ExtractionDiagnostics(
            n_entities=1, n_structural_surfaces=1, runtime_seconds=0.0,
            coverage_score=None, notes="",
        ),
        notes={},
    )


def _gate_g5(artifacts) -> tuple[bool, dict]:
    fixture = json.loads(P4_FIXTURE_PATH.read_text(encoding="utf-8"))
    surfaces = fixture["synthetic_surfaces"]
    extractor = OnSurfaceExtractor()
    cfg = OnSurfaceConfig()
    failures: list[str] = []
    synthetic_checked = 0
    for case in fixture["cases"]:
        if not case["synthetic"]:
            continue
        synthetic_checked += 1
        arts = _synth_artifacts(case, surfaces[case["surface_ref"]])
        edges, _ = extractor.extract(arts, cfg)
        emitted = len(edges) == 1
        if emitted != bool(case["expected_on_surface"]):
            failures.append(f"{case['id']}: emitted={emitted} expected={case['expected_on_surface']}")

    # real F1
    f1 = next(c for c in fixture["cases"] if c["id"] == "F1")
    real_edges, _ = extractor.extract(artifacts, cfg)
    real_pairs = {(e.source.uid, e.target.uid) for e in real_edges}
    f1_ok = (f1["entity_uid"], f1["surface_uid"]) in real_pairs
    if not f1_ok:
        failures.append(f"F1 real {(f1['entity_uid'], f1['surface_uid'])} not emitted")

    return not failures, {
        "synthetic_cases_checked": synthetic_checked,
        "real_f1_present": f1_ok,
        "failures": failures,
    }


# --- G6 default path preserved (read-and-compare; no prior gate main) ----


def _gate_g6(artifacts) -> tuple[bool, dict]:
    detail: dict = {"method": "read committed P2/P3 reports + in-memory default build"}
    p2 = json.loads(PHASE2_REPORT.read_text(encoding="utf-8"))
    p3 = json.loads(PHASE3_REPORT.read_text(encoding="utf-8"))
    p2_pass = bool(p2.get("overall_blocking_pass"))
    p3_pass = bool(p3.get("overall_blocking_pass"))
    detail["phase2_report_overall_pass"] = p2_pass
    detail["phase3_report_overall_pass"] = p3_pass

    # In-memory default Phase 2 candidate build (directional + proximity-v2 +
    # surface). No artifact written. ON_SURFACE must be absent.
    bundle, _diag = build_graph(
        artifacts, _phase2_runs(), density_policy="phase2_telemetry_only",
    )
    default_on_surface = sum(1 for e in bundle.edges if e.type == "ON_SURFACE")
    detail["default_build_on_surface_edges"] = default_on_surface
    detail["note"] = (
        "P2/P3 pass is trusted from the committed reports, NOT re-derived "
        "(re-deriving would turn this gate into a hidden P2/P3 rerunner). "
        "ON_SURFACE isolation is proven by the in-memory default build."
    )
    ok = p2_pass and p3_pass and default_on_surface == 0
    return ok, detail


# --- G8 threshold-ordering guard -----------------------------------------


def _gate_g8() -> tuple[bool, dict]:
    raised = False
    try:
        OnSurfaceConfig(contact_threshold_m=0.10)
    except ValueError:
        raised = True
    return raised, {
        "bad_config": "OnSurfaceConfig(contact_threshold_m=0.10)",
        "raised_value_error": raised,
        "reason": "hypot(0.10, 0.0) > near_surface_threshold_m 0.05",
    }


# --- schema gate (inline, temp dir) --------------------------------------


def _gate_schema(artifacts) -> tuple[bool, dict]:
    bundle = _on_surface_bundle(artifacts)
    # ensure there is at least one ON_SURFACE edge to exercise the EdgeType
    has_on_surface = any(e.type == "ON_SURFACE" for e in bundle.edges)
    roundtrip_ok = False
    v2_rejected = False
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle"
        dump_scene_graph_bundle(bundle, out)
        loaded = load_scene_graph_bundle(out)
        roundtrip_ok = array_aware_equal(bundle, loaded) and any(
            e.type == "ON_SURFACE" for e in loaded.edges
        )
        # downgrade manifest to v2 and expect strict rejection
        manifest = out / "manifest.json"
        payload = json.loads(manifest.read_text())
        payload["schema_version"] = 2
        manifest.write_text(json.dumps(payload))
        try:
            load_scene_graph_bundle(out)
        except SchemaVersionError:
            v2_rejected = True
    ok = has_on_surface and roundtrip_ok and v2_rejected
    return ok, {
        "v3_on_surface_roundtrip_ok": roundtrip_ok,
        "v2_manifest_strict_rejected": v2_rejected,
        "bundle_had_on_surface_edge": has_on_surface,
    }


# --- G7 helpers: tracked eval JSON byte snapshot -------------------------


def _tracked_eval_json() -> list[Path]:
    """Git-tracked eval JSON artifacts, excluding this gate's own report.
    Tracked-only so local scratch files never fail the gate."""
    out = subprocess.run(
        ["git", "ls-files", "scenes/replica_room_0/eval/*.json"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    paths = []
    for line in out.stdout.splitlines():
        rel = line.strip()
        if not rel or rel.endswith("phase4_exit_gate_report.json"):
            continue
        paths.append(REPO_ROOT / rel)
    return sorted(paths)


def _snapshot(paths: list[Path]) -> dict[str, str]:
    snap = {}
    for p in paths:
        if p.exists():
            snap[str(p.relative_to(REPO_ROOT))] = hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
    return snap


def main() -> int:
    if not (REPLICA_V2_DIR / "scene_graph.json").exists():
        print("Refusing: enriched-v2 importer output is missing.")
        return 1
    if not PHASE2_REPORT.exists() or not PHASE3_REPORT.exists():
        print("Refusing: committed Phase 2/3 exit-gate reports are missing.")
        return 1

    tracked = _tracked_eval_json()
    snap_before = _snapshot(tracked)

    artifacts = _real_replica_artifacts()
    bundle = _on_surface_bundle(artifacts)

    g1_pass, g1_detail = _gate_g1(artifacts)
    g2_pass, g2_detail = _gate_g2(artifacts)
    g3_pass, g4_pass, g34_detail = _gate_g3_g4(bundle)
    g5_pass, g5_detail = _gate_g5(artifacts)
    g6_pass, g6_detail = _gate_g6(artifacts)
    g8_pass, g8_detail = _gate_g8()
    schema_pass, schema_detail = _gate_schema(artifacts)

    # G7: snapshot AFTER all gate computation (none of which writes a tracked
    # artifact). Compare.
    snap_after = _snapshot(tracked)
    changed = sorted(k for k in snap_before if snap_before.get(k) != snap_after.get(k))
    g7_pass = not changed

    gates = {
        "G1_rest_contact_determinism": (g1_pass, g1_detail),
        "G2_subset_of_polygon_near_surface": (g2_pass, g2_detail),
        "G3_clean_inverse_supports": (g3_pass, {"clean_inverse": g3_pass, **g34_detail}),
        "G4_no_materialized_supports": (g4_pass, {"materialized_supports": g34_detail["materialized_supports"]}),
        "G5_phase4_smoke_fixture": (g5_pass, g5_detail),
        "G6_default_path_preserved": (g6_pass, g6_detail),
        # G7 records ONLY the claim ("no tracked eval artifact changed while
        # the gate ran"), not the dynamic git ls-files universe and not its
        # count -- both are repo-state-dependent and would make every later-
        # phase eval JSON addition look like old-report drift (a cross-phase
        # stability bug). We keep snapshotting the dynamic set internally; we
        # just don't persist anything that churns when artifacts are added.
        "G7_prior_artifacts_untouched": (g7_pass, {
            "changed": changed,
            "all_unchanged": g7_pass,
        }),
        "G8_threshold_ordering_enforced": (g8_pass, g8_detail),
        "schema_v3_roundtrip_and_v2_rejection": (schema_pass, schema_detail),
    }
    overall = all(p for p, _ in gates.values())

    payload = {
        "phase": "P4.06",
        "artifact_kind": "phase4_exit_gate_report",
        "scene_id": artifacts.scene_id,
        "schema_version": 1,
        "overall_blocking_pass": overall,
        "extractor_version": ON_SURFACE_VERSION,
        "gates": {name: {"pass": p, **d} for name, (p, d) in gates.items()},
        "artifact_stability": {
            "tracked_eval_json_unchanged": g7_pass,
            "telemetry_untouched": (
                "scenes/replica_room_0/eval/phase4_on_surface_telemetry.json"
                not in changed
            ),
            "method": "byte sha256 snapshot before/after; verifier writes only its own report",
        },
        "summary": {
            "on_surface_edges": g34_detail["on_surface_edges"],
            "support_facts": g34_detail["support_facts"],
            "materialized_supports": g34_detail["materialized_supports"],
            "subset_violations": g2_detail["violation_count"],
        },
        "policy_decisions_recorded": [
            "Verifier only: no prior-phase gate/telemetry main() is invoked; "
            "Phase 2/3 pass is read from committed reports, not re-derived.",
            "ON_SURFACE remains isolated — absent from any default builder run "
            "(G6 in-memory default build has 0 ON_SURFACE edges).",
            "SUPPORTS is derived-only; zero materialized SUPPORTS edges (G4).",
            "Phase 4 ships floor support only; table/chair/wall remain deferred "
            "(not empty) pending EntitySurface / wall-attachment geometry.",
        ],
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )

    print(f"\nPhase 4 exit gate report -> {ARTIFACT_PATH.relative_to(REPO_ROOT)}")
    for name, (p, _d) in gates.items():
        print(f"  [{'PASS' if p else 'FAIL'}] {name}")
    print(f"\nOverall blocking: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())

"""Phase 6 plug-and-chug demo: does ON_ENTITY_SURFACE survive a different backend?

ISOLATED PROTOTYPE (not part of the P6 phase code, not a committed artifact path).
We have only one real capture (Replica room_0) and one real reconstruction backend
(oracle_replica). To answer "is it accurate if you plug in a *different* model?"
without a second real capture, this script SIMULATES alternate reconstruction
backends by perturbing the entity boxes the oracle produced -- the same boxes a
NeRF / 3DGS / detector backend would estimate with its own error -- and runs the
*identical* Phase 6 reasoner QA pipeline on each.

What it shows (honestly):
  - the support relation is INVARIANT to global reconstruction drift (a uniform
    height bias preserves relative gaps -> answers unchanged);
  - it degrades PREDICTABLY under per-object differential error: once an object's
    rest gap is pushed outside the frozen contact band (0.02 / 0.03 m), it drops
    off the table (recall loss) or a near-miss object floats onto it (precision
    loss). That band is the reconstruction-quality budget.

This is simulated-backend evidence, not a second real scene. It demonstrates the
plug seam (ReconstructionAdapter -> EntityArtifacts -> graph -> Router) and the
relation's sensitivity profile; it is not a generalization claim.

Run: python3 demo/backend_swap_demo.py
"""
from __future__ import annotations

import dataclasses
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.router_qa import score_questions
from graph.builder import build_graph
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer
from tools.phase6_router_qa_eval import (
    QUESTIONS_PATH,
    REPLICA_V2_DIR,
    _build_replica_artifacts,
    _eval_runs,
)

# Ground-truth table-support set (the 5 clean rests under the frozen band) and
# the band-excluded borderline, from the frozen Phase 6 fixture / closeout.
GT_TABLE = {"obj_92", "obj_90", "obj_12", "obj_59", "obj_87"}
BORDERLINE = {"obj_43"}  # pot, +0.0349 float -- one nudge from flipping in/out


@dataclasses.dataclass(frozen=True)
class Backend:
    """A simulated reconstruction backend = a box-perturbation profile."""
    name: str
    blurb: str
    global_dz: float = 0.0      # uniform vertical bias (m) -- same for every box
    jitter_sigma: float = 0.0   # per-object independent Gaussian noise (m), all axes
    scale: float = 1.0          # multiply box half-extents about centroid
    seed: int = 0


BACKENDS = [
    Backend("oracle", "ground-truth boxes (baseline)"),
    Backend("global_drift_+3cm", "uniform +3cm height bias (whole scene floats)", global_dz=0.03),
    Backend("jitter_1cm", "independent per-object noise, sigma=1cm", jitter_sigma=0.01, seed=1),
    Backend("jitter_2cm", "independent per-object noise, sigma=2cm", jitter_sigma=0.02, seed=2),
    Backend("jitter_3cm", "independent per-object noise, sigma=3cm", jitter_sigma=0.03, seed=3),
    Backend("shrink_5pct", "boxes 5% smaller (GS under-segmentation)", scale=0.95),
]


def _perturb_vec(v, dz, rng, sigma):
    jx = rng.gauss(0.0, sigma) if sigma else 0.0
    jy = rng.gauss(0.0, sigma) if sigma else 0.0
    jz = rng.gauss(0.0, sigma) if sigma else 0.0
    return (v[0] + jx, v[1] + jy, v[2] + dz + jz)


def apply_backend(artifacts, b: Backend):
    """Return a new EntityArtifacts with each entity box perturbed per profile b.
    Structural surfaces (room layout) are held fixed -- layout is typically the
    more stable prior; this isolates object-box error, which drives support."""
    rng = random.Random(b.seed)
    new_entities = []
    for e in artifacts.entities:
        (lo, hi), c = e.bbox_aabb, e.centroid
        # scale half-extents about the centroid first
        if b.scale != 1.0:
            lo = tuple(c[i] + (lo[i] - c[i]) * b.scale for i in range(3))
            hi = tuple(c[i] + (hi[i] - c[i]) * b.scale for i in range(3))
        # one shared jitter draw per object so the box stays coherent (corners +
        # centroid move together), plus the global bias
        jx = rng.gauss(0.0, b.jitter_sigma) if b.jitter_sigma else 0.0
        jy = rng.gauss(0.0, b.jitter_sigma) if b.jitter_sigma else 0.0
        jz = rng.gauss(0.0, b.jitter_sigma) if b.jitter_sigma else 0.0
        shift = (jx, jy, b.global_dz + jz)
        lo = tuple(lo[i] + shift[i] for i in range(3))
        hi = tuple(hi[i] + shift[i] for i in range(3))
        c = tuple(c[i] + shift[i] for i in range(3))
        new_entities.append(dataclasses.replace(e, bbox_aabb=(lo, hi), centroid=c))
    return dataclasses.replace(artifacts, entities=new_entities)


def run_qa(artifacts, questions):
    bundle, _ = build_graph(artifacts, _eval_runs(), density_policy="phase2_telemetry_only")
    router = Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                    verbalizer=StandardVerbalizer())
    ctx = ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))
    card = score_questions(questions, bundle, router, ctx)
    table_pairs = sorted([e.source.uid, e.target.uid]
                         for e in bundle.edges if e.type == "ON_ENTITY_SURFACE")
    return card, table_pairs


def main() -> int:
    if not (REPLICA_V2_DIR / "scene_graph.json").exists():
        print("Refusing: enriched-v2 importer output is missing.")
        return 1
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    base = _build_replica_artifacts()

    print("=" * 78)
    print("Phase 6 plug-and-chug: ON_ENTITY_SURFACE across simulated backends")
    print("  pipeline: perturbed boxes -> graph -> Router QA (frozen expected answers)")
    print("  contact band (reconstruction-quality budget): 0.02 float / 0.03 penetrate")
    print("=" * 78)
    header = f"{'backend':18} {'QA met':7} {'false':5} {'Q3 table rests (of 5)':24} {'extra/missing'}"
    print(header)
    print("-" * 78)

    report = []
    for b in BACKENDS:
        arts = base if b.name == "oracle" else apply_backend(base, b)
        card, ent_pairs = run_qa(arts, questions)
        agg = card["aggregate"]
        q3 = next(r for r in card["per_question"] if r["question_id"] == "Q3")
        cited = set(q3["cited_uids"])
        found = cited & GT_TABLE
        missing = GT_TABLE - cited
        extra = cited - GT_TABLE
        met = "YES" if agg["all_expected_outcomes_met"] else "no"
        notes = []
        if missing:
            notes.append(f"-{','.join(sorted(missing))}")
        if extra:
            notes.append(f"+{','.join(sorted(extra))}")
        print(f"{b.name:18} {met:7} {agg['false_answer_count']:<5} "
              f"{str(len(found)) + '/5  ' + ','.join(sorted(found)):24} "
              f"{'  '.join(notes) if notes else '(exact)'}")
        report.append({
            "backend": b.name, "blurb": b.blurb,
            "params": {"global_dz": b.global_dz, "jitter_sigma": b.jitter_sigma,
                       "scale": b.scale, "seed": b.seed},
            "qa_all_met": agg["all_expected_outcomes_met"],
            "qa_category_counts": agg["category_counts"],
            "false_answer_count": agg["false_answer_count"],
            "q3_table_found": sorted(found), "q3_missing": sorted(missing),
            "q3_extra": sorted(extra),
            "on_entity_surface_pairs": ent_pairs,
        })

    print("-" * 78)
    print("Note: each jitter row above is ONE sampled backend (its own seed), so a")
    print("single noisy draw can look non-monotonic. The real accuracy signal is the")
    print("average over many draws -> the curve below.\n")

    # --- Monte Carlo accuracy-vs-box-error curve (the headline accuracy result) ---
    K = 30
    sigmas = [0.0, 0.005, 0.010, 0.015, 0.020, 0.030]
    print("=" * 78)
    print(f"Accuracy vs per-object box error (mean over K={K} simulated backends/seed)")
    print("=" * 78)
    print(f"{'box sigma':10} {'mean rests/5':13} {'P(all 5)':9} {'mean false-on-table':19}")
    print("-" * 78)
    curve = []
    for sigma in sigmas:
        recovered, allfive, falsetable = [], 0, []
        for s in range(K):
            b = Backend(f"mc_{int(sigma*1000)}_{s}", "mc", jitter_sigma=sigma, seed=1000 + s)
            arts = base if sigma == 0.0 else apply_backend(base, b)
            card, _ = run_qa(arts, questions)
            q3 = next(r for r in card["per_question"] if r["question_id"] == "Q3")
            cited = set(q3["cited_uids"])
            n = len(cited & GT_TABLE)
            recovered.append(n)
            allfive += 1 if n == 5 else 0
            falsetable.append(len(cited - GT_TABLE - {"obj_39"}))
            if sigma == 0.0:  # deterministic; one draw suffices
                recovered = [n] * K
                falsetable = [falsetable[0]] * K
                allfive = K if n == 5 else 0
                break
        mr = sum(recovered) / len(recovered)
        mf = sum(falsetable) / len(falsetable)
        print(f"{sigma*100:>6.1f}cm   {mr:>5.2f}/5       {allfive/K:>6.0%}    {mf:>6.2f}")
        curve.append({"sigma_m": sigma, "mean_rests_recovered": mr,
                      "p_all_five": allfive / K, "mean_false_on_table": mf})

    print("-" * 78)
    print("Reading the result:")
    print("  - oracle & global_drift: support is INVARIANT to uniform reconstruction")
    print("    bias (relative gaps preserved) -- only per-object error matters.")
    print("  - the curve is the accuracy budget: ~<=1cm box error keeps most rests;")
    print("    past the 0.02/0.03 band, rests fall off (recall) and near-misses leak")
    print("    on (precision). That tells you the box accuracy a backend must hit.")
    out = Path(__file__).resolve().parent / "backend_swap_report.json"
    out.write_text(json.dumps({"named_backends": report, "accuracy_curve": curve,
                               "monte_carlo_seeds": K, "ground_truth_table": sorted(GT_TABLE)},
                              indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

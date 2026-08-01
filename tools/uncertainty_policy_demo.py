"""Run the uncertainty-preserving proposal policy on saved raw-mask bundles.

Example:
    python3 tools/uncertainty_policy_demo.py \
        notebooks/bundle_room_1 notebooks/bundle_room_2

The report compares three graph-construction policies over identical backend
evidence. It does not use Replica oracle labels and does not claim accuracy
improvement:

* hard: current accepted-score operating policy;
* inclusive: all raw proposals considered and resulting nodes trusted equally;
* uncertainty: accepted nodes by default plus a provisional activation pool.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmenter.base import load_segmentation_output
from segmenter.uncertainty_policy import (
    UncertaintyPolicyConfig,
    assignment_summary,
    build_policy_view,
)
from tools.c1_resolve_sweep import load_raw_masks


DEFAULT_OUT = REPO_ROOT / "runs" / "phase8_c1" / "uncertainty_policy_demo.json"


def _state_counts(view) -> dict[str, dict[str, int]]:
    return {
        state: {
            "n_raw_proposals": sum(p.state == state for p in view.proposals),
            "n_uncertainty_materialized_nodes": sum(
                p.state == state and p.materialized for p in view.proposals
            ),
            "n_inclusive_materialized_nodes": sum(
                p.state == state and p.inclusive_materialized
                for p in view.proposals
            ),
        }
        for state in ("accepted", "provisional", "discarded")
    }


def analyze_bundle(
    bundle_dir: Path,
    config: UncertaintyPolicyConfig,
    *,
    max_candidates: int = 12,
) -> dict:
    source = load_segmentation_output(bundle_dir)
    masks, scores = load_raw_masks(bundle_dir)
    if masks.shape[1] != source.n_vertices:
        raise ValueError(
            f"raw masks have {masks.shape[1]} vertices; "
            f"source sidecar has {source.n_vertices}"
        )
    view = build_policy_view(masks, scores, config)
    state_counts = _state_counts(view)

    candidates = sorted(
        (p for p in view.proposals if p.activation_candidate),
        key=lambda p: (-p.backend_score, p.proposal_id),
    )
    provisional_vertices = int(sum(
        p.assigned_vertices for p in view.proposals if p.activation_candidate
    ))
    uncertain_summary = assignment_summary(view.uncertainty_assignment)

    return {
        "bundle_dir": str(bundle_dir),
        "source": {
            "segmenter_name": source.segmenter_name,
            "segmenter_version": source.segmenter_version,
            "source_output_sha256": source.output_sha256,
            "n_vertices": source.n_vertices,
            "n_raw_proposals": int(len(scores)),
            "score_semantics": (
                "backend score retained in native threshold space; evidence "
                f"transform={config.score_transform}"
            ),
        },
        "policies": {
            "hard": {
                "description": "accepted proposals only",
                "min_score": config.accepted_score,
                **assignment_summary(view.hard_assignment),
            },
            "inclusive": {
                "description": (
                    "all raw proposals considered; resulting nodes trusted equally"
                ),
                "min_score": view.inclusive_score,
                **assignment_summary(view.inclusive_assignment),
            },
            "uncertainty": {
                "description": (
                    "accepted nodes active by default; provisional nodes retained "
                    "for explicit query-scoped activation"
                ),
                "min_score": config.provisional_score,
                **uncertain_summary,
                "n_default_nodes": state_counts[
                    "accepted"
                ]["n_uncertainty_materialized_nodes"],
                "n_activation_candidates": len(candidates),
                "n_raw_activation_candidates": sum(
                    proposal.raw_activation_candidate
                    for proposal in view.proposals
                ),
                "n_provisional_claimed_vertices": provisional_vertices,
                "provisional_claimed_vertex_fraction": (
                    provisional_vertices / source.n_vertices
                    if source.n_vertices else 0.0
                ),
            },
        },
        "proposal_states": state_counts,
        "activation_candidates": [
            {
                "proposal_id": p.proposal_id,
                "backend_score": p.backend_score,
                "existence_evidence": {
                    "positive": p.existence_evidence.positive,
                    "negative": p.existence_evidence.negative,
                    "observations": p.existence_evidence.observations,
                },
                "raw_vertices": p.raw_vertices,
                "assigned_vertices": p.assigned_vertices,
                "association_confidence": p.association_confidence,
            }
            for p in candidates[:max_candidates]
        ],
        "activation_candidates_truncated": len(candidates) > max_candidates,
    }


def build_report(
    bundle_dirs: list[Path],
    config: UncertaintyPolicyConfig,
    *,
    max_candidates: int = 12,
) -> dict:
    config.validate()
    if not bundle_dirs:
        raise ValueError("at least one raw-mask bundle is required")
    if max_candidates < 0:
        raise ValueError(f"max_candidates must be non-negative, got {max_candidates}")
    return {
        "schema": "uncertainty_policy_demo_v1",
        "hypothesis": (
            "retaining a bounded provisional proposal pool preserves recoverable "
            "evidence without treating it as equally trusted graph truth"
        ),
        "interpretation_limit": (
            "proposal counts and geometry coverage are diagnostics, not answer "
            "accuracy; no oracle labels or expected answers are used"
        ),
        "config": {
            "accepted_score": config.accepted_score,
            "provisional_score": config.provisional_score,
            "min_vertices": config.min_vertices,
            "score_transform": config.score_transform,
            "inclusive_score": "minimum observed score per bundle",
        },
        "bundles": [
            analyze_bundle(path, config, max_candidates=max_candidates)
            for path in bundle_dirs
        ],
    }


def _print_summary(report: dict) -> None:
    print(
        f"{'bundle':>24} {'hard':>6} {'incl':>6} {'prov':>6} "
        f"{'hard_cov':>9} {'expand_cov':>10}"
    )
    for bundle in report["bundles"]:
        hard = bundle["policies"]["hard"]
        inclusive = bundle["policies"]["inclusive"]
        uncertainty = bundle["policies"]["uncertainty"]
        print(
            f"{Path(bundle['bundle_dir']).name:>24} "
            f"{hard['n_materialized_nodes']:6d} "
            f"{inclusive['n_materialized_nodes']:6d} "
            f"{uncertainty['n_activation_candidates']:6d} "
            f"{hard['claimed_vertex_fraction']:9.3f} "
            f"{uncertainty['claimed_vertex_fraction']:10.3f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("bundle_dirs", type=Path, nargs="+")
    parser.add_argument("--accepted-score", type=float, default=0.2)
    parser.add_argument("--provisional-score", type=float, default=0.05)
    parser.add_argument("--min-vertices", type=int, default=20)
    parser.add_argument(
        "--score-transform",
        choices=("identity_probability", "sigmoid_logit"),
        default="identity_probability",
    )
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    config = UncertaintyPolicyConfig(
        accepted_score=args.accepted_score,
        provisional_score=args.provisional_score,
        min_vertices=args.min_vertices,
        score_transform=args.score_transform,
    )
    try:
        report = build_report(
            args.bundle_dirs,
            config,
            max_candidates=args.max_candidates,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"[uncertainty_policy_demo] HARD FAIL: {exc}")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_summary(report)
    try:
        out_label = args.out.relative_to(REPO_ROOT)
    except ValueError:
        out_label = args.out
    print(f"[uncertainty_policy_demo] report -> {out_label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

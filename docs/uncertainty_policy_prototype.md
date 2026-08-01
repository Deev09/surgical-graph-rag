# Uncertainty-preserving proposal policy prototype

Date: 2026-07-26.

## Hypothesis

Retaining a bounded provisional proposal pool can preserve recoverable
segmentation evidence without treating it as equally trusted scene-graph truth.
This borrows the uncertainty-preservation pattern from PUF and the
transient-evidence-versus-map-truth distinction from GEM-Occ. It does not
reproduce either system.

The current Mask3D reference remains MIN_SCORE=0.2. The 0.05 provisional floor
is a prototype policy chosen to inspect pool size, not a newly tuned operating
point and not a model improvement.

## Insertion point

```text
Mask3D / future NeRF or 3DGS proposal source
                 |
          raw masks + scores
                 |
     uncertainty_policy.build_policy_view
        /              |                \
  hard graph      provisional pool    inclusive control
  (unchanged)     (inactive default)   (all equal trust)
        \              /
         explicit query-scoped activation
                     |
          existing EntityArtifacts -> graph builder -> reasoner
```

The prototype stays upstream of `EntityArtifacts` and `SceneGraphBundle`.
Consequently it does not change schema version 4, relation semantics, the
Router, expected answers, or any committed v1/Phase 5 benchmark artifact.

## Evidence semantics

Each raw proposal carries two separate signals:

- `existence_evidence`: additive positive/negative mass derived from the raw
  backend score. The score is explicitly recorded as uncalibrated evidence,
  not a probability.
- `association_confidence`: the fraction of the proposal's raw vertices that
  survive highest-score overlap composition. This is a transparent heuristic
  for composition stability, not learned association calibration.

Thresholds stay in the backend's native score space. Mask3D uses
`identity_probability`; Segment3D emits logits and uses `sigmoid_logit` only
when converting a score into evidence mass. A threshold of 0.2 therefore
preserves each backend's frozen resolver policy, but is not a claim that their
scores are calibrated or directly comparable.

States are:

- `accepted`: score >= 0.2; active in the default graph.
- `provisional`: 0.05 <= score < 0.2; retained but activated only explicitly.
- `discarded`: score < 0.05; visible in the inclusive control, absent from the
  uncertainty graph.

The API rejects activation of discarded, unknown, or non-materialized proposal
ids.

## Real-bundle smoke result

Command:

```bash
./.venv/bin/python tools/uncertainty_policy_demo.py \
  notebooks/bundle_office_0 \
  notebooks/bundle_room_2 \
  notebooks/bundle_room_1 \
  notebooks/bundle_frl_apartment_0 \
  --accepted-score 0.2 \
  --provisional-score 0.05 \
  --min-vertices 20
```

| scene | hard nodes | inclusive nodes | provisional activation candidates | hard vertex coverage | fully expanded coverage |
|---|---:|---:|---:|---:|---:|
| office_0 | 23 | 48 | 1 | 0.520 | 0.524 |
| room_2 | 23 | 46 | 1 | 0.460 | 0.462 |
| room_1 | 26 | 45 | 2 | 0.526 | 0.527 |
| frl_apartment_0 | 155 | 173 | 18 | 0.469 | 0.479 |

The result supports only the narrow engineering claim: the provisional pool is
bounded on these bundles (1, 1, 2, and 18 materialized candidates) and preserves
more evidence than the hard graph without globally trusting the 19–25 extra
nodes seen in three inclusive controls—or the 18 provisional nodes in
frl_apartment_0, where every raw score is already above 0.05.

It does **not** establish better question answering. No oracle labels, reviewed
answer keys, relation extraction, or Router scoring are used by this report.
The generated diagnostic is
`runs/phase8_c1/uncertainty_policy_demo.json`.

## Follow-up experiment

The query-scoped follow-up is implemented in
`tools/query_scoped_expansion_demo.py` and documented in
`docs/query_scoped_expansion_prototype.md`. It tests raw proposals before
composition, including accepted masks swallowed by winner-take-all resolution.
On the saved Mask3D bundles, no raw activation candidate satisfies the frozen
tabletop predicate. Segment3D activates raw shelf-local candidates and changes
prediction identity, but does not recover a new reference answer.

This is a negative answer-quality result, not a model improvement. The hard
graph remains the baseline and relation thresholds and Router semantics remain
fixed.

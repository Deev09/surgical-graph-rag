# C1-M2C — oracle-free selection repair (predeclared protocol)

**Status: DRAFT — awaiting project-owner sign-off. No rule has been
designed or tuned. This document freezes the experiment BEFORE that
happens; it becomes PREDECLARED when the owner confirms and the sign-off
block below is filled.**

Written 2026-07-31, after the stage 0/0b ceiling measurements
(`docs/c1_composition_ceiling.md`) and before any selection-rule code.

## Hypothesis

An oracle-free rule over the saved Segment3D raw masks — using only
scores, mask sizes, mutual-overlap structure, and retained-region ratios —
can recover a substantial part of the measured joint selection ceiling
(entity 17→30/53, QA micro-P 0.53→0.93, micro-R 0.18→0.29 vs the human
room_2 key) without re-admitting the M2 precision collapse.

## Measured anchors (all vs the human room_2 key, frozen evaluator stack)

| row | ent@0.5 | micro-P | micro-R | support | floor | edges |
|---|---|---|---|---|---|---|
| A oracle boxes (representational ceiling) | — | 0.95 | 0.41 | 5/20 | 13/13 | 518 |
| Segment3D delivered @0.2 (baseline) | 17/53 | 0.53 | 0.18 | 2/20 | 7/13 | 645 |
| Segment3D joint ceiling (oracle-guided) | 30/53 | 0.93 | 0.29 | 5/20 | 9/13 | 203 |
| Mask3D delivered @0.2 (reference backend) | 18/53 | 1.00 | 0.24 | 2/20 | 10/13 | 127 |
| Mask3D joint ceiling | 19/53 | 1.00 | 0.24 | 2/20 | 10/13 | 108 |

Honest value framing: at the ANSWER level the ceiling gain over simply
using Mask3D is modest (micro-R 0.24→0.29, support 2→5, at −0.07
precision); at the ENTITY level it is large (18→30 of 53). A deployable
rule is only worth adopting if it beats the Mask3D reference, not merely
its own baseline — gate G6 encodes that.

## Scope and rule constraints (frozen)

- The rule is a DETERMINISTIC function of oracle-free signals ONLY:
  per-mask score and vertex count; pairwise overlap/containment structure
  among the raw masks; each mask's retained fraction under
  winner-takes-all at a declared priority. Nothing else.
- Explicitly forbidden inputs at rule-application time: oracle vertex
  membership, `mesh_semantic.ply`, `info_semantic.json`, answer keys,
  variant A/B artifacts or answers, class labels of any kind.
- Complexity cap: at most 4 free scalar parameters; no per-scene
  constants; no learned components. (Anti-overfit: room_2 has only 53
  entities.)
- The rule outputs a priority ordering plus an optional suppression set
  (masks removed entirely — stage 0b finding 4: demotion is required, not
  just promotion). Dense assignment is then produced by the UNCHANGED
  frozen resolver (`segmenter/mask_resolve.py`); resolver mechanics,
  min_vertices=20, and the evaluation stack are all frozen.
- Everything downstream (derived bundle, graph configs, Router, key
  scorer `tools/c1_joint_ceiling.py::score_against_key`) is frozen.

## Data, isolation, and what may be consulted

- Development scene: room_2, saved Segment3D bundle
  (`notebooks/s3d_bundle_room_2`, hashes in
  `docs/c1_artifact_manifest.json`). Zero GPU during development.
- During development the room_2 HUMAN key and oracle diagnostics
  (per-entity IoU, failure classes) MAY be consulted to score and debug
  iterations — that is what a development scene is for. B-relative
  metrics MUST NOT be optimization targets (B's box artifacts).
- Control backend: the same rule applied unchanged to the Mask3D room_2
  raw masks (`runs/phase8_c1/bundles_ms02/room_2`) — a rule that damages
  the reference backend is overfit to Segment3D quirks.
- Holdout: room_1, SEALED. No Segment3D inference on room_1 exists; the
  single new GPU run happens only after the rule is frozen. Until then
  nothing about room_1 masks may be inspected.

## Predeclared development gates (room_2; ALL must pass)

| gate | criterion | anchors (baseline → ceiling) |
|---|---|---|
| G1 | entity matches@0.5 ≥ 24/53 | 17 → 30 |
| G2 | QA micro-P vs human key ≥ 0.85 | 0.53 → 0.93 (M2 failed at ~0.5) |
| G3 | QA micro-R vs human key ≥ 0.25 | 0.18 → 0.29; must clear Mask3D's 0.24 |
| G4 | support-answer hits ≥ 4/20 | 2 → 5 |
| G5 | must_not violations ≤ 1 | delivered 1, ceiling 0 |
| G6 | control non-regression: rule on Mask3D room_2 gives ent@0.5 ≥ 18/53 AND micro-P ≥ 0.95 AND micro-R ≥ 0.24 | its delivered values |

## Iteration budget and stopping rule

At most **3 frozen rule versions** (v1, v2, v3). Each version's complete
results (all gates, full table row, per-question detail) are recorded
before the next version is designed. If none passes all six gates:
**STOP** — commit the negative result, spend nothing on the holdout.
No post-hoc gate adjustment; if a gate turns out to be wrong, that is
recorded as a protocol-design finding, not amended mid-run.

## Holdout protocol (room_1; only after all dev gates pass)

1. Freeze the passing rule version (committed, hash recorded here).
2. ONE Segment3D inference on room_1 `mesh.ply` via the pinned M2
   notebook (same commit/checkpoint/params as `docs/c1_m2_protocol.md`).
3. From that single bundle, compute BOTH: delivered (frozen 0.2 WTA — the
   holdout baseline, no extra GPU) and the frozen rule's output.
4. Evaluate ONCE against the human room_1 key. No iteration on room_1.

Holdout gates (ALL must pass):

| gate | criterion |
|---|---|
| H1 | QA micro-P vs human room_1 key ≥ 0.80 |
| H2 | QA micro-R > room_1 delivered |
| H3 | entity matches@0.5 > room_1 delivered |
| H4 | support-answer hits ≥ room_1 delivered |
| H5 | must_not violations ≤ room_1 delivered |

Pass → the rule is adopted as candidate composition `s3d_sr<v>` (the
frozen 0.2 WTA baseline is preserved for comparison; this is a
candidate-generation change, A/B-able, not a benchmark change). Fail →
negative result committed; the rule is NOT adopted; no further holdout
scenes are purchased to rescue it.

## Reporting requirements

Every version reports: the standard table row (format above),
per-question P/R, failure-class counts, graph edge count, and — for the
final verdict — the same table extended with the holdout rows. All
numbers via the frozen tools; no hand-computed metrics.

## What this experiment explicitly does NOT address

The 23 proposal-uncovered room_2 entities (stage 0) and the 15
representationally unreachable support answers (stage 0b finding 3 —
AABB/allowlist limits that cap even variant A at 5/20). Attachment
recall (0–1/14 for every row including A) is a downstream-semantics
finding, out of scope for any selection rule.

## Sign-off

- [ ] Project owner approves gates, budget, and holdout seal
      (date: __________, by: __________)
- [ ] Protocol frozen at commit: __________

# C1-P2 — oracle-free composition over the frozen proposal banks (DRAFT)

**Status: DRAFT — awaiting project-owner sign-off. No ceiling has been
measured, no rule designed, no parameter chosen. Zero GPU anywhere in
this protocol: every input already exists and is hash-pinned.**

Written 2026-08-02 after C1-P1 passed development and both transfers.
This is the experiment that is finally ALLOWED to claim a raw-PLY QA
improvement — if and only if its gates pass. Until then P1 proposals
remain evaluation-only and the frozen Mask3D @0.2 row remains the C1
reference.

## Decision this experiment answers

Can an oracle-free rule compose the frozen `P1 ∪ Mask3D` proposal banks
into a dense entity extraction that improves HUMAN-KEYED QA over the
frozen C1 reference — without re-admitting the precision collapse that
killed C1-M2 (0.52) and selection-repair v2 (0.35)?

## Why the staged shape below (lessons encoded)

1. **Entity recovery ≠ answer recovery.** Many newly viable P1 entities
   (blinds, vents) sit in attached-to-wall answers that even variant A
   cannot reach (1/14 on room_2 — a relation-semantics ceiling, not a
   perception one). Gates set on entity counts alone would reward
   recovering objects the Router can never cite. Therefore Stage P2.0
   FIRST measures the QA ceiling that composition could possibly reach,
   and all rule gates are set RELATIVE to that measured ceiling.
2. **Selection without oracle has failed before** (M2C: three versions,
   structural signals insufficient). P2's rule family gets the evidence
   M2C lacked — multi-view co-membership cuts and SAM's
   natural-image-calibrated mask scores — and a non-destruction control
   so the frozen reference cannot be damaged.
3. **No sealed scenes exist.** All three banks and keys pre-date this
   protocol. Stage P2.2 is prospective transfer under a frozen rule,
   and publications must say so.

## Frozen inputs (all hash-pinned already)

- P1 proposal banks + masks sidecars + view id buffers
  (`docs/c1_artifact_manifest.json` → `c1p1_proposal_banks`);
- the frozen ms02 Mask3D bundles (dense assignment + raw masks + scores);
- human keys for room_2 (dev), office_0 and room_1 (transfer);
- the unchanged downstream: exact evaluator, derived-bundle builder
  (oracle labels via correspondence — declared evaluation scaffolding,
  as in all C1 rows), battery graph configs, Router, MVP key scorer.

## Frozen QA anchors (uid micro metrics vs human keys)

| scene | C1 reference row (P / R / support hits) | entity matches@0.5 | P1 pooled viable |
|---|---|---|---|
| room_2 | 1.00 / 0.2449 / 2 | 18/53 | 33/53 |
| office_0 | null (0 cited) / 0.00 / 0 | 12/47 | 19/47 |
| room_1 | 0.5714 / 0.1143 / 0 | 17/45 | 26/45 |

## Signal allowlist for the composer (oracle-free by construction)

Allowed: P1 proposal vertex sets, their confidence-cut levels, sizes,
and per-proposal view-support statistics recomputed from the saved
masks + id buffers; SAM per-mask `predicted_iou`/`stability_score`;
Mask3D raw masks and native scores; the frozen ms02 dense assignment
(itself oracle-free); mutual overlap/containment structure; retained
fractions under a declared priority; raw-mesh adjacency and RGB.

Forbidden: `mesh_semantic.ply`, `info_semantic.json`, oracle ids or
labels, human keys, A/B artifacts or answers, C2 labels, per-scene
constants. Oracle data and keys are opened only by evaluators, only
after each composition artifact is finalized and hash-stamped.

## Stage P2.0 — oracle-guided QA ceiling (evaluation-only diagnostic)

Extend the existing joint-ceiling method (`tools/c1_joint_ceiling.py`
precedent) to the pooled bank on room_2: each oracle entity nominates
its best single proposal from `P1 ∪ Mask3D-raw` (IoU priority, collision
fallback); materialize through the frozen resolver; run the REAL derived
bundle → graph → Router; score against the human key. Report the ceiling
row (uid micro-P/R, support hits, per-question detail) next to the C1
reference and variant A.

**Predeclared proceed/stop rule:** rule development (P2.1) is justified
iff the room_2 ceiling shows
`micro-R ≥ 0.285` (C1 + 0.04) **or** `support hits ≥ 4` (C1: 2).
Otherwise **STOP**: the recovered entities do not translate into
reachable answers under the frozen relation semantics, the finding
redirects to representation/semantics work, and no composer is built.
The ceiling numbers themselves are recorded either way.

## Stage P2.1 — development rule (room_2)

One rule family, predeclared here in shape; versions differ only in
their ≤ **4 free scalar parameters**; at most **3 versions**, each fully
recorded before the next is designed; no sweeps.

**Rule family (fixed shape):** start from the frozen ms02 dense
assignment as the base; ADMIT P1 proposals by descending confidence cut
(ties: larger view-support, then size, then smallest digest), where an
admitted proposal may (a) claim unassigned vertices freely, and (b)
carve vertices out of an existing instance only if the proposal is
small relative to it and the donor retains a declared fraction of its
vertices; admitted proposals below the frozen `min_vertices=20` after
conflict resolution are dropped; a final retention pass may suppress
any instance reduced below a declared retained fraction. Free scalars
(v1 values fixed at sign-off, not tuned): minimum admission cut,
carve-out max size fraction, donor retention minimum, minimum SAM
score percentile for admitted proposals.

Deliverable: a real `SegmentationOutput` bundle per version (provenance
`c1p2_composer_v<n>`, oracle-free flag, input hashes) evaluated by the
UNCHANGED stack.

**Development gates (all must pass; anchors above; `ceiling` = the
Stage-P2.0 row):**

| gate | criterion |
|---|---|
| D1 | uid micro-P vs human key ≥ **0.90** (C1: 1.00; M2 collapsed to 0.52) |
| D2 | uid micro-R ≥ C1 + **0.5 × (ceiling-R − C1-R)** (capture at least half the measured reachable gain) |
| D3 | support hits ≥ C1's, and ≥ C1+1 if the ceiling shows ≥ C1+1 |
| D4 | entity matches@0.5 ≥ **24/53** (C1: 18; pool viable: 33) |
| D5 | must_not violations ≤ the C1 row's count |
| D6 | **non-destruction control:** the identical rule run with an EMPTY P1 bank reproduces the frozen C1 row's metrics exactly (the rule may only ever add) |

3 versions exhausted without a full pass → **STOP**, negative result,
transfers unspent, no gate adjusted.

## Stage P2.2 — frozen prospective transfer (office_0, room_1)

Only after all D-gates pass: compute each transfer scene's P2.0 ceiling
AND run the frozen rule once per scene, finalizing both composition
artifacts before opening either scene's oracle/key. No code or
parameter change between scenes. Gates per scene (all must pass on
both):

| gate | criterion |
|---|---|
| T1 | uid micro-P ≥ **0.85**, or (office_0 only, where C1 cites nothing) any citations at P ≥ 0.85 |
| T2 | uid micro-R ≥ C1_scene + 0.5 × (scene ceiling-R − C1_scene-R), when the scene ceiling gain is positive; otherwise no-regression |
| T3 | support hits ≥ the scene's C1 row |
| T4 | must_not violations ≤ the scene's C1 row |
| T5 | non-destruction control passes on that scene |

## Budget, stopping, and what a pass means

- Zero GPU. Max 3 dev versions + 1 frozen transfer run per scene.
- Dev stop or transfer fail → negative result; the frozen C1 row stays
  the reference; P1 banks remain evaluation-only.
- **Full pass** → the composer becomes candidate composition
  `c1p2_v<n>`: the FIRST permitted raw-PLY QA-improvement claim, stated
  scene-by-scene against the human keys, with the frozen C1 row
  preserved for comparison. MVP-v0/v1 gain a C1-P2 row only via a
  recorded spec amendment after the pass. C2-style learned labels on
  the new composition would be a separate, later experiment.

## 2026-08-02 Stage-P2.0 verdict — STOP: the ceiling is semantic, not compositional

One oracle-guided ceiling run (`tools/c1p2_ceiling.py`; report
`runs/phase8_c1p2/replica_room_2_p2_ceiling.json`; the recomputed C1
reference row hard-matched the frozen anchors). 33 proposals nominated
from the pooled bank (15 from P1), 31/53 entities materialized through
the frozen resolver at QA precision 1.00 — and the human-keyed QA
ceiling is:

| row | ent@0.5 | uid micro-P | uid micro-R | support | attached | floor |
|---|---|---|---|---|---|---|
| C1 reference | 18/53 | 1.00 | 0.2449 | 2/20 | 0/14 | 10/13 |
| **P2.0 ceiling** | **31/53** | 1.00 | **0.2653** | **3/20** | 0/14 | 10/13 |

Proceed rule required micro-R ≥ 0.285 or support ≥ 4; the ceiling gives
0.2653 and 3. **STOP — no composer is built**, per the approved rule.

**The finding, stated precisely:** of the 13 newly viable in-key
entities P1 recovered, exactly ONE became a citable answer (one
additional support hit). The other twelve — blinds, vents, wall-plugs
recovered at IoU 0.86–0.95 — sit in attached-to-wall answers that the
frozen 2 cm ATTACHED_TO semantics cannot cite even from perfect
geometry (variant A: 1/14). Perception is no longer the QA bottleneck
on room_2: with P1's proposals, composition COULD deliver 31/53
entities at precision 1.00, and it would barely move a single QA
number. **The bottleneck has moved — for the first time in this
project — from perception to relation semantics/representation.**

Consequences: P1 banks remain frozen and evaluation-only; the frozen C1
row remains the reference; Stages P2.1/P2.2 are cancelled with their
budgets unspent. Any successor experiment must target the relation
semantics/representation layer (wall-mount attachment semantics, seat
surfaces, allowlist) — which changes ANSWER definitions for every
variant including A, and therefore requires its own carefully
comparability-guarded protocol. Nothing is authorized by this verdict.

## Explicitly out of scope

Relation semantics, thresholds, allowlists, keys, graph/Router changes;
SAM re-inference or new views (P1b territory); frl_apartment_0; any
claim about attached-to-wall answers (semantics-blocked even for
variant A — recovering those entities is real but not citable);
sealed-holdout generalization language.

## Sign-off

- [x] Owner approves the staged shape, Stage-P2.0 proceed/stop numbers,
      the rule-family shape and 4-scalar budget, all D/T gates, and the
      3-version budget (2026-08-02, project owner / deevyaswain —
      "approved, run P2.0")

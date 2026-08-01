# C1-M2C stages 0 + 0b — composition ceilings (room_2, zero GPU)

Two measurements, both oracle-guided diagnostics: stage 0 (per-entity union
ceiling) and stage 0b (joint + task-level ceiling through the real
downstream, scored against the human key). Stage 0b closes the logical gap
stage 0 left open — per-entity optima could have conflicted in one dense
assignment. They do not.

# Stage 0 — per-entity composition ceiling

Date: 2026-07-31. Tool: `tools/c1_composition_ceiling.py`; reports in
`runs/phase8_c1/composition_ceiling/`. Development scene: room_2 only, per
the agreed roadmap. This is an ORACLE-GUIDED achievability bound over the
saved raw masks — a diagnostic, not a composer and not a deployable result.

## Question predeclared for this measurement

Before designing any composition experiment: is there enough signal in the
saved raw-mask fragments for fragment ASSEMBLY (unions of 2–3+ masks) to
beat mask SELECTION (picking the single best existing mask), and by how
much?

## Answer: no. Unions add ZERO recall at IoU 0.5 on both backends.

Entity recall ceilings over 53 room_2 oracle entities (frozen bundles;
Mask3D "delivered" = the frozen 0.2 bundle `runs/phase8_c1/bundles_ms02/`):

| ceiling | Mask3D @0.2 (171 masks) | Segment3D @0.2 (501 masks) |
|---|---|---|
| delivered (current composition) | 0.34 (18/53) | 0.32 (17/53) |
| best single mask (selection) | 0.38 (20/53) | **0.57 (30/53)** |
| greedy union ≤2 | 0.38 | 0.57 |
| greedy union ≤3 | 0.38 | 0.57 |
| greedy union unlimited (≤8) | 0.38 | 0.57 |

The single/union2/union3 rows agreeing exactly with the earlier orthogonal
viability counts (20/53 and 30/53, wasted 2 and 13 —
`docs/c1_m2_protocol.md`) is a consistency check, not new information; the
new information is the union rows: **no entity below 0.5 as a single mask
crosses 0.5 under any greedy union**. Unions do polish already-recovered
objects (chairs 0.91 → 0.96, a window 0.75 → 0.90) and nudge hopeless ones
(vent 0.18 → 0.30), but the best stuck entity reaches only 0.44.

## What this kills and what it leaves

**Killed (for the saved room_2 masks): fragment-assembly composition.** The
M2 verdict hypothesized two sub-problems — selection (13 winnable) and
construction (fragment assembly for 22 merges). The construction half is
now measured dead at this ceiling: the fragments to assemble the missing
objects do not exist in the saved masks. Caveats: greedy union is a lower
bound (exhaustive subsets and mask differences unprobed), but a 0.44 best
stuck IoU is not a near-miss pattern; and this is one scene (room_2 is the
development scene; the finding should be re-checked on the holdout when one
is spent).

**Left alive: selection repair, worth exactly 13 entities (0.32 → 0.57).**
The Segment3D winnable set is support-relevant small-to-mid objects, and
the deliveries are not near-misses — they are winner-takes-all losses of
nearly perfect masks: plate single-IoU **0.999** delivered **0.000**; three
chairs 0.84–0.91 delivered 0.00–0.50; vases 0.58–0.65 delivered ~0.00;
indoor-plants, bottle, blinds, window, vent. A higher-scoring overlapping
mask claims the vertices and the near-perfect mask is discarded whole.

**Out of reach of ANY composition of the saved masks: the remaining 23
entities** (4 lamps, 4 boxes, 4 wall-plugs, 3 blinds, 3 vents, 2 windows,
rug, sculpture, switch — predominantly small wall-mounted objects). These
need better proposals (e.g. query-scoped re-perception on the mesh), not
better composition. This matches the Mask3D closeout conclusion from the
other direction: proposal coverage is the hard ceiling.

# Stage 0b — JOINT + task-level ceiling vs the HUMAN key

Date: 2026-07-31. Tool: `tools/c1_joint_ceiling.py`; report in
`runs/phase8_c1/joint_ceiling/`. Method: each viable entity nominates its
best mask (collisions fall back to the next viable mask — none were
needed); the nominated set is materialized through the FROZEN resolver
(`segmenter/mask_resolve.py`), becomes a real `SegmentationOutput` bundle,
and runs the REAL downstream (exact evaluator → derived C1 bundle → graph
builder → Router), scored against the human-verified room_2 key. Variant A
and the delivered Segment3D composition are scored identically as
references.

| row | ent@0.5 | QA micro-P | QA micro-R | support hit | floor hit | attached hit | edges |
|---|---|---|---|---|---|---|---|
| A (oracle boxes) | — | 0.95 | 0.41 | 5/20 | 13/13 | 1/14 | 518 |
| delivered (frozen S3D @0.2) | 17/53 | 0.53 | 0.18 | 2/20 | 7/13 | 0/14 | 645 |
| **joint selected_only** | **30/53** | **0.93** | **0.29** | **5/20** | 9/13 | 0/14 | **203** |
| joint selected_plus_rest | 31/53 | 0.58 | 0.31 | 5/20 | 10/13 | 0/14 | 842 |

(QA micro-P/R over the exhaustive answer questions of
`eval/questions/phase8/replica_room_2_qa.json`; 49 expected members total.)

## Findings

1. **Joint compatibility: PROVEN.** All 30 per-entity winners coexist in
   one dense assignment — zero collision fallbacks, 30/53 delivered at
   IoU 0.5 (31 in plus_rest: suppressing the merges even frees one extra
   non-nominated mask to match). The stage-0 selection ceiling is fully
   jointly achievable under the frozen resolver mechanics.
2. **Root cause is score calibration, not mechanics: 23 of the 30 winning
   masks score BELOW the frozen 0.2.** Segment3D's confidence is
   anti-correlated with mask quality on these objects — big merged masks
   outscore near-perfect small masks. This also explains why the M2 sweep
   was flat: at any threshold, winner-takes-all lets the merges win. A
   deployable rule therefore cannot be a threshold; it must reorder
   priority from oracle-free evidence.
3. **Support answers reach VARIANT A's OWN LEVEL: 5/20 at precision 1.00.**
   A with perfect boxes also scores exactly 5/20 — the other 15 human
   support answers (mostly shelf contents) are lost to downstream
   representation (AABB/allowlist semantics), not to segmentation.
   Selection repair recovers 100% of the support answers that are
   representationally reachable. Floor: 7→9 of 13 (A: 13).
4. **Precision restores to near-A only if the non-nominated masks are
   SUPPRESSED**: selected_only P 0.93 (A: 0.95, zero must_not violations,
   203 edges); keeping the rest at frozen scores collapses P to 0.58 —
   the same failure mode as M2's 0.52. The deployable rule must demote
   bad masks, not merely promote good ones.
5. **Attachment is not a segmentation problem**: even variant A scores
   1/14 against the human key's attached-to ruling (windows/blinds are
   attached in reality; the 2 cm ATTACHED_TO semantics can't see it). No
   selection rule can or should chase this; it is a downstream-semantics
   finding.

## Verdict: GO for the deployable M2C selection-repair protocol

The jointly achievable human-QA gain is real and large relative to the
representational ceiling: micro-R 0.18 → 0.29 (A's ceiling: 0.41), micro-P
0.53 → 0.93, support answers 2 → 5 (= A), with a 3× smaller graph. The
predeclared protocol should target selection repair with ORACLE-FREE
signals only (mask score, mutual-overlap structure, mask size vs
claimed-region ratio, retained-fraction after resolution), room_2 as the
development scene, gates set against the human key BEFORE tuning, an
explicit precision guard (plus_rest shows how it collapses), and room_1
sealed as the single new-GPU holdout. B-relative metrics remain disallowed
for optimization. What no selection rule can reach: the 23
proposal-uncovered entities (stage 0) and the 15 representationally
unreachable support answers (finding 3) — those belong to different
experiments (re-perception; seat-surface/OBB representation).

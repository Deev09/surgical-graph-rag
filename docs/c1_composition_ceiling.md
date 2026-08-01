# C1-M2C stage 0 — composition ceiling measurement (room_2, zero GPU)

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

## Implication for the C1-M2C protocol (to be predeclared separately)

The experiment should be scoped as **selection repair**, not fragment
assembly: why does frozen winner-takes-all resolution discard masks with
IoU 0.9+, and what oracle-free selection rule keeps them without
re-admitting the precision collapse that failed the M2 gate (0.52 vs B)?
Candidate signals, all available without oracle access: mask score, mutual
overlap structure, mask size vs claimed-region size, raw-mask co-membership.
Per the agreed order, the protocol must be evaluated against the
human-verified keys (room_2's key exists as of 2026-07-31), with gates
predeclared before any rule is tuned, and one holdout scene as the only new
GPU spend. Optimizing selection against B-relative metrics remains
disallowed (it would learn B's box artifacts).

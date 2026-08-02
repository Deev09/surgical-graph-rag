# Semantics-v2 evaluation track (DRAFT — do not execute)

**Status: DRAFT — awaiting project-owner sign-off. Nothing here is
implemented, run, or scored. This document exists so the semantic
definitions are frozen BEFORE any number is computed, per the owner's
instruction after the C1-P2.0 verdict.**

Written 2026-08-02. Motivation: P2.0 measured that the QA bottleneck has
moved from perception to relation semantics — with P1's proposals,
composition could deliver 31/53 entities at precision 1.00 and citable
answers barely move, because the frozen 2 cm attachment band, the
AABB-top support test, and the support-class allowlist cannot express
what the human keys already record as true.

## Track separation (the non-negotiables)

1. **A separately labeled track: `semantics_v2`.** Every result it
   produces lives in new files (`runs/semantics_v2/…`), carries the
   track id in its schema, and is reported ONLY as a
   **benchmark-definition change** — never as an improvement over the
   frozen track. The two tracks are not comparable and no table may mix
   their rows without both labels.
2. **Every existing artifact is preserved unchanged**: all human keys
   (verbatim — they were written to physical reality and already
   contain the truth this track tries to reach), all frozen v1 rows
   (A/B/C1/C2, P1 banks, ceilings), all extractor defaults, gates, and
   the MVP outputs. Hash-guard tests must prove the v1 battery configs
   and frozen bundles are byte-identical after the v2 code lands; v2
   semantics are opt-in configs only.
3. **The keys are the target, not a variable.** The v2 track re-scores
   the SAME `human_verified` keys with new SYSTEM semantics. No key
   edit, no new questions, no reweighting. (This is what makes the
   track honest: the keys already say the blinds are attached and the
   cushions are on the sofa; v1 semantics simply cannot cite them.)

## Semantic definitions (frozen at sign-off, BEFORE any scoring)

All constants below are physical-reasoning choices declared now; they
may be challenged at sign-off and are frozen findings afterward. No
sweep, no post-hoc adjustment.

### D1 — wall-mounted attachment (`ATTACHED_TO` v2)

An entity is wall-mounted iff ALL of:
- nearest wall-plane distance of its bbox ≤ **0.12 m** (widened from
  2 cm to absorb the measured ≥5 cm annotation-plane displacement,
  Stage 0m finding, plus box-source error);
- NOT floor-supported (existing disqualifier, unchanged), with the
  existing furniture-rest limitation carried over unchanged;
- bbox bottom ≥ **0.30 m** above the calibrated floor (excludes
  floor-standing furniture leaning on walls; declared trade-off: sofas
  against walls stay non-attached, which matches the keys);
- bbox depth toward the wall ≤ **0.35 m** (thin objects: blinds, vents,
  plugs, switches, pictures, panels, clocks; excludes deep furniture).

### D2 — seat / interior support surfaces (`ON_ENTITY_SURFACE` v2)

v1's test (rest on the supporter's AABB TOP) is kept, and a second
disjunct is added — **contained rest**: entity E is on supporter S also
iff ALL of:
- E's XY footprint center lies inside S's XY footprint, and E's
  footprint area ≤ **0.5 ×** S's;
- E's bbox bottom lies within S's vertical extent extended by the
  existing contact band (`S.bottom ≤ E.bottom ≤ S.top + band`);
- E is not floor-supported and S is not E.
This makes cushions-on-sofa, blanket-on-sofa, plate-on-lower-tier, and
items-on-seats expressible from AABBs alone (variant A can reach them).
Declared, accepted imprecision: objects INSIDE furniture volumes (e.g.
drawer contents) also fire — recorded as a v2 semantics property, not a
bug, and visible in precision if it costs.

### D3 — furniture-anchor classes (support allowlist v2)

v1 allowlist + **cabinet, nightstand, bed** (the measured gaps: room_0's
cabinet question, room_1's nightstand questions, bed rest cases). The
battery question set is unchanged; the added classes make existing key
questions answerable, they do not add questions.

## Execution stages (each requires the prior one; NOTHING runs now)

- **S1 — implementation + guards.** v2 extractor configs (opt-in flags
  or v2 extractor classes), a `semantics_v2` battery config, synthetic
  tests for each definition, and hash-guard tests proving every v1
  path is byte-identical. No scene scoring yet.
- **S2 — variant A first: the new representation ceiling.** Run A under
  v2 semantics on all four keyed scenes; report per-relation and micro
  P/R next to (clearly labeled, never merged with) the frozen A rows.
  **Predeclared proceed rule to S3:** learned variants are justified
  iff, on room_2, A-v2 reaches **micro-R ≥ 0.55** (A-v1: 0.4082) with
  **micro-P ≥ 0.85**, AND no scene's A-v2 micro-P falls below **0.80**.
  Otherwise STOP: the definitions do not unlock meaningful reachable
  recall, the v2 track closes as a measured negative, and the v1 track
  remains the project's only benchmark.
- **S3 — learned variants under v2 (only on S2 pass).** B, the frozen
  C1 (ms02), and — evaluation-only — the P2.0 pooled-bank ceiling
  re-scored under v2 semantics. That last row answers the arc's open
  question: do P1's recovered entities become citable once the
  semantics can express them? (Expected but unproven; this measures
  it.) No new GPU, no new perception, no composer — compositions are
  the frozen artifacts only.
- **S4 — reporting.** A `runs/semantics_v2/` scorecard + a labeled
  section in the narrative. Every table carries: "semantics_v2 track —
  benchmark-definition change; not comparable to the frozen track."

## Budget and prohibitions

Zero GPU. No key edits, no v1-artifact modification, no threshold
sweeps (the D1/D2 constants are single declared values), no new
questions, no frl, no composer work, no C2/C3 reopening. Failures stop
at their stage and are committed as findings.

## Sign-off

- [ ] Owner approves the track separation, definitions D1–D3 (with
      their frozen constants), the A-first proceed rule (0.55 / 0.85 /
      0.80), and the stage order (date: ______, by: ______)

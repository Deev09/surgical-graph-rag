---
title: Phase 5 closeout — wall contact + first end-to-end system eval
status: closed
date: 2026-06-07
tags: [phase-5, closeout, contacts-surface, wall, system-eval, reasoner]
---

# Phase 5 closeout

> [!info] One-page interpretation freeze, written at close while the evidence is fresh.
> Plan: [[phase5_plan]]. Prior: [[phase4_closeout]] (ON_SURFACE + SUPPORTS + floor QA).

## What shipped

- **`CONTACTS_SURFACE(entity, surface)`** — wall contact. Normal-side contact with a wall-capable (roughly vertical) surface: the Phase 4 rest-contact predicate with `support_capable` swapped for `wall_capable`. Pure-geometry predicate (`geometry/wall_contact.py`), isolated stored extractor (`graph/relations/contacts_surface.py`, wall-only by policy), new `CONTACTS_SURFACE` EdgeType, graph serde bumped v3→v4.
- **"Against the wall?" QA** — turned from *deferred* (P4) into *answerable* through the normal Router path, via one generic SurfaceRef-anchored stored-edge helper that also serves "near the wall?" (NEAR_SURFACE). SUPPORTS stays its own derived-view path.
- **First end-to-end reasoner system eval** — a reasoner-native QA harness (`eval/router_qa.py`) that runs a mixed question set through the real Router and scores each answer against an explicit expected outcome. This is the phase's real deliverable.
- **Cross-phase exit-gate stability fix** — gate G7 reports now record the *claim* ("no tracked eval artifact changed"), not the dynamic file list, so adding a tracked eval artifact in a later phase no longer staled prior gate reports.
- **Deferred (explicit, not faked)** — `ATTACHED_TO` ("attached to the wall?") stays deferred; "on the table?" stays deferred. *Against* ≠ *attached*.

## What the numbers mean

On **Replica room_0**:

| metric | value |
|---|---|
| real wall contacts (frozen 0.02/0.02 band) | **1** — `obj_6` (lamp) against `wall_33_yplus` |
| mixed QA scorecard | **6/6 expected outcomes met** |
| false answers (fabricated facts) | **0** |
| category breakdown | 4 true_answer + 2 correct_defer |

The QA scorecard: "what is on the floor?", "against the wall?", "left of the blanket?", "near the wall?" all answer correctly; "on the table?" and "attached to the wall?" correctly defer. The integrated reasoner answers across **entity↔entity** (directional) and **entity↔surface** (proximity, contact, support) facts, and refuses what it cannot earn — end to end, zero fabricated facts.

## What they do NOT mean

- **Not a v1 benchmark improvement.** The QA eval is a *separate reasoner-native track* (scores Router answers/deferrals, not top-k retrieval); it is not comparable to the v1 benchmark, which is untouched.
- **Not wall *attachment*.** `CONTACTS_SURFACE` is geometric contact, not affixment. `ATTACHED_TO` is deferred; a box against a wall is in contact, not attached.
- **Not high wall coverage.** The frozen band is precision-over-recall (symmetric 0.02/0.02, no fit bias on walls); 1 real wall contact is the honest result. Looser thresholds admit the picture/sofa but also structural contamination (pillars, window). 1 honest contact > many noisy ones for a system test.
- **Not generalization evidence.** Single scene (Replica room_0). The scorecard proves the integrated path works here, not that thresholds transfer.
- **Not a self-describing deferral signal.** The harness detects deferral via compile metadata, because `Answer.outcome="abstain"` is overloaded (deferred / generic out_of_schema / execution_error). A future reasoner enhancement could add a `deferred` flag to `Answer`.

## Validation

- **P5.05 exit gate: 9/9 blocking gates pass** — wall-contact determinism (G1), subset ⊆ polygon-mode NEAR_SURFACE (G2), wall-contact smoke fixture incl. real W1 + WN negatives (G3), the mixed-QA scorecard re-derived in-memory and matched against the committed P5.04 artifact (G4), floor-QA regression (G5), default-path isolation with 0 ON_SURFACE and 0 CONTACTS_SURFACE in the default build (G6), prior artifacts untouched (G7), threshold guard (G8), and serde v4 round-trip + v3 rejection.
- **541/541 tests across 36 suites.**
- **Verifier discipline** — P5.05 writes only its own report; it re-derives the QA scorecard and *compares* to the committed artifact rather than regenerating it. No prior-phase gate/tool `main()` is invoked.
- **Prior artifacts untouched** — Phase 1/2/3/4 reports, telemetry, eval tables, and the P5.04 QA eval artifact byte-unchanged; `CONTACTS_SURFACE` wired into no default builder run.
- **Schema bump v3→v4** isolated: `bundle_hash` is version-independent, no committed artifact records the graph version, so all prior hashes/reports stayed byte-identical.

## Next

Phase 6 candidates, each honestly named and each needing its own geometry/data: `ATTACHED_TO` as a gated view over `CONTACTS_SURFACE` (elevated + wall contact), `HANGS_FROM` (ceiling), `EntitySurface` (tabletop/seat) to unlock "what's on the table/chair?", `IN`/`CONTAINED_BY`, and broader/multi-scene QA eval sets. None are started.

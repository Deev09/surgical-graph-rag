---
title: Phase 5 — wall contact + first end-to-end system eval
status: draft — awaiting fixture-freeze (P5.00) before any code work
date: 2026-06-06
tags: [phase-5, draft, contacts-surface, wall, system-eval, reasoner]
---

# Phase 5 — wall contact + first end-to-end system eval

> [!info] Status: DRAFT — design decisions folded in 2026-06-06
> First **integration / system-eval** phase. Adds ONE new relation family
> (wall contact) AND a reasoner-native QA eval that scores the whole path
> end-to-end: correct answers, correct deferrals, and wrong graph facts.
> **Gate**: no code lands until the Phase 5 fixtures (P5.00) are frozen and committed.

Related: [[phase4_plan]] / [[phase4_closeout]] (ON_SURFACE + SUPPORTS + floor QA), [[phase3_plan]] (polygon-clipped substrate), [[phase0_design]] (frozen contracts).

---

## What this phase accomplishes (and what it does not)

**Accomplishes.** Two things, deliberately coupled:

1. **Wall contact** — a new relation `CONTACTS_SURFACE(entity, surface)` for entities against a wall. Where `ON_SURFACE` is gravity-resting on an up-facing surface, `CONTACTS_SURFACE` is normal-side contact with a (roughly vertical) wall-like surface. Turns "what is against the wall?" from *deferred* into *answerable*.
2. **First end-to-end system eval** — a reasoner-native QA harness that runs a *mixed* question set through the real Router (compiler → executor → verbalizer) and scores each question against an explicit expected outcome. This is the phase's real purpose: proving the integrated system answers, defers, and refuses correctly across multiple relation families — not just that one more predicate passes unit tests.

**Does not accomplish.** No `ATTACHED_TO` (deferred — see D2). No ceiling contact (`HANGS_FROM`), no `EntitySurface` / table-chair support, no containment. No replacement of the legacy `benchmark/runner.py`. No claim on the v1 benchmark.

---

## Design decisions (gating)

### D1. Wall contact = CONTACTS_SURFACE primitive (normal-side, structurally a sibling of rest-contact)

`CONTACTS_SURFACE(entity, surface)` emits when an entity is in contact with a **wall-capable** surface on its interior (normal) side. The predicate is structurally the Phase 4 rest-contact predicate with the orientation gate flipped from gravity-side to normal-perpendicular-to-gravity. See [Wall-contact predicate](#wall-contact-predicate). It reuses the Phase 3/4 geometry primitives (`point_to_plane`, the 8-corner signed extents, `aabb_to_polygon_planar`) — no new geometry math, only a new orientation gate.

### D2. ATTACHED_TO is deferred; "against the wall?" answers from CONTACTS_SURFACE

`ATTACHED_TO` is a reserved legacy `EdgeType`, but P5 does **not** emit it and ships **no** derived `ATTACHED_TO` view. Reason: *against* ≠ *attached*. A box pushed against a wall is in contact but not affixed; true attachment (a picture, a mounted shelf) needs extra evidence (e.g., elevated off the floor AND in wall contact) that this phase does not model. "What is against the wall?" is answered directly from stored `CONTACTS_SURFACE` edges. `ATTACHED_TO` as a derived/gated view is a later-phase candidate.

### D3. CONTACTS_SURFACE is a stored relation (not a derived view)

Unlike `SUPPORTS` (P4.03 derived view), `CONTACTS_SURFACE` is a **materialized** edge family with its own extractor — the direct analog of `NEAR_SURFACE` / `ON_SURFACE`. "Against the wall?" executes over stored `CONTACTS_SURFACE` edges, not a projection.

### D4. Adding CONTACTS_SURFACE is a schema change (v3 → v4)

`CONTACTS_SURFACE` is absent from the `EdgeType` literal. P5.02 adds it and bumps `graph.serde.CURRENT_SCHEMA_VERSION` from 3 to 4. Strict v4 loader rejects v3 bundles (no migration); existing v3 graph bundles are not rewritten or silently coerced. Treated as an artifact-schema change, not a model improvement. (Same discipline as the P4.02 v2→v3 bump.)

### D5. Router-native QA eval is a NEW eval track, not a replacement for benchmark/runner.py

The legacy `benchmark/runner.py` scores a different shape of system (top-k retrieval outputs, regex/LLM parsers). Forcing Router `Answer`s into `RunnerOutput` would blur exactly what Phase 5 measures: does the *reasoner path* work end-to-end, including honest deferrals? So Phase 5 ships a **separate** reasoner-native eval (`eval/router_qa.py`), borrowing the good ideas from `benchmark/runner.py` (stable question schema, explicit expected outcome, failure attribution, deterministic JSON artifact, clear categories) without pretending old v1 metrics and new reasoner metrics are the same species. The legacy benchmark stays untouched and comparable.

**Expected outcomes** (declared per question): `answer`, `empty`, `defer`, `unknown`, `parser_failure`, `execution_error`.

**Score categories:**
- `true_answer` — expected `answer`, system answered, cited entities cover the expected set.
- `false_answer` — system answered when expected was `defer`/`empty` (a fabricated fact), OR answered with wrong entities.
- `miss` — expected `answer`, system answered but missed the expected entity (or returned empty/unknown).
- `correct_defer` — expected `defer`, system deferred (out_of_schema → abstain with a `deferred:` note).
- `true_empty` — expected `empty`, system returned empty.
- `unexpected` — outcome did not match the expected category and is not otherwise classified (carries the actual outcome for attribution).

### D6. Isolation preserved

`CONTACTS_SURFACE` ships extractor + tests but is **not** wired into any default `GraphBuilder` run (same as `ON_SURFACE` / `NEAR_SURFACE` isolation). The QA eval and exit gate build explicit graphs (NEAR_SURFACE + ON_SURFACE + CONTACTS_SURFACE) to exercise the mixed relational set; default Phase 2/3 paths are untouched.

### D7. Threshold calibration is measured before freezing (no assumed numbers)

As with the P4 floor fit-bias, wall planes may carry a fit bias. P5.00 fixture authoring **measures** real Replica wall-contact gaps (signed min-corner distances to wall planes for plausibly-against-wall entities) before choosing `contact_threshold_m` / `penetration_tolerance_m`, and records a pass-count table in the fixture header. No threshold is asserted without data.

---

## Wall-contact predicate

Let `up = normalize(-gravity)`. Let `sd(p) = a·p.x + b·p.y + c·p.z + d` (signed plane distance; positive on the interior-facing normal side). Plane normal `(a,b,c)` assumed unit and interior-facing (validated; same precondition as P4.01).

```
wall_capable        := abs(dot((a,b,c), up)) <= sin(max_wall_tilt_deg)   # normal ~perpendicular to gravity
on_interior_side    := sd(centroid) >= 0                                  # entity on the normal side of the wall
sd_min, sd_max      := min/max over 8 AABB corners of sd(corner)
wall_gap            := sd_min                                             # signed; >0 standing off, <0 penetrating
contact             := (wall_gap <= contact_threshold_m)
                       and (wall_gap >= -penetration_tolerance_m)
footprint_ok        := aabb_to_polygon_planar(aabb, plane, polygon) <= footprint_tolerance_m
CONTACTS_SURFACE    := wall_capable and on_interior_side and contact and footprint_ok
```

This is the rest-contact predicate with `support_capable` (gravity-side) replaced by `wall_capable` (normal-perpendicular-to-gravity). Everything else — interior-side check, signed min-corner gap with a penetration band, polygon-footprint clip — is identical in structure, so the implementation reuses the same geometry primitives and the same config shape.

**Thresholds (frozen at P5.00 from measured wall gaps — NOT assumed):** `contact_threshold_m = 0.02`, `penetration_tolerance_m = 0.02` (symmetric — walls show no systematic fit bias, unlike the floor, so no asymmetric penetration band is imported), `max_wall_tilt_deg = 30`, `footprint_tolerance_m = 0.0`, wall `near_surface_threshold_m = 0.30` (Phase 2 wall NEAR threshold) for the subset guard. The sharp symmetric band favors precision over recall: the only real wall-contact positive on Replica room_0 is `obj_6` (lamp) against `wall_33_yplus`. Full measured distribution, pass-count table, and the auditable negatives (picture `obj_63`, pillar `obj_84`, window `obj_17`) live in the frozen `phase5_wall_contact_smoke.json` header.

**Subset invariant (G analog of P4 G2):** `CONTACTS_SURFACE ⊆ polygon-mode NEAR_SURFACE` on the same wall surface, holding iff `hypot(contact_threshold_m, footprint_tolerance_m) ≤ near_surface_threshold_m` (wall NEAR threshold). Enforced as a config-validation guard.

---

## Reasoner wiring (mixed relational set)

The mixed eval forces the executor to handle three anchor/relation shapes through one Router, via exactly **two** surface-anchored paths (not one branch per relation):

- **SUPPORTS — special derived-view path (unchanged).** `SUPPORTS(SurfaceRef("floor"), ?x)` stays on its own branch because `SUPPORTS` is **not stored** (it is the P4.03 view over `ON_SURFACE`). Do not route it through the generic helper.
- **ONE generic stored-edge helper** for the shape `RELATION(?x, SurfaceRef(surface_type))` over any stored **entity → surface** relation. It resolves the `SurfaceRef` against `structural_surfaces` by type, finds stored edges of `RELATION` targeting those surfaces, and binds the entity sources. This single helper covers both:
  - `NEAR_SURFACE(?x, SurfaceRef("wall"))` — "what is near the wall?"
  - `CONTACTS_SURFACE(?x, SurfaceRef("wall"))` — "what is against the wall?"

  NOTE: the current executor resolves anchors only via `EntityRef` (graph nodes); a wall is a structural surface, so these queries do not work today. The generic helper is what fixes that — kept **tiny and relation-agnostic** (no per-relation "near wall" / "against wall" branches; the relation type is a parameter).
- **`LEFT_OF(?x, EntityRef("cabinet"))`** — existing `EntityRef` directional path, unchanged.

This keeps P5 coherent: it is not just wall contact, it is the first real mixed QA test over **entity↔entity** (directional) and **entity↔surface** (proximity, contact, support) graph facts. The generic helper is small and additive to `reasoner/executor.py` + `reasoner/compiler_rules.py`.

---

## Scope

**In:**

- `geometry/wall_contact.py`: `wall_contact(aabb, centroid, plane, polygon, gravity, config) -> WallContactResult` (pure-geometry leaf; reuses `surface_distance` primitives).
- `graph/relations/contacts_surface.py`: `ContactsSurfaceConfig` + `ContactsSurfaceExtractor` (isolated). New `CONTACTS_SURFACE` EdgeType; serde bump v3→v4.
- `reasoner/`: SurfaceRef-anchored stored-relation executor branch (NEAR_SURFACE + CONTACTS_SURFACE); "against the wall?" / "near the wall?" compiler patterns; verbalizer unchanged (reuses bindings/abstain).
- `eval/router_qa.py`: reasoner-native QA scorer (new track). `tools/phase5_router_qa_eval.py`: runs it on Replica → committed artifact. `eval/questions/phase5_mixed_qa.json`: frozen mixed question set.
- Two new fixtures (wall-contact geometry smoke + mixed QA set).
- Phase 5 integration exit gate.

**Out (deferred, named explicitly):**

- `ATTACHED_TO` (D2), ceiling `HANGS_FROM`, `EntitySurface` / table-chair support, containment.
- Any change to `benchmark/runner.py` or the v1 benchmark artifacts.
- Any change to `ON_SURFACE` / `NEAR_SURFACE` / `NEAR` / directional edge bytes.
- Wiring any surface-relation extractor into a default builder run.
- Promotion of polygon-clip to default (separate P3.06 decision).

---

## Tasks

> P5.00 is gating. P5.01..P5.05 do not begin until P5.00 is committed.

| ID    | Title                                                                       | Insertion point                                                                                       | Gates on | Status |
|-------|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|----------|--------|
| P5.00 | Freeze fixtures: wall-contact geometry smoke + mixed QA set (D7 measured)    | `eval/questions/phase5_wall_contact_smoke.json`, `eval/questions/phase5_mixed_qa.json`                | —        | pending |
| P5.01 | `wall_contact` predicate (pure geometry) + tests                            | `geometry/wall_contact.py`, `tests/geometry/test_wall_contact.py`                                      | P5.00    | pending |
| P5.02 | `ContactsSurfaceExtractor` + `CONTACTS_SURFACE` EdgeType + schema v3→v4      | `graph/relations/contacts_surface.py`, `graph/schema.py`, `graph/serde.py`, `tests/relations/test_contacts_surface.py`, serde round-trip | P5.01 | pending |
| P5.03 | Reasoner wiring: SurfaceRef-anchored stored relations + wall QA patterns     | `reasoner/compiler_rules.py`, `reasoner/executor.py`, `tests/reasoner/test_wall_contact_qa.py`         | P5.02    | pending |
| P5.04 | Router-native QA eval harness (new track) + scored artifact                 | `eval/router_qa.py`, `tools/phase5_router_qa_eval.py`, `scenes/replica_room_0/eval/phase5_router_qa_eval.json`, `tests/tools/test_phase5_router_qa_eval.py` | P5.03 | pending |
| P5.05 | Phase 5 integration exit gate                                               | `tools/phase5_exit_gate.py`, `scenes/replica_room_0/eval/phase5_exit_gate_report.json`, `tests/tools/test_phase5_exit_gate.py` | P5.04 | pending |

---

## Fixtures (P5.00)

### Wall-contact geometry smoke (`phase5_wall_contact_smoke.json`)

Single-clause-isolating cases (the P4 discipline), thresholds set from **measured** wall gaps (D7). Minimum:

- **W1** — real Replica entity against a wall (grounded by measurement; the F6-analog made real).
- Synthetic positive against a wall; synthetic positive with slight penetration.
- Negative: near-but-not-touching the wall (rejected by `contact`).
- Negative: outside the wall polygon footprint at contact depth (rejected by `footprint_ok`).
- Negative: wrong side (behind the wall / exterior) (rejected by `on_interior_side` + `contact`).
- Negative: floor surface fed in (rejected by `wall_capable` — proves the orientation gate excludes horizontals).
- Threshold-pinning case (passes at a looser tolerance, rejected at the chosen one).

Header records the measured wall-gap distribution + pass-count table (auditable, per D7).

### Mixed QA set (`phase5_mixed_qa.json`)

Stable question schema (borrowed from `benchmark/schema.py` ideas): `question_id`, `question`, `expected_outcome`, `expected_entities` (for `answer`), `notes`. Minimum cases:

| id | question | expected_outcome | proves |
|----|----------|------------------|--------|
| Q1 | what is on the floor? | answer | existing SUPPORTS/ON_SURFACE path still works |
| Q2 | what is against the wall? | answer — must-contain `obj_6` (lamp); must-not-contain `obj_63`/`obj_84`/`obj_17` | new wall-contact path answers; the picture/pillar/window must NOT appear (no false contact) |
| Q3 | what is on the table? | defer | deferral still correct, not faked |
| Q4 | what is left of the cabinet? | answer | existing directional path still works |
| Q5 | what is near the wall? | answer | SurfaceRef-anchored NEAR_SURFACE works |
| Q6 | negative wall-contact probe | empty / defer | no fabricated attachment (false_answer guard) |

Q2/Q6 expected entities are pinned from the measured wall-contact set so the eval can distinguish `true_answer` from `false_answer`.

**Honest-empty policy (general — let the data decide).** We do NOT massage Replica to force a wall positive. Had the frozen predicate found zero true contacts, Q2's `expected_outcome` would be `empty` — a valid system-test result (the phase tests whether the system tells the truth, not whether every relation yields a demo), recorded explicitly as:

```json
{ "expected_outcome": "empty",
  "notes": "No CONTACTS_SURFACE wall contacts under frozen P5 thresholds; this is an honest empty, not a failed demo." }
```

**Resolved at P5.00:** measurement found a real contact, so Q2's frozen `expected_outcome` is **`answer`** with `expected_must_contain = ["obj_6"]` (lamp) and `expected_must_not_contain = ["obj_63", "obj_84", "obj_17"]` (picture / pillar / window — no false contact). The honest-empty wording above stays as standing policy, not the active Q2 expectation.

---

## Phase 5 exit gate (P5.05)

Integration verifier (reads-and-compares; writes only its own report). Blocking gates:

| Gate | Check |
|------|-------|
| G1   | wall-contact determinism: two `ContactsSurfaceExtractor` runs → identical edge ids + keys. |
| G2   | subset: `CONTACTS_SURFACE` ⊆ polygon-mode `NEAR_SURFACE` on wall surfaces (0 violations). |
| G3   | wall-contact smoke fixture passes (all synthetic + real W1). |
| G4   | **mixed QA scorecard**: every question in `phase5_mixed_qa.json` hits its expected outcome — Q1 floor answer, Q2 wall answer/empty, Q3 defer, Q4 directional answer, Q5 near-wall answer, Q6 no false attachment. Zero `false_answer`. |
| G5   | floor QA regression: "what is on the floor?" still returns the P4 answer (obj_39 stool present). |
| G6   | default paths preserved: committed P2/P3/P4 exit-gate reports pass (trusted, not re-derived); in-memory default build has 0 ON_SURFACE and 0 CONTACTS_SURFACE edges. |
| G7   | prior artifacts untouched: git-tracked eval JSON (except this gate's report) byte-unchanged. |
| G8   | threshold-ordering guard enforced (`ContactsSurfaceConfig` raises on `hypot(contact, footprint) > near`). |
| schema | graph serde v4: `CONTACTS_SURFACE` round-trips; v3 manifest strict-rejected (inline, temp dir). |

---

## Validation / success criteria (defined before any code)

The v1 benchmark is saturated and out of scope. Success is:

1. **Wall-contact predicate correctness** on the smoke fixture (each negative rejected by its named clause).
2. **Mixed QA scorecard is all-green** on real Replica: floor answers, wall answers-or-defers-honestly, table defers, directional answers, near-wall answers, negative produces **no false attachment**. This is the headline result — the first evidence the integrated reasoner works across relation families.
3. **No false answers.** A fabricated attachment on the negative probe is a hard fail (it is the exact failure this phase exists to detect).
4. **Floor QA unchanged** (P4 regression intact).
5. **Default paths preserved**; CONTACTS_SURFACE isolated; prior artifacts byte-untouched.
6. **Honest framing.** The QA eval is a NEW reasoner-native track, reported as such; it is not compared to v1 benchmark numbers. Wall coverage is reported honestly (may be small on Replica room_0).

---

## Risks / confounders

- **Confounder — sparse wall contacts.** Replica room_0 may have few entities genuinely against walls; "against the wall?" could be legitimately empty. Mitigation: D7 measures this at P5.00; if empty, Q2's expected_outcome is `empty` and the eval documents it honestly (an honest empty is a pass, not a failure).
- **Risk — wall fit bias.** Like the floor, wall planes may sit off the entity faces; D7 measures before freezing thresholds.
- **Risk — "near the wall" was never actually wired.** The current EntityRef-anchored executor cannot resolve a surface anchor. P5.03 fixes this generally (SurfaceRef-anchored stored relations); the mixed eval is what surfaces whether it truly works.
- **Risk — false attachment.** The negative probe (Q6) is the guard; G4 fails on any `false_answer`.
- **Non-risk.** No change to `bbox_to_plane`, `ON_SURFACE`, `NEAR_SURFACE`, or the legacy benchmark. `CONTACTS_SURFACE` is a new, additive, isolated family.

---

## What this plan deliberately does not do

- Does not ship `ATTACHED_TO` (D2) or conflate "against" with "attached".
- Does not add ceiling / EntitySurface / containment relations.
- Does not wire any surface relation into a default builder.
- Does not modify or replace `benchmark/runner.py`; does not claim v1 benchmark movement.
- Does not rewrite existing v3 artifacts despite the v3→v4 bump.

---

## Phase 6 preview (not part of this plan)

- `ATTACHED_TO` as a gated view over `CONTACTS_SURFACE` (elevated + wall-contact).
- `HANGS_FROM` (ceiling contact).
- `EntitySurface` (tabletop / seat) → "what's on the table/chair?".
- Broader QA eval sets; possibly reconciling the reasoner-native track with a refreshed benchmark.

None are started in Phase 5.

---

## Closing note

Phase 5 is the first time we ask "does the whole thing work?" instead of "does this predicate pass?". Its win is a mixed-relation reasoner that answers, defers, and refuses correctly — with wall contact as the new capability and an honest scorecard as the evidence. Anything that looks like a v1 benchmark lift, or an `ATTACHED_TO` claim we did not earn, would be the kind of overreach this plan is written to avoid.

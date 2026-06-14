---
title: Phase 6 — EntitySurface (furniture-top) support
status: implemented — see phase6_closeout
date: 2026-06-14
tags: [phase-6, draft, entity-surface, on-entity-surface, support, reasoner]
---

# Phase 6 — EntitySurface (furniture-top) support

> [!info] Status: implemented — see [[phase6_closeout]].
> Adds **entity-on-entity support**: "what is on the table?" turns from a forced
> *defer* into a real answer. Reuses the existing rest-contact geometry; the new
> work is deriving a support **surface** from a furniture box and wiring an
> entity-class anchor through the reasoner.
> **Gate**: no code lands until the Phase 6 fixtures (P6.00) are frozen — written
> and reviewed. Commit happens only after diff review (no commit/push before review).

Related: [[phase5_plan]] / [[phase5_closeout]] (wall contact + first system eval), [[phase4_closeout]] (ON_SURFACE + SUPPORTS + floor QA), [[phase0_design]] (frozen contracts).

---

## Why this, and why not ATTACHED_TO

Phase 5's closeout listed `ATTACHED_TO` as a candidate. **Measured against Replica
room_0, `ATTACHED_TO` cannot earn a single honest positive** under the proposed
gate (wall contact ∧ elevated / not-floor-supported):

| object | role | wall contact? | bottom-gap above floor (z_floor≈0.043) | verdict |
|---|---|---|---|---|
| `obj_6` lamp_3 | only wall contact | yes | **−0.032 m** (bbox to/below floor) | floor-standing lamp, **not elevated**; also a 2.5 m bbox (noisy segment) |
| `obj_63` picture | canonical attached object | **no** (outside 0.02 band) | +0.922 m | genuinely attached, but never reaches the gate |
| `obj_84` pillar / `obj_17` window | structural | no | +0.017 / +0.892 | correctly excluded |

`{wall contact} ∩ {elevated}` is empty. The leaky proxy "not in the ON_SURFACE
floor set" would admit `obj_6`, but that is a false positive (the lamp is absent
from the floor set for a footprint/predicate reason, not because it is off the
floor). So `ATTACHED_TO` stays **deferred** in P6 — honestly, because the evidence
is not there. If we ever ship it, the only honest form is a smoke-only gated view
that returns ∅ on room_0; low payoff, deferred.

By contrast, **furniture-top support has dense real signal**: under the frozen band,
**5 table rests + 1 plant-stand rest** in room_0 (cabinet excluded as a container;
the pot is a documented band-excluded borderline). This phase ships that.

---

## What this phase accomplishes (and what it does not)

**Accomplishes.**
1. **`ON_ENTITY_SURFACE(supported_entity, supporter_entity)`** — a *stored*
   entity → entity relation: a portable object rests (gravity-contact) on the top
   face of a support-capable furniture entity. Same rest-contact predicate as floor
   `ON_SURFACE`; the **derived** entity-top surface is carried in evidence instead
   of stored as a structural surface ref.
2. **`SUPPORTS` extended (entity supporters)** — a derived view over
   `ON_ENTITY_SURFACE` so "what is on the table/chair?" is answerable; the
   supporter is the furniture **entity**, named explicitly (no UID parsing).
3. **Q3 `defer → answer`** in the reasoner-native QA set, plus a new
   honest-**empty** case ("on the chair?" → nothing rests there). This is a
   **Phase 6 eval-definition change**, not a v1-benchmark accuracy improvement.

**Does not accomplish.** No `ATTACHED_TO` (deferred — evidence absent). No
ceiling `HANGS_FROM`. No containment `IN`/`CONTAINED_BY`; cabinets are excluded
as supporters because they are the container boundary, and support-furniture
classes are excluded as supported tabletop answers in P6. No change to the
floor/wall paths, the v1 benchmark, or `benchmark/runner.py`.

---

## Generalization (project goal: pluggable reconstruction backends)

This phase must run on a new scene/backend without edits. Therefore:

- **No room_0 UIDs in logic.** room_0 appears only in frozen fixtures/evidence.
- The extractor operates on **generic entity boxes + a class label + provenance**.
  "Support-capable" is a **configurable class allowlist** (default
  `{table, desk, chair, stool, bench, shelf, sofa, plant-stand, counter}`),
  not a UID list. The same support-class set is excluded from the **supported**
  side in P6, so support furniture itself does not become a tabletop answer
  (for example plant-stand→table). Class is normalized from the entity label via
  a documented backend-agnostic hook (`label` → class family); a backend that
  supplies explicit classes can plug into the same field.
- **Owner provenance is explicit and versioned** (decision #5): a derived
  entity-top surface carries `owner_entity_uid` + `owner_class`. No owner-by-UID
  string parsing anywhere.

---

## Design

### D1 — Entity-top surfaces are DERIVED at build time, not stored in the manifest
A new pure leaf `extractors/entity_surfaces.py` derives, from each support-class
entity, a top-face support surface: plane `normal = +up`, `d` from the bbox top z;
polygon = the axis-aligned top-face rectangle. Returned as a **new `EntitySurface`
dataclass** (fields: `surface_uid`, `owner_entity_uid`, `owner_class`, `plane`,
`polygon`, `source="derived_entity_top"`, `confidence`). It deliberately does
**not** enter `structural_surfaces`.

*Why:* keeps `structural_surfaces` byte-identical, so floor/wall/NEAR_SURFACE
paths and their hashes are untouched (strong isolation, A/B-able). And because the
surface is derived, **no `ENT_SCHEMA_VERSION` bump** — the entity manifest is
unchanged input.

*Why AABB top face (not OBB):* every supporter in room_0 has tilt ≤ 0.3°. AABB is
exact here and far simpler; OBB is a later refinement if a tilted-furniture scene
demands it. Documented limitation, not silent.

### D2 — `ON_ENTITY_SURFACE` is a distinct stored **entity → entity** edge type (decision #2)
New `graph/relations/on_entity_surface.py` (`OnEntitySurfaceExtractor`) consumes
`(entities, derived_entity_surfaces)` and emits `ON_ENTITY_SURFACE` edges, reusing
`geometry/rest_contact.py` unchanged.

**Edge shape — entity → entity (not a surface ref).** The derived entity-top
surfaces are deliberately *not* in `SceneGraphBundle.structural_surfaces`, and
`graph/builder.py:_validate_edge_refs` (lines 252-264) raises on a
`GraphRef(kind="surface")` whose uid isn't a registered structural surface. So a
`surface`-kind target would fail validation (or force a leaky exception). Instead:
- `source = portable entity ref` (the supported object),
- `target = supporter/owner **entity** ref` (the furniture) — a real node, validates
  via the `entity` branch (line 252),
- `evidence` carries the derived-top provenance + measurements: `entity_surface_uid`,
  `owner_entity_uid`, `owner_class`, the top-face plane/polygon (enough top-face
  fields to reproduce the surface), the full rest-contact evidence, and the thresholds.
- **Invariant asserted at build:** `edge.target.uid == evidence["owner_entity_uid"]`.

This keeps the supporter explicit and parse-free, needs **no new `GraphRef` kind and
no entity-surface registry** (a larger schema change P6 does not need). New
`ON_ENTITY_SURFACE` `EdgeType` → graph serde **v4→v5** (entity manifest serde
unchanged). Same isolation discipline: not wired into any default builder run.

### D3 — `SUPPORTS` view extended for entity supporters
New `entity_support_facts(bundle)` in `graph/views/support.py` projects each
`ON_ENTITY_SURFACE` edge → `SupportFact(supporter = edge.target entity,
supported = edge.source entity)`. Because the edge is already entity → entity, the
projection reads endpoints directly — **no UID parsing, no evidence round-trip
needed** to name the supporter (the `owner_entity_uid` invariant guarantees they
agree). The existing P4 `support_facts` (floor) stays **byte-unchanged** (separate
function, no coupling).

### D4 — Entity-class anchor in the reasoner
New AST operand `EntityClassRef(entity_class: str)` (kept distinct from the
structural `SurfaceRef`, preserving the structural-vs-entity separation). Compiler:
flip the `_DEFERRED` entries for `table/desk/chair/seat/stool` (compiler_rules.py
lines 60-65) into a compiled support query anchored on `EntityClassRef`. Executor:
new branch resolving the class → owner entities → their entity-top edges →
bind supported entities (via `entity_support_facts`). The floor `SUPPORTS` path and
the wall stored-helper path are untouched.

### D5 — Contact band (CHOSEN / FROZEN for P6.00)
Reuse the floor rest band: **`contact_threshold_m = 0.02`, `penetration_tolerance_m
= 0.03`**, `max_tilt_deg = 30`, `footprint_tolerance_m = 0.0`. Measured table rests:

```
lamp_1→table_1  -0.013     book_4→table_4  -0.010     book_1→table_5  -0.008
book_2→table_1  -0.007     book_7→table_4  +0.007     pot→table_4     +0.035
```

This band captures **5 of 6** (books + lamp); the **pot (+0.035 float)** is
excluded — precision-over-recall, consistent with the wall picture exclusion.
No new threshold is introduced; this reuses the already-chosen floor band.

---

## Fixtures to freeze (P6.00)

1. **`eval/questions/phase6_entity_surface_smoke.json`** — geometry/extractor smoke.
   Real positives across **two supporter classes** for cross-class genericity:
   book_1/obj_92→table_5 (gap −0.0077), lamp_1/obj_87→table_1 (−0.0130), **and
   indoor-plant/obj_35→plant-stand/obj_55 (+0.0180)** — each with `owner_entity_uid`
   pinned; real negative pot/obj_43→table_4 (+0.0349, `contact` fail — the documented
   recall boundary); synthetic cases (exact rest; float +0.10 → no; beside footprint
   → no via clip; under-table / wrong support side → no; tilted top > max_tilt → no
   via support-capable; band-edge pin). All measured with the live `rest_contact`
   predicate at the frozen 0.02/0.03 band.
2. **`eval/questions/phase6_mixed_qa.json`** — extends the P5 set. Q1 floor / Q2 wall
   / Q4 left-of-blanket / Q5 near-wall = regression (answer). **Q3 "on the table?" →
   answer**, must_contain `[obj_92, obj_90, obj_12, obj_59, obj_87]`, must_not_contain
   a floor object (`obj_39`); pot `obj_43` listed in neither (documented band-excluded
   borderline). **Q6 "attached to the wall?" → defer (unchanged).** New **Q7 "on the
   chair?" → empty** (EntitySurface exists for chairs; nothing rests on them — honest
   answerable-empty, distinct from defer). Q3's flip is labeled a Phase 6
   eval-definition change in the fixture header.

P5 fixtures and artifacts stay **byte-frozen**; the P6 QA set is a *new* file.

---

## Eval & exit-gate plan

- **`tools/phase6_router_qa_eval.py`** — same harness as P5.04, new question set;
  candidate graph = directional + polygon NEAR + ON_SURFACE + CONTACTS_SURFACE +
  **ON_ENTITY_SURFACE**. Writes `scenes/replica_room_0/eval/phase6_router_qa_eval.json`.
- **`tools/phase6_exit_gate.py`** (verifier-only, mirrors P5.05): G1 entity-top
  derivation + rest determinism; G2 footprint-subset invariant; G3 smoke fixture
  (synthetic float/beside/under/tilted negatives + real positives + owner provenance);
  G4 re-derive QA scorecard in-memory and match the committed P6 artifact; G5 floor
  **and** wall QA regression (P4/P5 paths byte-unchanged); G6 default-path isolation
  (0 ON_SURFACE, 0 CONTACTS_SURFACE, **0 ON_ENTITY_SURFACE** in default build); G7
  prior artifacts untouched (claim-only form); G8 threshold guard; schema v5
  round-trip + v4 rejection.

---

## Risks / confounders

- **ON vs IN.** Cabinet rests excluded by class allowlist (container boundary →
  future `IN`/`CONTAINED_BY`). Documented, not silent.
- **Pot exclusion.** +3.5 cm float is a real recall gap; surfaced as a band
  limitation, not hidden.
- **Noisy boxes.** The lamp story shows segmentation noise is real; entity-top
  surfaces inherit it. The band is measured, not assumed.
- **Eval-definition change.** Q3 defer→answer makes the P6 scorecard
  non-comparable to P5's 6/6; stated explicitly, P5 artifact frozen.

---

## Step plan (per-step sign-off, no commit until reviewed)

- **P6.00** plan + freeze fixtures *(this doc + the two JSON files)*.
- **P6.01** `extractors/entity_surfaces.py` (pure derivation) + tests.
- **P6.02** `graph/relations/on_entity_surface.py` + `ON_ENTITY_SURFACE` EdgeType +
  serde v4→v5 + tests.
- **P6.03** `entity_support_facts` view + tests.
- **P6.04** reasoner: `EntityClassRef`, compiler flip, executor branch + tests.
- **P6.05** `tools/phase6_router_qa_eval.py` + artifact.
- **P6.06** `tools/phase6_exit_gate.py` + tests.
- **P6.07** closeout.

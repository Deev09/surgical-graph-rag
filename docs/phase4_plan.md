---
title: Phase 4 — ON_SURFACE rest-contact + derived SUPPORTS (Design B)
status: draft — awaiting fixture-freeze (P4.00) before any code work
date: 2026-06-06
tags: [phase-4, draft, relations, support, on-surface]
---

# Phase 4 — ON_SURFACE rest-contact + derived SUPPORTS

> [!info] Status: DRAFT — design decisions folded in 2026-06-06
> First *reasoning-substrate* phase built on the Phase 3 polygon-clipped geometry. Adds ONE materialized relation — `ON_SURFACE(entity, surface)` — defined as gravity-supported rest-contact, and ONE derived view — `SUPPORTS(surface, entity)` — that is never materialized.
> **Gate**: no code lands until the Phase 4 smoke fixture (P4.00) is frozen and committed.

Related: [[phase0_design]] (frozen contracts), [[phase2_summary]] (geometry substrate), [[phase3_plan]] (polygon-clipped NEAR_SURFACE — the substrate this phase stands on).

---

## What this phase accomplishes (and what it does not)

**Accomplishes.** Turns the Phase 3 polygon-clipped proximity substrate into a first *support* relation. `NEAR_SURFACE` answers "is this entity close to the finite surface?"; `ON_SURFACE` answers the stricter "is this entity *resting on* the finite surface, with gravity and contact evidence?" From `ON_SURFACE` we derive `SUPPORTS` as a read-side inverse, with zero new edges in the graph.

**Does not accomplish.** No learned backends. No wall-attachment, no ceiling-hangs-from, no "what's on the table/chair?". Those need either separate relations (`CONTACTS_SURFACE` / `ATTACHED_TO` / `HANGS_FROM`) or real furniture-top geometry (`EntitySurface`), and both are deferred (see Design Decisions D1, D2). Phase 4 ships the support primitive on the one surface type Replica room_0 can feed honestly: the floor.

This distinction matters for evaluation: P4's QA surface is intentionally narrow ("what is on the floor?"). That is a faithfulness phase, not a benchmark-accuracy phase.

---

## Design decisions (gating — folded in from review)

### D1. ON_SURFACE = gravity-supported rest-contact ONLY (Design B)

`ON_SURFACE` is **not** a generic finite-surface contact predicate. It applies only to *support-capable* (up-facing) surfaces. The geometric definition is the rest-contact predicate in [Rest-contact predicate](#rest-contact-predicate) below.

Rejected alternative (Design A): a generic `CONTACTS_SURFACE(entity, any-surface)` with `SUPPORTS` as a role-filtered inverse. Rejected because (a) the name "ON" over-claims for walls/ceilings — the same infinite-plane-style dishonesty Phase 3 removed; (b) the contact gate would have to be orientation-parameterized (gravity-side for horizontal, normal-side for vertical) inside one relation; (c) it builds fixture cases and contact logic for semantics (`ATTACHED_TO`, `HANGS_FROM`) that are deferred anyway. Design B keeps one clean predicate per relation and makes `SUPPORTS` a *clean unfiltered inverse*.

Consequence for later phases: wall-contact and ceiling-contact, when needed, become their **own** primitives (`CONTACTS_SURFACE` / `ATTACHED_TO` / `HANGS_FROM`), honestly named, not retrofitted into `ON_SURFACE`.

### D2. Defer EntitySurface entirely

Replica room_0's structural surfaces are floor / wall / ceiling from Habitat labels (7 polygoned surfaces). There is **no tabletop or seat geometry** in the data. Introducing `EntitySurface` as a first-class target in P4 would mean either (a) synthesizing tops from furniture AABBs — which is exactly the `SUPPORTS(table, cup)`-from-whole-AABB fake this plan forbids — or (b) a schema lift (new `GraphRef` target kind, importer top-fitting, serde, the C1/G7 surface-completeness gate, bundle_hash impact) that dwarfs "smallest useful change".

So P4 scopes `ON_SURFACE` to structural surfaces we already have clipped polygons for. "What's on the table / chair?" and "what's against the wall?" are **named non-goals**, deferred to a phase with real furniture-part geometry (different dataset, or an explicit provenance-tagged top-extraction step that stands on its own). Nothing provisional ships unless it carries an explicit `provisional` / `evidence_quality` tag — and P4 ships nothing provisional.

### D3. SUPPORTS is a reserved legacy EdgeType; P4 does not emit it

Correction (verified in code): `SUPPORTS` already exists as a reserved `EdgeType` (`graph/schema.py:21`), and a legacy `ON_TOP_OF → SUPPORTS` alias exists (`graph/relations/base.py:42`). P4 therefore does **not** claim `SUPPORTS` is new or absent from the schema. Instead:

- P4 emits **zero** `SUPPORTS` edges and adds **no** SUPPORTS-emitting extractor.
- The derived view does **not** reuse the `SUPPORTS` EdgeType — it returns a read-side projection structure (tuples / a small dataclass in `graph/views/support.py`), **not** graph `Edge` objects. Nothing of `type == "SUPPORTS"` ever enters `bundle.edges` via P4.
- Because `ON_SURFACE` (Design B) is already only support-capable rest-contacts, the inverse needs **no role filter**: `SUPPORTS_count == ON_SURFACE_count`.

The guarantee is a **count** invariant (G4: zero materialized `SUPPORTS` edges), not an absence-from-schema claim — `SUPPORTS` stays in the schema for legacy reasons. This mirrors how `FAR` is treated: a concept never stored (`graph/schema.py:25`).

### D6. penetration_tolerance_m = 0.03 absorbs a measured floor-fit bias (auditable)

Measured before freezing the fixture: on Replica room_0, the fitted `floor_25` plane sits ~high (plane at z ≈ 0.043, normal ≈ dead-up), so **every** floor-resting candidate has a *negative* `bottom_gap` (lowest AABB corner below the fitted plane). Across the 19 entities whose footprint clips the floor polygon with centroid above:

| stat | value |
|------|-------|
| min `bottom_gap` | −0.05689 (rug, obj_60) |
| max `bottom_gap` | −0.01451 (pillar, obj_46) |
| median `bottom_gap` | −0.02801 |
| all negative? | yes (19/19) |

Pass-count vs `penetration_tolerance_m` (contact band `[-tol, 0.02]`):

| tol | candidates contact-pass |
|-----|-------------------------|
| 0.01 | 0 / 19 |
| 0.02 | 4 / 19 (mostly structural columns/door) |
| **0.03** | **11 / 19 (clean furniture cluster: stool, basket, tables)** |
| 0.05 | 17 / 19 |
| 0.06 | 19 / 19 |

**Decision: `penetration_tolerance_m = 0.03`.** Rationale: P4's job is to prove the support predicate is *sharp*, not to maximize floor-membership recall on a biased plane. 0.03 captures the clean furniture cluster (including the basket) and keeps the deeper sofa (−0.043) / rug (−0.057) cases from silently defining the relation; 0.05 was rejected as too permissive for a first support-contact predicate.

> **Limitation (recorded):** `penetration_tolerance_m = 0.03` is Replica-calibrated to absorb observed floor-plane fit bias. It is **not** a physical claim that 3 cm penetration is acceptable. Future importer/floor-refit work should revisit this threshold (and could shrink it toward a physical ~1 cm if the floor plane is refit lower).

This is subset-safe (does not threaten G2): any resting object with negative `bottom_gap` straddles the plane → `bbox_to_plane = 0` → polygon-mode NEAR distance `hypot(0, 0) = 0 ≤ near_threshold`. Only `contact_threshold` (the hovering side) is constrained by D4a; `penetration_tolerance` is free w.r.t. the subset claim. The full pass-count table is also recorded in the fixture header (`threshold_calibration`) so the choice stays auditable against the data.

### D5. Adding ON_SURFACE is a schema change, not a model improvement

`ON_SURFACE` is absent from the `EdgeType` literal (`graph/schema.py:18-27`), and the schema header states that changing that list requires bumping `schema_version`. So P4.02 **bumps `graph.serde.CURRENT_SCHEMA_VERSION` from 2 to 3**, updates the serde round-trip tests, and records this as an **artifact schema change**, explicitly *not* a model/accuracy improvement. Existing Phase 1/2/3 artifacts are **not** rewritten — they remain at their authored `schema_version`. Existing v2 graph bundles are not rewritten; if loaded through strict v3 graph serde they require a rebuild or an explicit migration / backcompat path — **P4 does not silently coerce old graph artifacts**. The Phase 2/3 exit-gate artifacts and eval tables stay byte-frozen.

### D4. Three corrections (from review)

- **D4a — threshold ordering.** `ON_SURFACE` uses its own *tighter* `contact_threshold_m`, distinct from `NEAR_SURFACE`'s proximity threshold ("5 cm near the floor" ≠ "on the floor"). The subset invariant `ON_SURFACE ⊆ polygon-mode NEAR_SURFACE` holds **iff `hypot(contact_threshold_m, footprint_tolerance_m) ≤ near_surface_threshold_m`**. With the P4 default `footprint_tolerance_m = 0.0` this reduces to `contact_threshold_m ≤ near_surface_threshold_m`. The full `hypot` form is the real guard: the polygon-mode NEAR distance combines the normal gap and the in-plane gap via `hypot`, so if `footprint_tolerance_m` were ever relaxed above 0, a resting entity could sit inside the contact band on the normal axis *and* inside the footprint band in-plane yet still exceed `near_surface_threshold_m` in combination. This ordering is asserted as a config-validation that raises on violation (not assumed) — see G8.
- **D4b — signed plane evidence, not bare `bbox_to_plane`.** `bbox_to_plane` returns 0 for *any* straddle and so cannot tell "resting on" from "embedded in" or "poking through from below." Contact is defined on the *signed* min-corner distance (`bottom_gap = sd_min`) with an explicit penetration band — see the predicate.
- **D4c — split ceiling negatives by declared clause.** Under Design B a ceiling is not support-capable, so both ceiling cases are `ON_SURFACE` *negatives*. They differ in *which clauses* reject them, which is what the fixture asserts via `expected_failed_clauses`: F7 is flush/in-contact and rejected by `support_capable` **alone** (the role gate — `contact` actually passes, so this isolates role from proximity); F8 is below the ceiling but not touching and is rejected by `support_capable` **and** `contact` together (a multi-clause sanity case, not a clean single-gate test). Declaring the exact clause set per case is what keeps a failure localizable.

---

## Rest-contact predicate

Let `up = normalize(-frame.gravity)` (Replica frame: `gravity = (0,0,-1)` → `up = (0,0,1)`). Let `sd(p) = a·p.x + b·p.y + c·p.z + d` be the signed plane distance (`geometry.surface_distance.point_to_plane`; positive on the normal side).

**Precondition (validated, D4/normal-orientation):** the surface normal `(a,b,c)` is interior/up-facing — the same convention Phase 3 fixtures use. `support_capable` and `centroid_on_support_side` both rely on it; if a surface stored a flipped normal they would invert together and silently lie. P4.01 validates `‖(a,b,c)‖ ≈ 1` and (for support-capable surfaces) that the orientation is consistent; a violation raises rather than emitting a wrong edge.

```
support_capable        := dot((a,b,c), up) >= cos(max_tilt_deg)          # up-facing surface
centroid_on_support_side := sd(centroid) >= 0                            # entity sits on the normal side
sd_min, sd_max         := min/max over the 8 AABB corners of sd(corner)  # reuses bbox_to_plane internals
bottom_gap             := sd_min                                          # signed; >0 hovering, <0 penetrating
contact                := (bottom_gap <= contact_threshold_m)
                          and (bottom_gap >= -penetration_tolerance_m)
footprint_ok           := aabb_to_polygon_planar(aabb, plane, polygon) <= footprint_tolerance_m   # P3.01 reuse
ON_SURFACE             := support_capable and centroid_on_support_side and contact and footprint_ok
```

Defaults (provisional, Replica-calibrated — NOT generalization evidence; the fixture pins behavior):
- `contact_threshold_m = 0.02` (≤ `near_surface_threshold_m = 0.05`, satisfying D4a).
- `penetration_tolerance_m = 0.03` (see D6 — Replica-calibrated to absorb floor-plane fit bias; chosen over 0.05 to keep the predicate sharp).
- `max_tilt_deg = 30` (`support_capable` uses `cos(30°)`).
- `footprint_tolerance_m = 0.0` (strict overlap; the polygon clip returns 0 only when the projected AABB actually overlaps/clips the surface polygon).

**Why this is sound (sketch, to be proven by tests, not asserted):**
- *Hovering* entity (bottom_gap > `contact_threshold`) → `contact` false → rejected, even though `centroid_on_support_side` is true. Footprint and side alone do not make "on."
- *Embedded / wrong-side* entity (all corners below, `sd_min ≪ 0`) → `bottom_gap < -penetration_tolerance` → rejected; and `centroid_on_support_side` also false.
- *Outside-footprint* entity at resting height (in-plane gap > 0) → `footprint_ok` false → rejected. This is the Phase-3 plane-only-false-positive case promoted to `ON_SURFACE`.
- *Ceiling / wall* → `support_capable` false → rejected regardless of proximity.

**Subset invariant (provable given D4a):** for any `ON_SURFACE` edge, `footprint_ok` ⇒ in-plane gap `≤ footprint_tolerance_m`, and `contact` ⇒ `bottom_gap ≤ contact_threshold_m` (so the non-negative normal gap `bbox_to_plane ≤ contact_threshold_m`, whether the entity hovers — `bbox_to_plane = bottom_gap` — or straddles — `bbox_to_plane = 0`). The polygon-mode `NEAR_SURFACE` distance is `hypot(bbox_to_plane, in_plane_gap) ≤ hypot(contact_threshold_m, footprint_tolerance_m) ≤ near_surface_threshold_m` by D4a. Hence `ON_SURFACE ⊆ polygon-mode NEAR_SURFACE` under the same surface and the threshold ordering. With `footprint_tolerance_m = 0` the in-plane term vanishes. G2 tests this on real data; G8 enforces the ordering precondition.

---

## SUPPORTS derived view (D3)

```
supports_view(bundle) -> list[(surface_uid, entity_uid, evidence_ref)]
    = [ (e.target.uid, e.source.uid, e.edge_id)
        for e in bundle.edges if e.type == "ON_SURFACE" ]
```

- No new edges. No `EdgeType` named `SUPPORTS`.
- `SUPPORTS_count == ON_SURFACE_count` exactly (clean inverse — Design B needs no role filter).
- The view carries the originating `ON_SURFACE` `edge_id` so evidence is referenced, not copied (no double-count of evidence).

---

## Scope

**In:**

- `geometry/surface_distance.py` (or a new `geometry/rest_contact.py`): pure `rest_contact(aabb, plane, polygon, up, config) -> (bool, dict)` implementing the predicate above; reuses `point_to_plane`, the 8-corner signed extents, and `aabb_to_polygon_planar`. No `graph/` or `extractors/` imports (same leaf discipline as `bbox_to_surface`).
- New `EdgeType` literal `ON_SURFACE` (schema change → `CURRENT_SCHEMA_VERSION` 2→3, D5); new isolated `OnSurfaceExtractor` + `OnSurfaceConfig`. **Not wired into any default GraphBuilder run** — ships with tests only, exactly as `SurfaceProximityExtractor` did at P2.09. This is how "default paths preserved" is guaranteed: no existing edge family changes bytes, and the schema bump does not rewrite existing artifacts.
- `graph/views/support.py`: the `SUPPORTS` derived view (D3).
- QA template: "what is on the floor?" answered from `ON_SURFACE(entity, floor)` edges.
- New Phase 4 smoke fixture, separate from Phase 2/3 fixtures.
- ON_SURFACE coverage telemetry on Replica room_0 (honest "floor only" report).
- Phase 4 exit gate with the new invariants.

**Out (deferred, named explicitly):**

- `EntitySurface` / furniture-top geometry (D2). "What's on the table/chair?" is a non-goal.
- Wall-contact (`CONTACTS_SURFACE` / `ATTACHED_TO`) and ceiling-contact (`HANGS_FROM`) (D1).
- Any learned backend or noisy-input robustness evaluation.
- Promotion of polygon-clip to the default `SurfaceProximityConfig` (still gated on the P3.06 conditions, unrelated to P4).
- Any change to Phase 2/3 smoke fixtures — they stay frozen as regression.
- Any change to `NEAR_SURFACE`, `NEAR`, or directional edge bytes.

---

## Tasks

> P4.00 is gating. P4.01..P4.06 do not begin until P4.00 is committed.

| ID    | Title                                                                  | Insertion point                                                                 | Gates on | Status |
|-------|------------------------------------------------------------------------|---------------------------------------------------------------------------------|----------|--------|
| P4.00 | Freeze ON_SURFACE smoke fixture (D4c, declared-clause smoke cases)      | `eval/questions/phase4_on_surface_smoke.json`                                    | —        | pending |
| P4.01 | `rest_contact` pure-geometry helper + normal-orientation validation + tests | `geometry/rest_contact.py`, `tests/geometry/test_rest_contact.py`            | P4.00    | pending |
| P4.02 | `OnSurfaceExtractor` + `OnSurfaceConfig` (isolated) + `ON_SURFACE` EdgeType + schema bump v2→v3 + threshold-ordering validation + tests | `graph/relations/on_surface.py`, `tests/relations/test_on_surface.py`, `graph/schema.py` (add `ON_SURFACE` EdgeType), `graph/serde.py` (`CURRENT_SCHEMA_VERSION` 2→3), serde round-trip tests | P4.01 | pending |
| P4.03 | `SUPPORTS` derived view + no-double-count + subset invariant tests      | `graph/views/support.py`, `tests/graph/test_support_view.py`                     | P4.02    | pending |
| P4.04 | QA template "what is on the floor?" + explicit deferral of table/chair/wall | `reasoner/` (template), `tests/reasoner/test_on_floor_query.py`             | P4.03    | pending |
| P4.05 | ON_SURFACE coverage telemetry (honest "floor only")                    | `tools/phase4_on_surface_telemetry.py`, `scenes/replica_room_0/eval/phase4_on_surface_telemetry.json` | P4.03 | pending |
| P4.06 | Phase 4 exit gate (invariants + default-preservation)                  | `tools/phase4_exit_gate.py`, `scenes/replica_room_0/eval/phase4_exit_gate_report.json` | P4.05 | pending |

---

## Phase 4 smoke fixture (P4.00 composition)

Each case declares `expected_failed_clauses` — the clause(s) that reject it — so a failure localizes the bug (the P3 S-case discipline). Most negatives isolate a **single** clause; the ceiling cases legitimately trip more than one under Design B (a ceiling is neither support-capable nor, when far, in contact) and are labeled multi-clause rather than pretended to be single-clause. Minimum cases:

- **F1 — floor positive (Replica-grounded).** A real entity resting on `floor_*`. `ON_SURFACE` yes; `SUPPORTS` derives.
- **F2 — floor positive, slight penetration.** `bottom_gap ∈ [-penetration_tolerance, 0)`. Yes. Pins `penetration_tolerance_m`.
- **F3 — floating above floor.** Footprint ok, centroid on support side, `bottom_gap > contact_threshold`. **No** — `expected_failed_clauses: ["contact"]`. The clean single-clause gap rejection, and a NEAR_SURFACE-yes / ON_SURFACE-no case (satisfies "≥1 NEAR-but-not-ON").
- **F4 — outside footprint at resting height.** `bottom_gap` small but in-plane gap > 0. **No** — `expected_failed_clauses: ["footprint_ok"]`. Single-clause footprint rejection; also NEAR-near / ON-no depending on threshold.
- **F5 — wrong side (below floor).** Centroid below, `sd_min ≪ 0`. **No** — `expected_failed_clauses: ["centroid_on_support_side", "contact"]` (below the floor fails both; labeled multi-clause).
- **F6 — wall contact, role-rejected.** Entity flush against a wall (in contact, centroid on interior side). **No** — `expected_failed_clauses: ["support_capable"]`. Single-clause: the up-facing gate excludes verticals even when contact/side would otherwise pass.
- **F7 — ceiling flush, role-rejected (D4c).** Entity in contact with the ceiling (so `contact` would pass). **No** — `expected_failed_clauses: ["support_capable"]`. Single-clause role rejection — isolates the up-facing gate from the contact gate.
- **F8 — below ceiling but not touching, gap-rejected sanity (D4c).** Entity well below the ceiling (bottom_gap 0.30, far beyond both contact and near thresholds). **No** — `expected_failed_clauses: ["support_capable", "contact"]` (fails the role gate *and* the contact gate together). Explicitly multi-clause under Design B; F3 already covers the pure single-clause gap rejection, so F8 is a ceiling sanity case, not a clean gap test.
- **F9 — deep penetration, threshold-pinning (D6).** Entity straddling the floor with `bottom_gap = -0.04` (below `-penetration_tolerance_m = 0.03`), centroid above, footprint ok. **No** — `expected_failed_clauses: ["contact"]`. Single-clause rejection on the penetration side. This case *would pass* at `penetration_tolerance = 0.05` and is correctly rejected at 0.03 — it mirrors the real sofa (obj_9, −0.043) and makes the D6 threshold choice auditable as a fixture assertion, not just a note.

Header carries `numeric_tolerance_m`, the four config defaults, and `initial_fixture_frozen_before_extractor_code: true`. Each case carries `expected_on_surface` (bool) and, for negatives, `expected_failed_clauses` (list — single-element for clean isolations, multi-element for the ceiling/wrong-side sanity cases). Synthetic cases carry inline `entity_aabb` + `surface_record` (Plane `{a,b,c,d}` + polygon) so numeric assertions don't depend on whatever bundle is loaded; the Replica-grounded F1 references `entity_uid` / `surface_uid`.

---

## Phase 4 exit gates

| Gate | Name                                       | Pass condition |
|------|--------------------------------------------|----------------|
| G1   | Rest-contact determinism                   | Two runs of the extractor produce identical `ON_SURFACE` edge_id sets. |
| G2   | **Subset invariant**                       | Every `ON_SURFACE(e, s)` corresponds to a polygon-mode `NEAR_SURFACE(e, s)` edge on the same surface (under the D4a ordering guard). Violation count 0. |
| G3   | **Clean inverse**                          | `SUPPORTS_count == ON_SURFACE_count` (Design B; no role filter needed). |
| G4   | **No materialized SUPPORTS**               | Zero edges of `type == "SUPPORTS"` in `bundle.edges` — a *no-emit* count check, NOT an absence-from-schema check (`SUPPORTS` is a reserved legacy EdgeType, `graph/schema.py:21`). `supports_view` returns a projection (not `Edge` objects) and references `ON_SURFACE` edge_ids rather than copying evidence. |
| G5   | Phase 4 smoke passes                       | All cases in `phase4_on_surface_smoke.json` pass; each negative rejected by exactly its declared `expected_failed_clauses`. |
| G6   | **Default paths preserved (new isolated family)** | `ON_SURFACE` is a new, isolated edge family — preservation is by isolation + non-wiring, NOT P3-style byte-equality of a modified family (there is no shared bundle to compare): `OnSurfaceExtractor` is not wired into any default builder run (isolated like P2.09); existing `NEAR_SURFACE` / `NEAR` / directional edge bytes unchanged; Phase 2/3 exit gates still pass; canonical Phase 1/2/3 artifacts untouched by the P4 gate run. |
| G7   | Deterministic + timestamp-free artifacts   | Telemetry + report byte-stable on rerun; no timestamp keys; canonical Phase 1/2/3 artifacts untouched by the P4 gate run. |
| G8   | **Threshold-ordering enforced**            | `OnSurfaceConfig` validation raises if `hypot(contact_threshold_m, footprint_tolerance_m) > near_surface_threshold_m` (with the P4 default `footprint_tolerance_m = 0` this is `contact_threshold_m > near_surface_threshold_m`). The subset claim's precondition is a guarded config invariant, not an assumption. |

---

## Validation / success criteria (defined before any code)

The v1 benchmark is saturated and irrelevant here. Success is:

1. **Predicate correctness on the fixture.** Every fixture case resolves as specified, each negative via its named clause (P4.00 + G5).
2. **Invariants hold on real Replica room_0.** G2 (subset), G3 (clean inverse), G4 (no materialized SUPPORTS) all green; G8 ordering enforced.
3. **Default behavior preserved.** Phase 2/3 exit gates still pass; no existing edge family's bytes move; `ON_SURFACE` ships isolated (G6).
4. **Honest coverage telemetry.** Report `ON_SURFACE` edge count per surface type on Replica. Expectation: floor-only and small. If the count is 0 even on the floor, we say so and treat it as a fixture/threshold question, not a silent pass.
5. **One answerable QA template.** "What is on the floor?" returns the `ON_SURFACE(entity, floor)` set from graph structure (not retrieval). "On the table/chair?" and "against the wall?" return an explicit *deferred* result, not a fake.
6. **No benchmark-improvement claim** unless a QA eval actually changes under comparable settings. This is a graph-faithfulness phase.

---

## Risks / confounders

- **Confounder #1 — narrow support surface.** Only the floor is a real support surface in Replica room_0, so `ON_SURFACE` coverage is small and floor-only. Mitigation: telemetry reports honestly; success is the predicate + invariants + the fixture (which we control), not opportunistic Replica counts.
- **Confounder #2 — provisional thresholds.** `contact_threshold`, `penetration_tolerance`, `max_tilt`, `footprint_tolerance` are Replica-calibrated config, not generalization evidence. The fixture pins behavior; changing a threshold that breaks the fixture is a behavior change, not a tuning step.
- **Risk — normal orientation.** The predicate assumes interior/up-facing normals. A flipped normal would invert `support_capable` + `centroid_on_support_side` together. Mitigation: P4.01 validates orientation and raises rather than emitting a wrong edge.
- **Risk — footprint strictness vs mesh noise.** `footprint_tolerance_m = 0` is honest but brittle if importer noise pushes a resting entity's projected footprint a hair off the polygon. Mitigation: the fixture pins it; relaxation is an explicit, documented config change with a new exit-gate fixture, not a silent bump.
- **Risk — SUPPORTS leakage.** A future change could accidentally materialize `SUPPORTS`. Mitigation: G4 is a structural test (`"SUPPORTS"` never an `EdgeType`), not a count check.
- **Non-risk to call out.** This does NOT touch `bbox_to_plane`, `bbox_to_surface`, or `NEAR_SURFACE`. `ON_SURFACE` is a new, additive, isolated edge family.

---

## What this plan deliberately does not do

- Does not introduce `EntitySurface` or any furniture-top geometry (D2).
- Does not introduce wall/ceiling contact relations (D1).
- Does not materialize `SUPPORTS` (D3).
- Does not wire `ON_SURFACE` into any default builder run (G6).
- Does not change Phase 2/3 smoke fixtures, `NEAR_SURFACE`, `NEAR`, or directional edges.
- Does not promote polygon-clip to default (separate P3.06 decision).
- Does not rewrite existing Phase 1/2/3 artifacts despite the `schema_version` 2→3 bump (D5) — they stay at their authored version; the bump is an additive artifact-schema change, not a model improvement.
- Does not emit `SUPPORTS` edges or reuse the `SUPPORTS` EdgeType for the derived view (D3).
- Does not claim accuracy lifts on the v1 benchmark.

---

## Phase 5 preview (not part of this plan)

With a clean support primitive and the derived-view pattern proven, later phases can add — each honestly named and each needing its own geometry or data:

- **`CONTACTS_SURFACE` / `ATTACHED_TO`** for wall-mounted entities (normal-side contact, no gravity-rest), with `ATTACHED_TO` derivable the same way `SUPPORTS` is here.
- **`HANGS_FROM`** for ceiling-mounted entities.
- **`EntitySurface`** (tabletop / seat) as a first-class target with real provenance, unlocking "what's on the table/chair?" — requires furniture-part geometry, not whole-AABB synthesis.
- **`IN` / `CONTAINED_BY`**, requiring concave-region detection or labeled containers.

None of these are committed in Phase 4.

---

## Closing note

Phase 4 takes the more-faithful geometry Phase 3 produced and reads one honest relation off it: *resting on*. It adds exactly one materialized edge family, derives `SUPPORTS` without storing it, and refuses to answer "what's on the table?" until the data can support that honestly. The win is a small, correct support primitive with provable invariants — not a reasoning leap, and not a benchmark number.

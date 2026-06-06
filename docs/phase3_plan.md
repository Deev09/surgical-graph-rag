---
title: Phase 3 — Polygon-clipped NEAR_SURFACE (amended draft)
status: draft — awaiting fixture-freeze (P3.00) before any code work
date: 2026-06-04
tags: [phase-3, draft, geometry, near-surface]
---

# Phase 3 — Polygon-clipped NEAR_SURFACE

> [!info] Status: DRAFT — amendments folded in 2026-06-04
> Geometry-correctness phase. Closes the infinite-plane limitation recorded in [[phase2_summary]]. Does NOT yet add support / contact / containment reasoning — that remains a later phase.
> **Gate**: no code lands until the Phase 3 smoke fixture (P3.00) is frozen and committed.

Related: [[phase0_design]] (frozen contracts), [[phase1_summary]] (baseline), [[phase2_summary]] (geometry substrate), [[phase2_plan]] (prior phase).

---

## What this phase accomplishes (and what it does not)

**Accomplishes.** Replaces the infinite-plane distance used by NEAR_SURFACE with a finite-polygon distance. Phase 2 asks "is this entity close to the wall's infinite mathematical plane?"; Phase 3 asks "is it close to the wall's actual finite polygon?" That removes false positives where an entity lies near the same plane but outside the surface's extent.

**Does not accomplish.** Support / contact / containment relations ("what's on the chair?", "what's inside the box?") are deferred. Phase 3 makes the geometry substrate more faithful so a future phase can build those relations on top of it. NEAR_SURFACE remains a proximity relation, not a support relation.

This distinction matters for evaluation: a clean Phase 3 may show *fewer* edges, not more, and that is the correct outcome.

---

## Scope

**In:**

- `geometry/surface_distance.py`: new `aabb_to_polygon_planar(aabb, plane, polygon) -> float` — generic polygon-vs-polygon distance in plane-local 2D.
- `geometry/surface_distance.py`: new dispatcher `bbox_to_surface(aabb, surface_record) -> tuple[float, dict]` returning final distance + evidence dict (normal gap, in-plane gap, polygon-clipping-applied flag, fallback reason).
- `graph/relations/surface.py`: opt-in `use_polygon_clip: bool = False` on `SurfaceProximityConfig`. Default OFF preserves Phase 2 NEAR_SURFACE bytes; enabled path emits a new extractor version and richer evidence.
- New telemetry: per-surface-type plane-vs-polygon edge-flip counts, monotonicity check artifact.
- New exit gate G8 (polygon-clip determinism + monotonicity).
- **New** Phase 3 smoke fixture, separate from Phase 2's.

**Out (deferred, named explicitly):**

- OBB-to-OBB surface distance (Phase 2 A5 still parked).
- Promotion of sparse_v2 to primary candidate path.
- Any change to density policy (`phase1_block` / `phase2_telemetry_only` unchanged).
- Any change to floor/wall/ceiling extraction in the importer.
- Any change to `SurfaceRecord` schema or graph serde version.
- Any new relation type (no SUPPORTS / IN / ON).
- Any change to Phase 2 smoke list — it stays frozen as regression.

---

## Required amendments (gating)

### A1. New fixture, do not amend Phase 2 smoke

Phase 2's `eval/questions/phase2_near_surface_smoke.json` stays frozen as a regression bed. Create a *separate* file `eval/questions/phase3_near_surface_polygon_smoke.json` containing the harder cases that distinguish plane-only from polygon-clipped behavior.

Minimum content (≥ 8 cases):

- ≥ 2 *true near* cases inside polygon extent (must pass in both modes).
- ≥ 2 *plane-only false positives*: entity within threshold of plane but outside polygon footprint in plane-local coordinates. These must be NEAR in Phase 2 mode and NOT_NEAR in Phase 3 mode — they are the cases this phase exists to fix.
- ≥ 1 *boundary case*: entity straddling polygon edge.
- ≥ 1 *fallback case*: surface with `polygon=None` where dispatcher falls back to plane distance. Because NEAR_SURFACE skips `synth_bbox_fallback` by default (Phase 2 A3), this case must either (a) use a non-synth surface that happens to have `polygon=None`, or (b) be authored with `include_synth_fallback=True` and document that opt-in.
- Each case carries `expected_distance_metric` ("bbox_to_plane" or "polygon_clipped") and `expected_in_plane_gap` so the test asserts on evidence, not just the boolean.

**Two additional fixture rules (added 2026-06-04):**

- **Prefer synthetic fixture entries for plane-vs-polygon disagreement cases.** Real Replica room_0 may not naturally contain entities that are close to a wall's plane but outside the wall's lateral extent — the room geometry constrains where objects can be. Disagreement cases (the ones this phase exists to fix) should be authored as *synthetic* fixture entries with hand-specified AABBs and surfaces, not scraped from the Replica entity bundle. Mark each synthetic case with `synthetic: true` and include the constructed `entity_aabb` and `surface_record` inline. Replica-grounded cases are still welcome for the "true near" and "boundary" classes, where the room geometry can supply them.
- **Include explicit geometry for any case that requires exact `expected_in_plane_gap`.** The fixture (or a companion fixture module loaded by the test) must carry the actual `entity_aabb`, `surface_polygon`, and `surface_plane` for every case that asserts a numeric `expected_in_plane_gap`. Otherwise the assertion silently depends on whatever bundle is loaded that day. For cases that only assert booleans (`is_near` / `is_not_near`), referencing `entity_uid` / `surface_uid` is acceptable.

Fixture frozen and committed BEFORE any geometry code in `geometry/surface_distance.py` is touched.

### A2. Generic 2D polygon distance — not rectangle-specialized

A projected AABB does not always produce a rectangle in plane-local 2D (only when the AABB is axis-aligned with the plane; arbitrary plane orientations can yield up to a hexagon). The algorithm:

1. Project all 8 AABB corners into plane-local 2D coords (orthonormal basis on the plane).
2. Take the convex hull of those 8 points → convex polygon `A` (3 to 6 vertices, generically).
3. Compute `dist_2d(A, B)` where `B` is the surface polygon (treated as a region, not an oriented curve):
   - If `A` and `B` overlap (any vertex of A in B, any vertex of B in A, or any edge crossing) → `dist_2d = 0`.
   - Otherwise: `dist_2d = min(min vertex-of-A → edge-of-B, min vertex-of-B → edge-of-A, min edge-of-A → edge-of-B)`.
4. Combine with normal-axis gap: `final = hypot(dist_2d, bbox_to_plane(aabb, plane))`.

Winding-agnostic: containment uses ray casting (no signed-area assumption); edge-edge uses segment-segment distance (no orientation assumption). Polygons need not be convex in general, though Replica's are rectangles for now.

### A3. Wording fix

"Outside floor xz polygon" is wrong because "xz" hard-codes a world-axis assumption that the gravity-aligned floor may not satisfy in general. Replace throughout this plan and any future doc with **"outside the floor polygon in plane-local coordinates"** or **"outside the floor footprint"**. Same fix for ceilings.

### A4. Phase 2 byte-equality must be tested, not assumed

`SurfaceProximityConfig.use_polygon_clip` must be declared with `field(default=False, metadata={"hash_omit_if_default": True})` exactly like Phase 2's `ProximityConfig.sparse_version`. This is the mechanism that keeps the bundle hash byte-equal for default Phase 2 configs.

Mandatory new test, in `tests/relations/test_near_surface_polygon.py`:

```python
def test_default_phase3_config_produces_phase2_byte_equal_graph():
    # Build Replica room_0 graph with SurfaceProximityConfig() (default Phase 3 = Phase 2 behavior).
    # Build same graph with explicit SurfaceProximityConfig(use_polygon_clip=False).
    # Assert bundle_hash equality with the canonical Phase 2 graph hash.
    # Assert edge_id-by-edge_id equality with the canonical Phase 2 NEAR_SURFACE edge set.
```

If this test fails, the build is not allowed to merge.

### A5. Polygon mode gets its own extractor version + richer evidence

Confirmed: Phase 2 NEAR_SURFACE edges carry `extractor_version="0.1"` (plain, no suffix — see `graph/relations/surface.py:123`). Phase 3 polygon-clipped edges carry a distinct version string so plane-mode and polygon-mode edges cannot collide on `edge_id`:

- `extractor_version = "0.2-near_surface_polygon_clipped"` (truthful — algorithm changed; distinct from `"0.1"`).
- `evidence["distance_metric"] = "polygon_clipped"` (vs `"bbox_to_plane"` — confirmed current Phase 2 value, `graph/relations/surface.py:113,192`).
- `evidence["normal_gap_m"]` — output of `bbox_to_plane`.
- `evidence["in_plane_gap_m"]` — output of `dist_2d`.
- `evidence["distance_m"]` — the combined dispatcher distance (same key Phase 2 uses for its single-component answer; in polygon mode this is `hypot(normal_gap_m, in_plane_gap_m)`).
- `evidence["threshold_m"]` — the configured threshold for this surface type (unchanged from Phase 2).
- `evidence["polygon_clipping_applied"] = True` / `False` (False means the dispatcher fell back to plane).
- `evidence["fallback_reason"] = "polygon_none"` if applicable; absent otherwise.

The plane-only default path keeps Phase 2's evidence schema unchanged (no new keys) so byte equality holds.

### A6. Monotonicity tests

For any AABB `B`, plane `P`, polygon `poly` on `P` and threshold `t`:

- `bbox_to_surface(B, sr_with_poly).distance >= bbox_to_plane(B, P)` — clipping can only increase (or hold) the *final* dispatcher distance. The bare `aabb_to_polygon_planar` helper is only the in-plane gap and is NOT dimensionally comparable to `bbox_to_plane`; never compare them directly. Property test with random AABBs + random polygons (seeded RNG; we do not use `Math.random()` semantics, fixed seed via fixture).
- For the same scene and same `t`: `{edges_in_polygon_mode} ⊆ {edges_in_plane_mode}` **for surfaces where polygon is present**. Surfaces with `polygon=None` are excluded from this subset claim because the dispatcher falls back to plane mode and the edge sets coincide for those surfaces.
- Telemetry artifact `phase3_polygon_clip_monotonicity.json` records both edge counts per surface type and any subset violation (should be zero — if not, geometry is wrong).

### A7. Synth-fallback test care

Phase 2's `SurfaceProximityExtractor` skips `source="synth_bbox_fallback"` surfaces unless `include_synth_fallback=True`. The Phase 3 fallback case (entity near a surface where `polygon is None`) must be designed so the skip behavior does NOT swallow it:

- Option (a): use a `source="mesh_ransac"` surface (or another non-synth source) that legitimately has `polygon=None`. Document why polygon is missing for that source.
- Option (b): if no such surface exists in Replica room_0, author the fallback test with `SurfaceProximityConfig(use_polygon_clip=True, include_synth_fallback=True)` and explicitly state that this configuration is for the fallback test only, not the default Phase 3 build.

Whichever option is chosen, the fixture entry documents the rationale in a `notes` field.

---

## Tasks

> P3.00 is gating. P3.01..P3.07 do not begin until P3.00 is committed.

| ID    | Title                                                                  | Insertion point                                                              | Gates on | Status |
|-------|------------------------------------------------------------------------|------------------------------------------------------------------------------|----------|--------|
| P3.00 | Freeze Phase 3 smoke fixture (A1)                                      | `eval/questions/phase3_near_surface_polygon_smoke.json`                      | —        | ✅ done (commit 99af4dc; S7 amended in P3.05 commit, see `post_freeze_amendments` header) |
| P3.01 | Implement `aabb_to_polygon_planar` (generic, A2) + monotonicity tests  | `geometry/surface_distance.py`, `tests/geometry/test_polygon_distance.py`     | P3.00    | ✅ done (commit fa76bff) |
| P3.02 | Implement `bbox_to_surface` dispatcher + evidence dict                 | `geometry/surface_distance.py`, `tests/geometry/test_bbox_to_surface.py`      | P3.01    | ✅ done (commit ffed11d) |
| P3.03 | Wire `use_polygon_clip` opt-in into NEAR_SURFACE (A4 byte-equality test) | `graph/relations/surface.py`, `tests/relations/test_near_surface_polygon.py` | P3.02    | ✅ done (commit fe0507c). A5 (distinct extractor version + richer evidence) folded into P3.03 per sign-off — emitting opt-in edges with version "0.1" would have collided with Phase 2 edge_ids. |
| P3.04 | Update extractor version + evidence schema (A5)                        | `graph/relations/surface.py`                                                  | P3.03    | ✅ no-code closeout — A5 materially landed in P3.03 (`POLYGON_CLIPPED_VERSION="0.2-near_surface_polygon_clipped"`; opt-in edges carry `normal_gap_m`, `in_plane_gap_m` / `fallback_reason`, `polygon_clipping_applied`, `distance_metric`; verified by `tests/relations/test_near_surface_polygon.py`). |
| P3.05 | Polygon-clip telemetry + monotonicity report (A6)                      | `tools/phase3_polygon_clip_telemetry.py`, `tests/tools/test_phase3_polygon_clip_telemetry.py`, `scenes/replica_room_0/eval/phase3_polygon_clip_telemetry.json` | P3.03 (P3.04 folded in)    | ✅ done. Replica room_0 result: 98 plane-mode edges → 76 polygon-mode edges (22 plane-only false positives removed); A6 subset claim holds with 0 violations; A4 bundle_hash equivalence holds at full-scene scale; deterministic + timestamp-free + byte-identical on rerun. Also loosened `POLYGON_ON_PLANE_TOL_M` from 1e-6 to 1e-4 m after surveying real-data drift (worst case 3e-6 m). |
| P3.06 | Decide: promote polygon-clip to default or keep opt-in                 | docs only (this file → closeout section)                                      | P3.05    | ✅ done — **keep opt-in**. See "P3.06 closeout" section below for evidence, reasoning, and promotion gate for a future phase. |
| P3.07 | Phase 3 exit gate (G1–G7 ported + new G8)                              | `tools/phase3_exit_gate.py`, `tests/tools/test_phase3_exit_gate.py`, `scenes/replica_room_0/eval/phase3_exit_gate_report.json` | P3.06    | ✅ done. 8 blocking gates (G1, G2, G3, G4 both-candidates, G5a Phase 2 smoke under default config, G5b Phase 3 smoke under opt-in, G7, G8 multi-claim) all pass on real Replica room_0. G6 records combined density for BOTH candidates (plane: 16.151/entity; polygon: 15.849/entity; Phase 1 cap 14 telemetry-only). Report is deterministic + timestamp-free + byte-stable on rerun (tested). Phase 1/2 artifacts and the P3.05 telemetry artifact are not touched (tested). |

---

## Phase 3 exit gates

Ported from Phase 2 unchanged where applicable; G8 is new.

| Gate | Name                                       | Pass condition                                                                                                           |
|------|--------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| G1   | Structural surfaces present                | Same as Phase 2.                                                                                                          |
| G2   | World-frame OBBs                           | Same as Phase 2.                                                                                                          |
| G3   | Phase 1 compat reproduction                | Same as Phase 2 (byte-equal 5414/5414).                                                                                   |
| G4   | Deterministic + replayable                 | Same as Phase 2, extended to cover polygon-clip mode.                                                                     |
| G5a  | Phase 2 NEAR_SURFACE smoke (regression)    | 12/12 cases pass with default config — Phase 2 fixture unchanged.                                                         |
| G5b  | **Phase 3 polygon smoke (new)**            | All cases in `phase3_near_surface_polygon_smoke.json` pass with `use_polygon_clip=True`; evidence keys match expectations.|
| G6   | Density telemetry                          | Same as Phase 2; recorded, not blocking.                                                                                  |
| G7   | Builder structural completeness            | Same as Phase 2.                                                                                                          |
| **G8** | **Polygon-clip determinism + monotonicity** | (a) Two runs produce identical edges in polygon mode. (b) Default (plane) config produces Phase 2 byte-equal graph hash. (c) Polygon edge set is a subset of plane edge set on surfaces with polygon present. (d) `bbox_to_surface(B, surface_with_polygon).distance >= bbox_to_plane(B, P)` holds across property test. (`aabb_to_polygon_planar` alone is only the in-plane component — not dimensionally comparable to `bbox_to_plane`. The monotonicity claim is on the dispatcher's final distance, not the planar helper.) |

---

## Validation / success criteria (defined before any code)

Phase 2 is already correct/correct on its smoke. Success here is measured by:

1. **Flip count > 0 on the new Phase 3 fixture.** If polygon-clip and plane-only produce identical edges on the amended-hard fixture, the fixture was too easy. We harden the fixture, NOT skip the change.
2. **Phase 2 byte-equality preserved.** A4 test green. Default Phase 2 graph hash and edge IDs are bit-identical post-merge.
3. **Monotonicity holds.** A6 property tests green; subset relation holds on Replica room_0 telemetry.
4. **Phase 2 NEAR_SURFACE regression bed unchanged.** Phase 2 fixture (12 cases) still 12/12.
5. **G1–G8 all green** in `phase3_exit_gate_report.json` (deterministic artifact, no timestamps, same discipline as Phase 2).
6. **Honest telemetry on real Replica room_0.** Report flip count per surface type. If zero on real Replica even with polygon-clip on, we say so and queue this for future scenes — we do not claim improvement on a scene where the flaw does not manifest.

---

## Risks / confounders

- **Confounder #1.** Replica room_0 polygons may be tight enough relative to objects that the plane-vs-polygon distinction is rare. Mitigation: telemetry reports honestly; success is measured on the new fixture (which we control) + property tests, not on opportunistic Replica wins.
- **Confounder #2.** The floor polygon for room_0 may be authored to extend under furniture, masking outside-polygon cases. Investigate at P3.00 fixture authoring time; design cases around what the polygons actually cover.
- **Risk.** Generic polygon-vs-polygon distance has more edge cases than rectangle-only (collinear edges, vertex-on-edge, near-zero segment-segment distances). Property tests + named edge-case unit tests in `tests/geometry/test_polygon_distance.py`.
- **Risk.** Polygon winding inconsistency across importers. A2 (winding-agnostic algorithm) mitigates; also add an explicit test that reversing polygon vertex order produces identical distance.
- **Risk.** `hash_omit_if_default` on `use_polygon_clip` must be added correctly or Phase 2 byte-equality breaks silently. A4 test catches this; review at code time.
- **Non-risk to call out.** This does NOT change `bbox_to_plane` itself. Existing callers (only the surface extractor today) are unaffected on the default path.

---

## What this plan deliberately does not do

- Does not modify Phase 2 smoke fixture (A1 forbids it).
- Does not touch sparse_v1 / sparse_v2 (separate decision).
- Does not change density policy.
- Does not change `SurfaceRecord` schema or graph serde version.
- Does not introduce support / contact / containment relations.
- Does not specialize for rectangles (A2 forbids it).
- Does not regenerate the canonical enriched-v2 artifact.

---

## Phase 4 preview (not part of this plan)

After Phase 3 closes, the substrate is ready for support / contact reasoning. Likely candidates:

- **SUPPORTS / SUPPORTED_BY** between entity pairs, derived from gravity + AABB overlap + small vertical gap.
- **ON_SURFACE** between entity and SurfaceRecord, derived from polygon-clipped near-floor / near-table proximity + gravity-up axis check.
- **IN / CONTAINED_BY**, requiring concave-region detection or labelled containers (likely needs more importer work).

None of these are committed in Phase 3.

---

## P3.06 closeout — default vs opt-in decision

**Decision: keep `use_polygon_clip` opt-in. Do not promote to default in Phase 3.**

### Evidence (from `scenes/replica_room_0/eval/phase3_polygon_clip_telemetry.json`)

- Plane mode: **98** NEAR_SURFACE edges (19 floor / 55 wall / 24 ceiling).
- Polygon mode: **76** edges (19 floor / 46 wall / 11 ceiling).
- Flipped near → not_near: **22** (0 floor / 9 wall / 13 ceiling). Flipped not_near → near: **0**.
- A6 subset claim on polygoned surfaces: **0 violations**.
- A4 byte-equivalence at GraphBuilder bundle_hash level: holds.

### Why not promote yet

- One scene, one importer (Replica room_0 via the oracle adapter), one relation family (NEAR_SURFACE). The 22-edge reduction is geometrically sound but is not corroborated by a second scene or by a downstream relation-eval result.
- Promotion changes default graph contents. The Phase 2 baseline assumptions, the canonical exit-gate artifact, and any downstream eval table that joins on edge sets would shift. The discipline this repo has held since Phase 1 — default paths stay comparable, candidate paths are explicit — is the discipline this commit preserves.
- The ceiling-polygon flips (13 / 22) are concentrated in a small number of room-geometry-specific configurations. There is no per-relation human-labeled "true near" set against which to verify that those 13 are *all* false positives versus a few being borderline-true near-ceiling cases the polygon clip dropped too aggressively. Until such a check exists, the conservative reading is "polygon mode is a better candidate" — not "polygon mode is the new ground truth."

### Promotion gate (for a future phase)

Promote `use_polygon_clip=True` to the default `SurfaceProximityConfig` only after **both** of the following land:

1. A downstream relation-eval (or QA evaluation that exercises NEAR_SURFACE-derived behavior) shows no regression — i.e., the 22 dropped edges do not include cases the downstream system was relying on as true positives.
2. At least one additional scene OR a hand-labeled NEAR_SURFACE relation set confirms the false-positive reduction generalizes beyond Replica room_0.

Until both gates clear, polygon mode remains the explicit candidate. The constants `PLANE_MODE_VERSION` / `POLYGON_CLIPPED_VERSION` and the `hash_omit_if_default` metadata mean opting in or out is a one-line config flip with no Phase 2 byte impact.

### Implications for P3.07 (Phase 3 exit gate)

P3.07 must assert **both** claims simultaneously:

- Phase 2 default behavior remains byte-equivalent (default config produces the Phase 2 baseline graph hash; default-path NEAR_SURFACE edge_id set matches the Phase 2 reference exactly).
- Polygon candidate passes G8 (determinism + A6 subset monotonicity + A4 hash divergence proving opt-in is hashed) and is ready for future promotion.

The honest framing: polygon mode is better geometry, but not yet the new baseline.

---

## Closing note

Phase 3 is a geometry-correctness phase, not a reasoning phase. Its win is a more faithful proximity graph and a removed limitation. The Phase 2 → Phase 3 diff should look like *fewer*, more honest, NEAR_SURFACE edges on cases where the infinite plane was lying — visible only when callers opt in. Anything else — claiming reasoning improvements or accuracy lifts on the 10-query baseline — would be a benchmark-semantics change dressed as a model improvement.

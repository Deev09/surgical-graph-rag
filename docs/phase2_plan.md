---
title: Phase 2 — Geometry Enrichment (amended)
status: amended v2 — conditional sign-off received; ready to start at P2.01
date: 2026-05-31
tags: [phase-2, amended, geometry]
---

# Phase 2 — Geometry Enrichment

> [!info] Status: AMENDED v2 — conditional sign-off received 2026-05-31
> Scope confirmed correct. Eight amendments folded in below; Q1–Q6 decided. Phase 2 starts at P2.01 once raw Replica inputs are on disk.
> Original draft text retained where unchanged; amended passages are marked **AMENDED:**.

Phase 2 builds the geometry substrate that Phase 3 support / containment / attached relations will rely on. It does NOT touch learned backends — that is Phase 4 per the corrected scope.

Related: [[phase0_design]] (frozen contracts), [[phase1_summary]] (what landed).

---

## P2.10 closeout — 2026-06-01

All blocking gates green on Replica room_0. Decisions and limitations recorded so they survive into Phase 3.

### Density policy

| Build path | Policy | Behavior |
|---|---|---|
| Phase 1 sparse-v1 (default) | `density_policy="phase1_block"` | Raises `GraphBuildError` when `logical_edges / entity_count > 14`. UNCHANGED. |
| Phase 2 candidate (sparse-v2 + NEAR_SURFACE) | `density_policy="phase2_telemetry_only"` | Records `density_ratio` in `BuildDiagnostics`. Does NOT raise. Caller opts in explicitly. |

`SPARSE_DENSITY_LIMIT = 14` is unchanged. The cap was NOT silently raised. The combined Phase 2 candidate runs at **density 16.151/entity** on Replica room_0 (G6 telemetry), recorded honestly in `phase2_exit_gate_report.json`.

### C1 — graph-level `SurfaceRecord` (landed)

- `graph/schema.py`: new `SurfaceRecord(uid, surface_type, plane, polygon, source, confidence)`. `SceneGraphBundle` gained `structural_surfaces: list[SurfaceRecord]`; `structural_surface_refs` is now derived from this list (NOT from the edge set).
- `graph/builder.py`: populates `structural_surfaces` from the entity bundle; rejects edges referencing unknown entity OR surface UIDs with `GraphBuildError` (G7).
- `graph/serde.py`: `CURRENT_SCHEMA_VERSION` bumped 1 → 2; the new field round-trips through dump/load.
- `BuildDiagnostics` gained `density_policy`, `density_ratio`, `sparse_density_limit` (recorded, not hashed — Phase 1 bundle_hash is preserved bit-identical).

### NEAR_SURFACE infinite-plane limitation (Phase 2 acceptable; Phase 3 may revisit)

`bbox_to_plane` measures distance to the **infinite plane** defined by `SurfaceRecord.plane`, NOT clipped to `polygon` extents. An entity above the same plane but well outside the polygon footprint still registers as NEAR. This is documented in:

- `graph/schema.py:SurfaceRecord` docstring,
- `tools/phase2_exit_gate.py` artifact `limitations_recorded_for_phase3`,
- this section.

Phase 3 may add polygon-clipped variants if support / containment detection requires them.

### Phase 2 exit gate snapshot (canonical)

```
G1_structural_surfaces            PASS  (1 floor + 5 walls + 1 ceiling, all habitat_label)
G2_world_frame_obbs               PASS  (73/73 entities with bbox_obb)
G3_phase1_compat_reproduction     PASS  (byte-equal 5414/5414)
G4_deterministic_and_replayable   PASS  (two runs equal; dump→load equal)
G5_near_surface_smoke             PASS  (frozen 12 cases; 6 near + 6 not_near)
G6_density_telemetry              16.151/entity (cap 14; exceeds=True)  ← telemetry only
G7_builder_structural_completeness PASS (full retention + unknown-UID rejection)
```

Artifact: `scenes/replica_room_0/eval/phase2_exit_gate_report.json` (deterministic, no timestamp).

---

## Review response — 2026-05-31

Conditional sign-off received. Recording the eight required amendments and the Q1–Q6 calls so the rest of this document is unambiguous. Each amendment is also threaded into the task that owns it.

### Eight required amendments

| # | Amendment | Threaded into |
|---|---|---|
| A1 | Do **not** overwrite the Phase 1 replay fixture. `adapters/oracle_replica.py:58` hashes file contents, so additive fields would silently change Phase 1 bundle hashes. Emit enriched importer outputs to a versioned path: `scenes/replica_room_0/enriched/v2/`. Phase 1 reads from the existing `scene_graph.json` / `capture_meta.json` paths, untouched. | Hard constraints, P2.02, P2.03, P2.06, Retirement/preservation |
| A2 | Fix the NEAR expectation in P2.08. Surface-to-surface distance is generally **smaller** than centroid-to-centroid distance, so the same 1.0m threshold typically yields **more** edges, not fewer. "Typically smaller NEAR set" was wrong. | P2.08, G6 (renamed from G-density-telemetry) |
| A3 | `capture_meta.room_bbox` is **not** a canonical surface fallback. `importers/replica.py:123` derives it from object centroids, not room geometry. Bbox synthesis is allowed **only** as a clearly labeled non-blocking experiment, with `source="synth_bbox_fallback"` on every emitted surface. | Q3, P2.03 |
| A4 | Tighten geometry math in P2.07. `bbox_to_plane` is **non-negative** and returns 0 when the plane intersects the box (not "min over signed corner distances"). `aabb_to_aabb_surface` is the **Euclidean norm of positive axis gaps**, not min-gap on the dominant axis. | P2.07 |
| A5 | Reject closest-corner OBB-OBB distance as a quality gate — it misses face-to-face and intersecting-box cases. Phase 2 uses **exact AABB surface distance** as the boring provisional metric. OBBs are still emitted as a payload (P2.02) for Phase 3, where SAT/GJK can be added if support detection needs it. | P2.07, P2.08, Q5 |
| A6 | Extend the builder. `graph/builder.py:228` currently records only surfaces referenced by emitted edges. The Phase 2 graph must **retain all structural surfaces** from the entity bundle AND **reject edges referencing unknown entity or surface UIDs** with a typed `GraphBuildError`. | P2.10, G7 (new) |
| A7 | Add per-surface provenance. `StructuralSurface` in `extractors/base.py:38` needs a `source: Literal["habitat_label", "mesh_ransac", "synth_bbox_fallback"]` field, propagated through the importer → extractor → graph node. | P2.03, P2.04, P2.06 |
| A8 | G5 must include negative controls. A positives-only NEAR_SURFACE smoke list cannot catch an extractor that fires too freely. Frozen smoke list adds explicit `not_near` expectations alongside `near` ones. | G5 |

### Q1–Q6 — decisions

| # | Question | Decision |
|---|---|---|
| Q1 | Raw Replica data acquisition | **User supplies locally.** Phase 2 ships a `tools/verify_replica_inputs.py` verifier that prints sha256 + sizes and exits 0/1. No silent canonical fallback. If verifier fails, Phase 2 is gated at P2.01. |
| Q2 | Importer `schema_version` | **Add `schema_version: 2`.** Emit to the new versioned output path `scenes/replica_room_0/enriched/v2/` (see A1). |
| Q3 | Surface-fitting strategy | **Habitat structural labels first, mesh RANSAC second. Bbox synthesis is experiment-only**, tagged `source="synth_bbox_fallback"`, never the canonical path (see A3). |
| Q4 | NEAR_SURFACE thresholds 0.05 / 0.30 / 0.10 m | **Accepted as provisional recorded config values, not quality evidence.** Same status as Phase 1's `sparse_max_distance=2.5` — Replica-calibrated, not generalization evidence. |
| Q5 | OBB-to-OBB distance | **Use exact AABB surface distance for Phase 2** (see A5). Robust OBB distance deferred to Phase 3 if support detection needs it. |
| Q6 | Sparse-v2 density cap | **Record as telemetry; do not block on `≤ 14`** for sparse_v2. The blocking `≤ 14` rule stays attached to frozen sparse_v1 only. Phase 2 exit gate logs `actual_ratio` without failing on it. |

### Scope alignment (recorded)

Phase 2 builds the geometry substrate required for "on the chair" / "near the wall" relations. It does **not** prove spatial reasoning quality. That starts in Phase 3, when support / containment / attached extractors are evaluated against labeled edges.

### Post-amendment clarifications

| # | Clarification | Status | Threaded into |
|---|---|---|---|
| C1 | **Graph-level Surface record needed.** `SceneGraphBundle.structural_surface_refs: list[str]` (graph/schema.py:113) currently stores **UIDs only**. A6 (retain all surfaces) and A7 (provenance survives) require an explicit graph-side schema, not a string list. Surfaces are NOT graph nodes (the Node docstring at graph/schema.py:43 says so) — they get a dedicated record type. | NON-BLOCKING for P2.01; MUST land before P2.10 G7 can pass | New schema work below; P2.05 round-trip; P2.10 G7 |

**C1 — proposed schema** (frozen here so P2.05 round-trips and the G7 test target the same thing):

```python
# graph/schema.py — NEW alongside Node / Edge
@dataclass(frozen=True)
class SurfaceRecord:
    """Graph-level structural-surface record.
    Surfaces are NOT graph nodes — edges reference them via
    GraphRef(kind="surface", uid=...). This record carries the geometry
    and provenance forward from EntityArtifacts.StructuralSurface so the
    graph alone is sufficient for downstream consumers.
    """
    uid: str
    surface_type: Literal["floor", "wall", "ceiling"]
    plane: Plane
    polygon: list[Vec3] | None
    source: Literal["habitat_label", "mesh_ransac", "synth_bbox_fallback"]
    confidence: float
```

**C1 — SceneGraphBundle change.** Add `structural_surfaces: list[SurfaceRecord]` alongside the existing `structural_surface_refs: list[str]` (kept for back-compat read paths; `refs` becomes a derived view = `[s.uid for s in structural_surfaces]`). This is a graph-schema additive change — bump `CURRENT_SCHEMA_VERSION` on the graph side per the existing rule in `graph/schema.py`.

**C1 — builder change** (folded into A6, P2.10, G7):
- The builder populates `structural_surfaces` from `EntityArtifacts.structural_surfaces` directly, including provenance.
- `structural_surface_refs` is derived from `structural_surfaces`, not the edge set.
- Edges with `GraphRef(kind="surface", uid=...)` whose `uid` is not in `{s.uid for s in structural_surfaces}` are rejected with `GraphBuildError`. Same rule for unknown entity UIDs.

**C1 — serde + round-trip.** P2.05 covers the new field. The graph-side schema-version bump goes in the same task that lands C1.

**C1 status.** Non-blocking for P2.01 (data acquisition does not touch the graph). Must land **before** P2.10 because G7 explicitly asserts the builder retains all surfaces and rejects unknown-UID edges — without C1, "retain all surfaces" cannot be expressed in the bundle.

---

## Goals & non-goals

**Goal.** Make the geometry inside `EntityArtifacts` and the new `StructuralSurface` slots rich enough that the Phase 3 support / containment / attached extractors have a trustworthy substrate. Phase 1's centroid-only world is too thin to reason about contact or surface-relative position correctly.

**Non-goals.**

- No ON_TOP_OF, ATTACHED_TO, INSIDE, CONTAINS, or SUPPORTS extractors. Phase 3.
- No learned backends (NeRF / 3DGS / OpenMask3D / LangSplat / OpenSplat3D). Phase 4 — gated on §10 G1–G6.
- No reasoner changes. The new edges show up but the rules compiler is unchanged.
- No benchmark schema changes. v0.2 stays.

---

## Hard constraints

| Constraint | Implication |
|---|---|
| Raw Replica inputs are not checked in (`info_semantic.json`, mesh) | Phase 2 is gated on acquiring them locally (Q1). `tools/verify_replica_inputs.py` is the gate; no silent canonical fallback. |
| **AMENDED (A1):** Do not overwrite the Phase 1 replay fixture | `adapters/oracle_replica.py:58` hashes the contents of `scene_graph.json` + `capture_meta.json`. Additive importer fields would change Phase 1 bundle hashes. Phase 2 emits enriched importer outputs to a **new versioned path** `scenes/replica_room_0/enriched/v2/` (see Retirement / preservation). Phase 1 reads stay byte-identical. |
| Phase 1 compat must reproduce exactly | The compat reproduction gate (P1.08) must still pass byte-for-byte after every Phase 2 change. No exceptions. |
| New behavior goes in a new sparse config version | The `sparse_v1` config from Phase 1 stays semantically frozen. New geometry-based behavior lives under `sparse_v2` and is opt-in. Default `mode="sparse"` continues to mean `sparse_v1` until explicitly upgraded. |
| Surface enrichment must have stable identity | Every emitted `StructuralSurface` carries a `surface_uid` that is deterministic and replayable across re-runs of the same input (just like `object_uid`). |
| **AMENDED (A7):** Every surface carries provenance | `StructuralSurface.source ∈ {"habitat_label", "mesh_ransac", "synth_bbox_fallback"}`. Synth fallback surfaces are flagged everywhere they appear so the gate logic and downstream consumers cannot mistake them for oracle ground truth. |
| No learned signal | Every Phase 2 emission is derived geometrically from oracle inputs. No CLIP, no clustering, no thresholds tuned on out-of-sample data. |

---

## Architecture additions (where new things go)

```
importers/
  replica.py                       — EXTENDED: emit OBBs + structural surfaces +
                                     per-surface `source` provenance.
                                     AMENDED (A1, A2, A7): writes to a NEW versioned
                                     output path scenes/replica_room_0/enriched/v2/
                                     with schema_version: 2; the Phase 1 outputs at
                                     scenes/replica_room_0/{scene_graph.json,
                                     capture_meta.json} are NEVER modified.

extractors/
  base.py                          — EXTENDED: StructuralSurface gains
                                     source: Literal["habitat_label","mesh_ransac",
                                                     "synth_bbox_fallback"]
  oracle_replica.py                — EXTENDED: read enriched importer outputs from
                                     enriched/v2/, populate bbox_obb,
                                     structural_surfaces (with provenance) on
                                     EntityArtifacts. Phase 1 read path stays.

geometry/                          — NEW package, pure functions only
  __init__.py
  obb.py                           — OBB construction, sanity checks, world-frame ops
  plane.py                         — plane construction, normalization, point/bbox distance
  surface_distance.py              — AMENDED (A4, A5): point-to-plane (signed),
                                     bbox-to-plane (NON-NEGATIVE; 0 on intersect),
                                     point-to-aabb (non-neg; 0 inside),
                                     aabb-to-aabb (Euclidean norm of positive axis
                                     gaps; 0 on overlap or touch).
                                     point-to-obb kept as a helper; OBB-to-OBB
                                     distance is DEFERRED to Phase 3.
  validators.py                    — gravity alignment, OBB sanity, plane normalization,
                                     surface extents, surface-source consistency,
                                     deterministic hashes

graph/
  builder.py                       — AMENDED (A6): retain ALL structural surfaces
                                     from EntityArtifacts (not only edge-referenced
                                     ones); reject edges whose source/target UID is
                                     not in the entity/surface tables (typed
                                     GraphBuildError).
  relations/
    proximity.py                   — EXTENDED: add sparse_v2 path using
                                     aabb_to_aabb_surface (sparse_v1 unchanged).
    surface.py                     — NEW: NEAR_SURFACE extractor (first
                                     surface-aware relation; targets StructuralSurface
                                     via GraphRef(kind="surface", uid=...)).

tests/
  geometry/                        — NEW: per-helper validators + property-style sanity
  importers/                       — NEW: enriched importer output tests (versioned path)
  graph/test_builder_surfaces.py   — NEW: builder retains all surfaces, rejects
                                     unknown-UID edges
  relations/test_near_surface.py   — NEW
  relations/test_proximity_sparse_v2.py — NEW

tools/
  verify_replica_inputs.py         — NEW: prints sha256 + sizes of raw Replica
                                     inputs; exits 0 if all present and matching
                                     pinned hashes, else 1.
  phase2_gates.py                  — NEW: per-Phase-2 artifact gates + summary
  phase2_exit_gate.py              — NEW: blocking exit assertion script
```

---

## Per-task plan (proposed)

Numbered for tracking; each task ships with its own tests and a measurable done condition.

### P2.01 — Acquire / reference raw Replica inputs

Locate `info_semantic.json` plus the semantic mesh for `room_0` (Q1 decision: user supplies locally). Document the canonical path under a new top-level `data/replica/` (gitignored, like `runs/`). Record provenance in `docs/data_inventory.md` (paths, sha256, source URL or origin).

Ship `tools/verify_replica_inputs.py`:

- prints sha256 + byte size of each expected raw input
- compares against pinned hashes (recorded once on first verified run)
- exits 0 on full match, 1 otherwise

**Done when:** raw inputs exist on local disk at a documented path AND `tools/verify_replica_inputs.py` exits 0.

**Failure mode (Q1, A1):** if the verifier fails, Phase 2 is gated. We do **not** silently fall back to the replay fixture as oracle. The labeled experimental path (Q3 option C, `source="synth_bbox_fallback"`) is allowed only as a separate, clearly tagged experiment, never to satisfy the canonical exit gates.

### P2.02 — Extend the raw Replica importer to emit world-frame OBBs

**AMENDED (A1, Q2):** The importer's enriched output goes to a **new versioned directory** `scenes/replica_room_0/enriched/v2/` with `schema_version: 2`. The Phase 1 outputs at `scenes/replica_room_0/scene_graph.json` and `scenes/replica_room_0/capture_meta.json` are NOT touched — `adapters/oracle_replica.py:58` hashes those files, so any byte-level change would silently invalidate Phase 1 bundle hashes.

Modify `importers/replica.py` to preserve `oriented_bbox.orientation.rotation` (quaternion) for every kept entity and emit, into the v2 output:

- `bbox_obb.center` (world frame, after the existing quat rotation already applied to `abb.center`)
- `bbox_obb.extents` = half of `abb.sizes` (local frame; rotation is what places them in world)
- `bbox_obb.rotation_quat` = the orientation quaternion in `(x, y, z, w)` order

The world-frame AABB is then derived from the OBB (8 corners → min/max). The Phase 1 oracle extractor's approximate AABB-from-sizes shortcut continues to read from the Phase 1 path; the new v2 read path (P2.06) consumes the tight box from the OBB, and only there is the `bbox_aabb_is_approximate` diagnostic flipped off.

**Done when:** the v2 directory contains an enriched importer output with `bbox_obb` populated for every kept entity, `schema_version: 2`, and a tight world-frame `bbox_aabb`; the Phase 1 files at the original paths are byte-identical to before the change (verified by checksum).

### P2.03 — Emit stable structural surfaces

**AMENDED (A3, A7, Q3):** Surface fitting follows a strict precedence and every surface carries provenance.

Modify `importers/replica.py` (writing to `scenes/replica_room_0/enriched/v2/`) to emit a `structural_surfaces` list containing at minimum:

- 1 floor plane (gravity-perpendicular, lowest z extent)
- 2+ wall planes (gravity-parallel, fitted to the room footprint)
- 1 ceiling plane (gravity-perpendicular, highest z extent)

Surface-fitting precedence (Q3 decision):

1. **Primary — `source="habitat_label"`:** parse `floor`, `wall`, `ceiling` instances directly from `info_semantic.json` (the categories Phase 1's `STRUCTURAL_DROP` removed from the entity set). This is the canonical Phase 2 path.
2. **Secondary — `source="mesh_ransac"`:** when a category is absent, fit the missing plane(s) from the semantic mesh via RANSAC on triangle normals filtered by gravity alignment. Acceptable for canonical Phase 2 only when the fit residual is below a recorded threshold (validators.py enforces this).
3. **Experiment-only — `source="synth_bbox_fallback"`:** synthesize from `capture_meta.room_bbox`. **NOT canonical.** `room_bbox` is derived from object centroids in `importers/replica.py:123`, not room geometry, so it is not ground truth. Emitted surfaces with this source are tagged everywhere they appear and are excluded from the canonical exit gates (G1, G2, G5, G7). Allowed only as a non-blocking experiment.

`StructuralSurface` gains a `source` field (see P2.04, P2.06, and `extractors/base.py:38` amendment).

`surface_uid` is deterministic: `floor_0`, `ceiling_0`, `wall_<n>_<orientation_tag>` where the suffix encodes the wall's outward-normal direction (e.g. `wall_n_yplus`). Each surface gets a `polygon` when extent is recoverable; `None` when only the plane is known.

**Done when:** Replica room_0 produces ≥ 1 floor, ≥ 2 walls, ≥ 1 ceiling, with every surface carrying a non-empty `source` field; surface_uids stable across re-runs of the importer; no surface with `source="synth_bbox_fallback"` appears in the canonical output unless explicitly enabled by an `--experiment` flag, in which case the v2 directory carries an `experiment.json` marker.

### P2.04 — Geometry validators

`geometry/validators.py` with explicit functions, each raising a typed error on failure:

- `validate_gravity_alignment(frame, tolerance=0.05)` — `frame.gravity` is approximately a unit vector within tolerance of one canonical axis.
- `validate_obb_sanity(obb)` — extents are positive; quaternion is unit-norm; center is finite.
- `validate_plane_normalized(plane)` — `(a, b, c)` is a unit vector.
- `validate_surface_extents(surface, room_bbox)` — surface polygon (when present) lies inside or on the room bbox.
- **AMENDED (A7):** `validate_surface_source(surface)` — `source ∈ {"habitat_label", "mesh_ransac", "synth_bbox_fallback"}`; raise if missing or unrecognized.
- **AMENDED (A7):** `validate_canonical_surface_set(surfaces)` — at least one floor, two walls, one ceiling, all with `source != "synth_bbox_fallback"`. Used by the canonical exit gates (G1). The experiment path skips this validator deliberately.
- `validate_deterministic_hash(bundle_a, bundle_b)` — two bundles from the same input produce equal bundle_hash.

**Done when:** every validator has unit tests covering the happy path plus at least one failure mode each, including a test that `validate_canonical_surface_set` rejects an all-`synth_bbox_fallback` set.

### P2.05 — Serde round-trips for enriched data

`extractors/serde.py` already round-trips OBBs and StructuralSurface; verify against enriched real data. Add tests against actual oracle output (not just synthetic fixtures) so OBB-derived AABBs survive disk write/read.

**Done when:** the enriched oracle bundle round-trips through serde with `array_aware_equal`.

### P2.06 — Update oracle extractor to surface the importer's enriched output

`extractors/oracle_replica.py` already produces `EntityArtifacts`. Phase 2 changes:

- **AMENDED (A1):** add a new constructor / config option `enriched_path: Path | None`. When set, the extractor reads enriched OBBs + structural surfaces from `scenes/replica_room_0/enriched/v2/`. When `None`, behavior is Phase-1-identical (still reads the original `scene_graph.json` / `capture_meta.json` only).
- Populate `bbox_obb` from the v2 emission (currently set to `None`).
- Replace the approximate-AABB-from-sizes derivation with the tight world-frame AABB from the OBB **only on the v2 read path**. The Phase 1 read path keeps the old AABB derivation byte-identical.
- Populate `structural_surfaces` from the v2 emission (currently `[]`).
- **AMENDED (A7):** carry `source` from each importer-emitted surface through to `StructuralSurface.source` on `EntityArtifacts`.
- Update `provides_oriented_bboxes` and `provides_structural_surfaces` capability flags to `True` **on the v2 path only**.
- Remove the `bbox_aabb_is_approximate` diagnostic **on the v2 path only**.

**Done when:** when constructed with `enriched_path=<v2 dir>`, all 73 oracle entities have `bbox_obb` populated and `len(structural_surfaces) >= 4` with every surface carrying a valid `source`; when constructed with `enriched_path=None`, the extractor's `EntityArtifacts.bundle_hash` is bit-identical to the Phase 1 value and all existing P1.04 + P1.05 tests pass unchanged.

### P2.07 — Surface-distance helpers

**AMENDED (A4, A5, Q5):** the math is tightened and the closest-corner OBB-OBB approximation is dropped. Phase 2's quality-gate distance metric is exact AABB surface distance. OBBs are still emitted as a payload (P2.02) so Phase 3 has them when support / containment detection needs SAT or GJK.

`geometry/surface_distance.py` with pure functions:

- `point_to_plane(p, plane) -> float` — **signed** scalar (negative on the plane's back side). The signed form is exposed for callers that need it (e.g. classifying which side of a wall an entity centroid lies on).
- `point_to_aabb(p, aabb) -> float` — **non-negative**; 0 when `p` is inside or on the box. Implemented as the Euclidean norm of per-axis excesses: `sqrt(sum(max(0, lo[i] - p[i], p[i] - hi[i])**2 for i))`.
- `aabb_to_aabb_surface(a, b) -> float` — **non-negative**; 0 when the boxes overlap or touch. **AMENDED:** computed as the **Euclidean norm of the positive axis gaps** between the boxes — `sqrt(sum(max(0, a.lo[i] - b.hi[i], b.lo[i] - a.hi[i])**2 for i))`. Not "min gap on the dominant axis" (which over-reports distance for diagonal separations).
- `bbox_to_plane(bbox, plane) -> float` — **non-negative**; 0 when the plane intersects (or touches) the box. **AMENDED:** computed via `max(0, min_signed_corner_distance) when all corners on the negative side, max(0, -max_signed_corner_distance) when all corners on the positive side, else 0`. Equivalently: `0` iff `min(signed_corner_distances) <= 0 <= max(signed_corner_distances)`. Not "min over the 8 corners" (which silently goes negative when the plane intersects).
- `point_to_obb(p, obb) -> float` — kept as a helper for completeness; non-negative; 0 inside.
- **DEFERRED to Phase 3:** `obb_to_obb_surface`. The closest-corner approximation misses face-to-face and intersecting-box cases and is rejected as a quality metric. When Phase 3 needs it, add SAT (boxes are convex, so SAT gives exact separation) or full GJK.

Each function has property-style tests covering:

- inside / on-boundary → 0 distance (point-to-aabb, point-to-obb, bbox-to-plane on intersection),
- separated cases match a hand-computed reference,
- symmetry where applicable (aabb-to-aabb is symmetric in its arguments),
- aabb-to-aabb diagonal separation: two unit cubes at `(0,0,0)` and `(2,2,2)` → `sqrt(3 * 1**2) ≈ 1.732`, NOT `1` (which is what min-on-dominant-axis would have given).

**Done when:** all helpers tested with the diagonal-separation regression case explicitly green; `obb_to_obb_surface` is **not** present in `geometry/surface_distance.py` for Phase 2.

### P2.08 — `sparse_v2` NEAR (proximity) using surface distance

**AMENDED (A2, A5, Q5, Q6).** Use exact AABB surface distance, expect more edges (not fewer) at the same threshold, and treat the density ratio as telemetry rather than a blocker for v2.

Extend `ProximityConfig`:

```python
@dataclass(frozen=True)
class ProximityConfig:
    mode: Literal["compat", "sparse"]
    sparse_version: Literal[1, 2] = 1     # Phase 1 default unchanged
    sparse_near_threshold: float = 1.0    # interpreted differently per version:
                                          #   v1 — centroid-to-centroid (UNCHANGED)
                                          #   v2 — AABB surface-to-surface (A5)
```

`extract_sparse` dispatches on `sparse_version`. `sparse_v1` code path is byte-frozen (Phase 1 reproduction tests must pass unchanged). `sparse_v2` is a **new** function that uses `aabb_to_aabb_surface` from `geometry/surface_distance.py`. `obb_to_obb_surface` is **not** used (deferred — see P2.07 amendment).

**Expectation correction (A2).** Surface-to-surface distance is bounded above by centroid-to-centroid distance for non-degenerate boxes (and is generally strictly smaller for any non-coincident pair). At a fixed threshold of 1.0 m, sparse_v2 should produce **more** NEAR edges than sparse_v1, not fewer. The earlier "typically smaller" claim was wrong and is retracted.

If the team prefers v2 to produce a comparable or sparser edge set, the threshold has to be re-tuned (e.g. 0.5 m) as a separate, recorded decision — NOT silently glued to 1.0 m and reported as a free win.

**Done when:**

- `sparse_v1` output is bit-identical to Phase 1 on Replica (Phase 1 compat gate still green).
- `sparse_v2` runs deterministically and emits a NEAR set whose size and per-pair contents are written to a `sparse_v2_telemetry.json` artifact (per-pair distances, threshold, edge total, density ratio).
- The density ratio for v2 is **recorded only**; the Phase 2 exit gate does NOT block on `≤ 14` for v2 (Q6).

### P2.09 — `NEAR_SURFACE` extractor

`graph/relations/surface.py` with `SurfaceProximityExtractor` implementing the same Protocol as the existing extractors but emitting `NEAR_SURFACE(entity, surface)` edges with `target=GraphRef(kind="surface", uid=...)`.

Threshold per-surface-type (Q4 decision — **provisional recorded config, not quality evidence**):

| Surface type | near_threshold (m) | Rationale (provisional) |
|---|---|---|
| floor | 0.05 | tighter — "on the floor" semantics |
| wall | 0.30 | looser — "by the wall" semantics |
| ceiling | 0.10 | medium |

These are recorded provisional defaults like Phase 1's `sparse_max_distance=2.5`. Flagged everywhere they appear (docstring, config dataclass, gate artifact) as Replica-calibrated; **not generalization evidence and not a quality claim**.

Edge semantics (A4): `NEAR_SURFACE(a, s)` iff `bbox_to_plane(a.bbox_aabb, s.plane) <= threshold_for(s.surface_type)`. `bbox_to_plane` is non-negative; intersection / touching is distance 0, so an entity sitting on the floor satisfies the predicate. Symmetric flag not set (entity → surface is directional).

**AMENDED (A3, A7):** the canonical extractor refuses to emit `NEAR_SURFACE` against surfaces whose `source == "synth_bbox_fallback"`. Synth surfaces are skipped with a recorded diagnostic. The experiment-only path opts back in via an explicit config flag.

**Done when:** every floor-mounted Replica entity (e.g. tables, sofas) has `NEAR_SURFACE(?, floor_0)`; every wall-mounted entity has `NEAR_SURFACE(?, wall_x)` for at least one wall; **no canonical `NEAR_SURFACE` edge targets a synth-fallback surface.**

### P2.10 — Phase 2 exit gates

**AMENDED (A6, A8, Q6).** Adds builder-level structural-completeness checks, requires negative controls in the smoke list, and records sparse_v2 density as telemetry rather than a block.

`tools/phase2_gates.py`: re-runs the Phase 1 compat gate verbatim AND adds Phase 2 assertions:

- **G1:** ≥ 1 floor, ≥ 2 walls, ≥ 1 ceiling emitted, **all with `source != "synth_bbox_fallback"`**.
- **G2:** Every eligible entity has a non-None `bbox_obb` on the v2 path.
- **G3:** Phase 1 compat reproduction still byte-exact (`tools/phase1_gates.py` compat artifact `pass: true`).
- **G4:** New geometry bundle deterministic across two runs; serde round-trip with `array_aware_equal`.
- **G5:** Hand-authored NEAR_SURFACE smoke list passes, **including negative controls** (entities expected NOT to be near a given surface — see A8 below).
- **G6 (telemetry only — does NOT block, per Q6):** sparse_v2 NEAR density recorded. Compares against sparse_v1 baseline. Either direction permitted; the gate logs the ratio and the absolute edge counts.
- **G7 (new — A6):** the built graph `SceneGraphBundle` retains the **full** structural-surface list from `EntityArtifacts` (not only edge-referenced UIDs), and a synthetic edge referencing an unknown entity or surface UID is rejected with a typed `GraphBuildError`. Test covers both directions of the rejection.

`tools/phase2_exit_gate.py`: blocking script that runs the full Phase 1 test sweep + Phase 2 gates G1–G5, G7 + Phase 2 test sweep + cross-stage `_internal` check. G6 is logged, not blocked.

**Smoke list (A8) — `eval/questions/phase2_near_surface_smoke.json`.** Frozen before any extractor code lands. Each case is one of:

```json
{ "entity_uid": "obj_42", "surface_uid": "floor_0", "expectation": "near"     }
{ "entity_uid": "obj_42", "surface_uid": "wall_n_yplus", "expectation": "not_near" }
```

Must include **at least 4 `near` cases and at least 4 `not_near` cases**. The not_near cases are what distinguishes a tight extractor from one that fires too freely.

**Done when:** exit gate green; G1–G5 and G7 pass; G6 telemetry recorded; all per-task tests green; both Phase 1 artifact gates still green.

---

## Config-versioning policy

The general rule for Phase 2 and beyond: any change that would alter the edge set of an existing mode bumps the mode's version, doesn't mutate it.

- `ProximityConfig.sparse_version: Literal[1, 2] = 1` (default unchanged).
- `DirectionalConfig` does not change in Phase 2 (no Phase 2 work touches directional logic).
- Existing call sites that pass `ProximityConfig(mode="sparse")` continue to get v1 behavior; explicit opt-in via `ProximityConfig(mode="sparse", sparse_version=2)`.
- `tools/phase1_exit_gate.py` continues to call v1; a new `tools/phase2_exit_gate.py` opts into v2 for the sparse density gate.

This is the same pattern we'll use in Phase 3 when support / containment / attached land (likely `sparse_version=3`).

---

## Pre-registered exit gates (amended)

Frozen before coding so we can't goalpost-shift. **AMENDED:** G5 now requires negative controls (A8); G6 is added for sparse_v2 density as telemetry-only (Q6); G7 is added for builder structural-completeness checks (A6).

| # | Gate | Pass condition | Block / telemetry |
|---|---|---|---|
| G1 | Structural surfaces emitted | ≥ 1 floor, ≥ 2 walls, ≥ 1 ceiling on Replica room_0, **all with `source != "synth_bbox_fallback"`** | BLOCK |
| G2 | World-frame OBBs | Every eligible oracle entity (73 on Replica) has non-None `bbox_obb` on the v2 read path | BLOCK |
| G3 | Phase 1 compat reproduction | `tools/phase1_gates.py` compat artifact still `pass: true`, `missing=[]`, `extra=[]`, 5414/5414 | BLOCK |
| G4 | New geometry bundle deterministic + replayable | Two runs produce identical `EntityArtifacts.bundle_hash` AND round-trip through serde with `array_aware_equal` | BLOCK |
| G5 | NEAR_SURFACE smoke (with negative controls) | Hand-authored list at `eval/questions/phase2_near_surface_smoke.json`, frozen before extractor code: **≥ 4 `near` AND ≥ 4 `not_near` cases**, all match | BLOCK |
| G6 | sparse_v2 density telemetry | Record `actual_ratio = logical_v2 / entity_count` to the gate artifact; compare to sparse_v1 baseline; **does not block on `≤ 14`** | TELEMETRY |
| G7 | Builder structural-completeness | `SceneGraphBundle` retains every structural surface from `EntityArtifacts` (not just edge-referenced UIDs); edges referencing unknown entity/surface UIDs raise `GraphBuildError`. Test covers both rejection directions. | BLOCK |

The G5 smoke list lives at `eval/questions/phase2_near_surface_smoke.json` and is frozen **before** any extractor code lands (the negative controls are what give the gate teeth — see A8).

---

## Retirement / preservation map (amended)

| Asset | Phase 2 disposition |
|---|---|
| `importers/replica.py` | EXTENDED, not replaced. **AMENDED (A1):** the Phase 1 outputs at `scenes/replica_room_0/{scene_graph.json, capture_meta.json}` are byte-frozen — they are hashed by `adapters/oracle_replica.py:58`. The enriched output is written to a NEW versioned path `scenes/replica_room_0/enriched/v2/` with `schema_version: 2`. |
| `scenes/replica_room_0/scene_graph.json`, `capture_meta.json` | UNCHANGED. Any modification would silently change Phase 1 bundle hashes. |
| `scenes/replica_room_0/enriched/v2/` | NEW directory; Phase 2 canonical enriched output lives here. |
| `extractors/base.py` | EXTENDED: `StructuralSurface` gains a `source` field (A7). |
| `extractors/oracle_replica.py` | Updated to read from the v2 enriched path **when configured to**; default constructor keeps Phase 1 read path bit-identical. Capability flags flip to `True` only on the v2 read path. |
| `graph/builder.py` | EXTENDED (A6): retain all structural surfaces from `EntityArtifacts`; reject edges referencing unknown entity/surface UIDs. |
| Phase 1 sparse defaults | UNCHANGED. `sparse_v1` is frozen. |
| `graph/relations/proximity.py` `extract_sparse` | Phase 1 v1 function untouched; new `extract_sparse_v2` function added; dispatch on `sparse_version`. |
| `geometry/surface_distance.py` | NEW. Includes point-to-plane, point-to-aabb, aabb-to-aabb (Euclidean norm of positive axis gaps), bbox-to-plane (non-negative). **Does NOT include `obb_to_obb_surface`** in Phase 2 — that's Phase 3. |
| `tools/phase1_gates.py`, `tools/phase1_exit_gate.py` | UNCHANGED. They continue to assert Phase 1 properties. |
| `scenes/replica_room_0/eval/*` Phase 1 artifacts | Untouched; still the canonical Phase 1 record. Phase 2 emits NEW artifacts to `scenes/replica_room_0/eval/phase2_*.json` (including `sparse_v2_telemetry.json`). |
| Bathroom v1 wrapper | UNCHANGED. The hand-authored graph never had OBBs or surfaces; not in scope. |

---

## Open questions — RESOLVED

All six open questions were answered in the 2026-05-31 review. Repeating the calls inline so this section is self-contained.

| # | Question | Decision |
|---|---|---|
| Q1 | How do we get raw Replica `info_semantic.json` + mesh? | **User supplies locally.** Phase 2 ships `tools/verify_replica_inputs.py` (sha256 + size + exit code). No silent canonical fallback; verifier failure gates Phase 2. |
| Q2 | Does the importer's emitted JSON get a `schema_version`? | **Yes — `schema_version: 2`**, emitted to the new versioned path `scenes/replica_room_0/enriched/v2/`. |
| Q3 | Surface-fitting strategy | **Habitat structural labels first, mesh RANSAC second.** Bbox synthesis (`source="synth_bbox_fallback"`) is allowed only as a clearly labeled non-blocking experiment; the canonical exit gates exclude it. |
| Q4 | NEAR_SURFACE thresholds | **Accepted as provisional recorded config values (0.05 / 0.30 / 0.10 m), not quality evidence.** Same status as Phase 1's `sparse_max_distance=2.5` — Replica-calibrated, not generalization evidence. |
| Q5 | OBB-to-OBB surface distance approximation | **Use exact AABB surface distance for Phase 2** (Euclidean norm of positive axis gaps). Robust OBB-to-OBB distance (SAT/GJK) is deferred to Phase 3 only if support detection needs it. The closest-corner approximation is rejected as a quality metric. |
| Q6 | Sparse-v2 density check at the Phase 2 exit gate | **Record as telemetry; do not block on `≤ 14` for v2.** The blocking `≤ 14` rule stays attached to frozen sparse_v1 only. G6 logs `actual_ratio` and absolute counts without failing. |

---

## Review checklist — status

| # | Item | Status |
|---|---|---|
| 1 | Scope correct? Geometry enrichment only; no support / containment / attached relations; no learned backends. | CONFIRMED |
| 2 | Task list complete (P2.01–P2.10)? | CONFIRMED with amendments A1–A8 threaded in |
| 3 | Module layout — new `geometry/` package; new `graph/relations/surface.py`; importer/extractor extensions are additive; enriched output to versioned path | CONFIRMED (A1, A7) |
| 4 | Config-versioning policy — `sparse_v1` byte-frozen, `sparse_v2` opt-in | CONFIRMED |
| 5 | Pre-registered exit gates — now G1–G7, with G6 telemetry-only and G5 requiring negative controls | CONFIRMED (A6, A8, Q6) |
| 6 | Open questions Q1–Q6 | RESOLVED — see "Open questions — RESOLVED" |
| 7 | Retirement / preservation — Phase 1 file paths and bundle hashes byte-frozen | CONFIRMED (A1) |
| 8 | Out-of-scope reminders — ON_TOP_OF, ATTACHED_TO, INSIDE, CONTAINS, SUPPORTS all Phase 3; learned backends Phase 4 | CONFIRMED |

Phase 2 starts with **P2.01 (data acquisition)**. If `tools/verify_replica_inputs.py` fails (raw Replica data missing), Phase 2 pauses; the labeled-fallback experiment (Q3 option C, `source="synth_bbox_fallback"`) is available as a sidecar but never satisfies the canonical exit gates.

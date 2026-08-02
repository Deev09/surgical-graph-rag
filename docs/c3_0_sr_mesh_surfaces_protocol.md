# C3.0-SR — raw-mesh structural surfaces, corrected input contract

**Status: PREDECLARED 2026-08-02, owner-instructed** ("The next valid
experiment is C3.0-SR: predeclare one deterministic quad-to-triangle rule,
add a real Replica format fixture, refreeze, then rerun room_2"). Written
after C3.0-S closed as a Stage-0 preparation-contract negative result
(triangles-only parser vs an all-quads pinned input; verdict `9908fec`)
and before any SR estimator run on real data.

## Inheritance

Everything not listed under "Deltas" is inherited VERBATIM from
`docs/c3_0_mesh_surfaces_protocol.md`: the question it answers, scene
roles (room_2 development; room_1/office_0/room_0 prospective transfer),
the isolation boundary, the fixed `mesh_region_fit_v1` estimator and all
its constants, metric definitions, the `B+S_mesh` downstream row, all
development gates G1–G7 and transfer gates H1–H6, the budget (one dev run
+ three transfer runs, zero GPU, no sweeps, no rescue), and the stopping
rules. No gate, threshold, or constant changes. C3.0-S's frozen anchors
(room_2 B: surface-question micro-P 0.9375, micro-R 0.5172) are unchanged.

## Deltas (exactly three)

1. **Quad-to-triangle rule (the one predeclared parser change).** The raw
   PLY loader accepts face arity 3 and 4 only. A quad `(v0,v1,v2,v3)` in
   stored index order is split on the fixed v0–v2 diagonal into
   `(v0,v1,v2)` and `(v0,v2,v3)`, emitted adjacently in face order. Any
   other arity is a hard failure. There is no planarity- or
   shape-adaptive diagonal choice. Precedent: this is the SAME rule this
   repository already froze for Replica quads in the C1-M2 Segmentator
   triangulation (commits `282cf53`/`b3c7332`) and in the C1 notebook's
   triangulated demo mesh — it is a citation, not a new design decision.
   The loader records `n_source_quads` in the estimator diagnostics and
   the artifact telemetry.
2. **Real-format Stage-0 fixtures.** New synthetic fixtures that match
   the ACTUAL Replica raw layout (binary_little_endian, uint8 face-count,
   int32 indices, quad faces): (a) the existing synthetic room written as
   NATIVE QUADS must parse to faces byte-identical to the triangle
   fixture (the room builder already splits quads by the same v0–v2
   rule, so equivalence is exact) and must estimate identical surfaces;
   (b) a mixed arity-3/4 fixture parses; (c) an arity-5 fixture
   hard-fails; (d) a dataset-guarded probe asserts the PINNED room_2
   `mesh.ply` (locked sha `e58a7c71…`) parses to exactly 722,398 quads →
   1,444,796 triangles with zero trailing bytes.
3. **Refreeze.** The corrected loader, fixtures, and this protocol are
   committed as the SR preparation freeze BEFORE the single authorized
   room_2 development run. The C3.0-S artifacts and verdict remain
   untouched as the negative-result record.

## What SR does NOT change

The estimator algorithm, its parameters, gate numbers, scorer, keys,
graph configs, thresholds, question wording, and every previously frozen
artifact. If room_2 fails any G gate: STOP, record, transfers unspent —
identical to C3.0-S's rule. A pass proceeds to the three frozen transfer
scenes under H1–H6, one run each, no changes between scenes.

## Sign-off

- [x] Owner instruction to predeclare and run C3.0-SR (2026-08-02,
      project owner / deevyaswain, quoted above).
- [x] SR preparation freeze commit: `b05931a` (quad-aware parser +
      real-format fixtures; frozen C3.0-S Stage-0 tests unchanged and
      passing; 67/67 canonical test files green; committed BEFORE the
      single authorized room_2 development run)

## 2026-08-02 verdict — STOPPED: all seven development gates FAIL

The single authorized room_2 run executed from `5c03aa0`. Unlike C3.0-S,
the input contract held: the pinned quad mesh parsed (722,398 quads →
1,444,796 triangles, 414 degenerate dropped), the estimator ran to
completion, the hash-stamped surface artifact was finalized
(`b5b23ee4b61f4a52…`), and the evaluator then opened oracle surfaces and
the human key. This is a REAL geometry negative result, not a
preparation failure.

Result: **0 floors, 0 ceilings, 1 wall** (`invalid_reason:
fewer_than_two_qualifying_horizontal_components`). G1–G7 all FAIL:

| measure | value | gate |
|---|---|---|
| floor / ceiling emitted | 0 / 0 | G1 FAIL |
| wall oracle area coverage | 0.066 (1 of 4 walls, 19% of it) | G3 FAIL |
| compatible-plane errors (the one survivor) | 0.31°, 0.033 m | G4 FAIL (offset > 0.03) |
| surface-question micro-P / micro-R vs key | null (0 cited) / 0.00 | G5/G6 FAIL (B: 0.9375 / 0.5172) |
| NEAR_SURFACE(wall) F1 vs B | 0.00 | G7 FAIL |

Failure class, read from the frozen artifact diagnostics only:
**component-acceptance collapse on real mesh roughness.** Candidate
faces were abundant (611,972 horizontal; 707,342 vertical) and region
growth produced 15 components over the area floor — but 14 of 15 were
rejected at the PCA-fit / boundary-loop stage (`n_rejected_fit_or_
boundary: 14`, `n_fit_components: 1`). The constants that did this
(0.015 m RMS residual, 0.02 m growth band, simple-boundary requirement)
were fixed against synthetic planar fixtures and never measured against
real captured-surface statistics; real Replica floors/walls carry
carpet/texture/curvature that a clean plane fit rejects wholesale. The
one component that survived fit accurately (0.31°) — the estimator is
precise on what it accepts and accepts almost nothing.

Per the inherited dev-failure stopping rule: **C3.0-SR is closed as a
negative result.** Transfer runs remain unspent (3), no constant was
altered, no rescue run occurred. The lesson for any successor protocol
is methodological: BOTH C3.0 failures came from freezing contact with
reality too late — first the input format, then the acceptance
constants. A successor must include a predeclared, measurement-first
Stage 0m that MEASURES real-mesh statistics (face-arity census,
per-component point-to-plane residual distributions on the pinned
meshes) read-only BEFORE any constant is fixed, exactly as the
benchmark side of this project has always measured before gating.
Whether to open that successor is an owner decision; nothing is
authorized by this verdict.

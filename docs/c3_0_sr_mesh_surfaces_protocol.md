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
- [ ] SR preparation freeze commit: __________ (filled at refreeze,
      before the room_2 run)

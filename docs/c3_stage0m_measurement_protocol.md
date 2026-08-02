# C3 Stage 0m — read-only real-mesh measurement (DRAFT for approval)

**Status: DRAFT — awaiting owner sign-off. Nothing has been measured.
This protocol predeclares a MEASUREMENT, not an estimator: no surface
artifact is produced, no gate is evaluated, no constant is frozen, and no
successor experiment is authorized by running it.**

Written 2026-08-02, after C3.0-S (input-contract failure) and C3.0-SR
(acceptance-constant collapse: 14/15 grown components rejected on real
mesh roughness). Both failures froze contact with reality too late; this
stage measures reality first. Owner decision rule, quoted:

> "If real floor/wall components separate cleanly from clutter by
> residual/boundary statistics, draft a successor C3 protocol. If they
> overlap heavily, stop this estimator family and pivot to C1-P1."

## What is measured (and on which scenes)

**Labeled measurements: room_2 ONLY** (the established development
scene). Oracle surfaces and object labels are read freely — this is a
characterization study, declared as such; its numbers may inform a
successor's constants. To preserve prospectivity, **room_1, office_0,
and room_0 receive NO labeled measurement** — they get only the
label-free census (M1) and remain clean transfer scenes for any
successor.

- **M1 — label-free census, all four pinned meshes:** face-arity counts,
  degenerate-face counts, vertex/triangle totals, global per-face normal
  statistics. (Read-only; no oracle data.)
- **M2 — true-surface cohesion curves, room_2:** for each of the six
  oracle structural surfaces, and for each residual band
  `b ∈ {0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05} m`: the slab set =
  faces whose three vertices lie within `b` of the oracle plane and
  project inside the oracle polygon; report the point-to-plane residual
  distribution (p50/p90/p95/p99, RMS), the normal-deviation
  distribution, the number of connected components of the slab under the
  frozen mesh-edge adjacency, and the oracle-area coverage of the
  LARGEST component. A curve over `b` is a measurement; choosing a point
  on it would be an estimator constant and is out of scope here.
- **M3 — planar-impostor census, room_2:** at each band `b`, connected
  planar components with mesh area ≥ 1.5 m² (the frozen family's area
  gate, cited as the relevance threshold, not adopted as a new constant)
  that are NOT within 0.05 m of any oracle structural plane; for each,
  its oracle object attribution (majority face labels — e.g. "table",
  "cabinet"), orientation family, and whether it lies inside the frozen
  boundary-percentile band. This answers "what could be confused for a
  wall/floor at that band."
- **M4 — boundary-loop feasibility, room_2:** for each oracle surface's
  largest component at each band, run the FROZEN boundary-loop extractor
  (unchanged code, read-only reuse) and record success/failure — the
  second rejection mechanism in C3.0-SR, measured directly.

Measurement parameters above (the band grid, the 0.05 m attribution
slab, majority-label attribution) are measurement choices, declared
here; they are not estimator constants.

## Predeclared decision rule (applies the owner's rule with numbers)

Coverage requirements are inherited verbatim from the frozen C3.0
geometry gates (G2/G3): floor ≥ 0.85, ceiling ≥ 0.85, each wall ≥ 0.75.

- **CLEAN → draft a successor C3 protocol** iff there exists a band
  `b ≤ 0.05 m` where (i) every one of the six oracle surfaces meets its
  coverage requirement via its largest component, (ii) that component's
  frozen boundary-loop extraction succeeds (M4), and (iii) the M3
  impostor census at `b` is zero.
- **OVERLAP → stop the estimator family, pivot to C1-P1** iff no band
  satisfies (i) — real surfaces do not cohere at any tolerance before
  clutter-scale bands.
- **MIXED → owner review** iff some band satisfies (i)+(ii) but (iii)
  fails: the report must name each impostor (object, area, position),
  and the default is STOP unless every impostor is demonstrably
  handled by a rejection mechanism the frozen family already possesses
  (e.g. the boundary-percentile gate). No new mechanism may be invented
  to rescue a MIXED verdict.

## Budget and isolation

Zero GPU. One measurement run; deterministic; report JSON under
`runs/phase8_c3/stage0m/` plus a verdict section appended here. No
surface artifact, no downstream graph/Router run, no key is opened (the
human key plays no role in geometry characterization). No estimator
code is modified; the frozen extractor is imported read-only for M4.
Implementation: `tools/c3_stage0m_measure.py` + synthetic tests for the
slab/component/attribution logic.

## What running this does NOT authorize

No successor estimator, no constant selection, no transfer-scene
labeled measurement, no C1-P1 start. The output is a verdict under the
decision rule; acting on it is a separate, owner-approved step either
way.

## Sign-off

- [ ] Owner approves the measurement set, scene scoping (labeled =
      room_2 only), band grid, and decision rule
      (date: ______, by: ______)

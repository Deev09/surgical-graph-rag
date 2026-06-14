---
title: Phase 6 Closeout — EntitySurface support
status: complete
date: 2026-06-14
tags: [phase-6, closeout, entity-surface, on-entity-surface, support, reasoner]
---

# Phase 6 Closeout — EntitySurface support

Phase 6 adds furniture-top support as a reasoner-native relation:

- stored edge: `ON_ENTITY_SURFACE(supported_entity, supporter_entity)`
- derived view: `SUPPORTS(EntityClassRef("table" | "chair" | ...), ?x)`
- QA behavior: `"what is on the table?"` now answers; `"what is on the chair?"`
  returns true empty; `"what is attached to the wall?"` still defers.

This is **not** a v1 benchmark improvement claim. It is a new reasoner-native
capability and a Phase 6 eval-definition change: Q3 changed from defer to answer,
and Q7 is a new empty row. The P5 and P6 scorecards are not directly comparable.

## What shipped

- `extractors/entity_surfaces.py`
  - derives AABB top faces from support-class entity boxes;
  - keeps derived tops out of `EntityArtifacts.structural_surfaces`;
  - normalizes class labels backend-agnostically (`table_5`, `plant-stand_1`,
    etc.).
- `graph/relations/on_entity_surface.py`
  - emits `ON_ENTITY_SURFACE` entity→entity edges;
  - stores derived top-surface provenance in evidence;
  - reuses the floor rest band: `contact_threshold_m=0.02`,
    `penetration_tolerance_m=0.03`.
- `graph/views/support.py`
  - keeps floor support via `support_facts`;
  - adds entity support via `entity_support_facts`.
- Reasoner
  - adds `EntityClassRef`;
  - compiles table/chair/desk/stool/bench/shelf/sofa/plant-stand/counter support
    queries to `SUPPORTS(EntityClassRef(...), ?x)`;
  - preserves unsupported-class deferral.
- Eval artifacts
  - `eval/questions/phase6_entity_surface_smoke.json`
  - `eval/questions/phase6_mixed_qa.json`
  - `scenes/replica_room_0/eval/phase6_router_qa_eval.json`
  - `scenes/replica_room_0/eval/phase6_exit_gate_report.json`

## Measured behavior on Replica room_0

`ON_ENTITY_SURFACE` emits six real positives:

- table rests: `obj_92`, `obj_90`, `obj_12`, `obj_59`, `obj_87`
- plant-stand rest: `obj_35` on `obj_55`

Important excluded cases:

- `obj_43` pot on table is excluded by the frozen band (`+0.0349 m` float).
- `obj_55` plant-stand on table is excluded because support-furniture classes are
  not supported tabletop answers in P6.
- `ATTACHED_TO` remains deferred: room_0 has no honest wall-contact-and-elevated
  positive under the Phase 5 evidence.

## Exit gate

`python3 tools/phase6_router_qa_eval.py`

- total questions: 7
- category counts: `{"correct_defer": 1, "true_answer": 5, "true_empty": 1}`
- false answers: 0
- all expected outcomes met: true

`python3 tools/phase6_exit_gate.py`

- all Phase 6 gates pass;
- schema is now graph serde v5;
- v4 graph manifests are rejected by the strict loader;
- default Phase 2 build still emits 0 `ON_SURFACE`, 0 `CONTACTS_SURFACE`, and
  0 `ON_ENTITY_SURFACE` edges.

## Interpretation

This is enough to start plug-and-chug testing with multiple reconstruction
backends, but it is still single-scene evidence. The next useful experiment is
not to tune the band on room_0; it is to run the same fixtures and gates on new
backend outputs and record where the relation fails:

- missing or noisy support-class boxes;
- box top too high/low for the frozen contact band;
- class-label mismatch;
- chairs/tables detected as separate parts instead of owner entities;
- containment cases that should become future `IN` / `CONTAINED_BY`, not `ON`.

---
title: Phase 4 closeout — ON_SURFACE rest-contact + derived SUPPORTS
status: closed
date: 2026-06-06
tags: [phase-4, closeout, on-surface, supports]
---

# Phase 4 closeout

> [!info] One-page interpretation freeze, written at close while the evidence is fresh.
> Plan: [[phase4_plan]]. Substrate: [[phase3_plan]] (polygon-clipped NEAR_SURFACE).

## What shipped

- **`ON_SURFACE(entity, surface)`** — gravity-supported rest-contact (Design B): support-capable up-facing surface, centroid on the support side, bottom-gap within the contact band, AABB footprint clipping the surface polygon. Pure-geometry predicate (`geometry/rest_contact.py`), isolated extractor (`graph/relations/on_surface.py`), new `ON_SURFACE` EdgeType, graph serde bumped v2→v3.
- **`SUPPORTS`** — derived read-side view only (`graph/views/support.py`). Inverts each `ON_SURFACE` edge to a `SupportFact(supporter=surface, supported=entity)`. Never stored; strict guards raise on a materialized `SUPPORTS` edge or a malformed `ON_SURFACE` endpoint.
- **Floor QA** — "what is on the floor?" answered through the normal compiler → executor → verbalizer path, anchored on `SurfaceRef("floor")`, citing the stored `ON_SURFACE` edges.
- **Deferred (explicit, not faked)** — "on the table / chair" and "against the wall" compile to `out_of_schema` with `deferred:` notes; the verbalizer says so plainly.

## What the numbers mean

On **Replica room_0**, default `OnSurfaceConfig`:

| metric | value |
|---|---|
| ON_SURFACE edges (floor) | **11** |
| support facts | **11** (clean inverse) |
| materialized SUPPORTS edges | **0** |
| wall / ceiling ON_SURFACE edges | 0 (not support-capable, by design) |

These say: 11 entities rest on the floor under the calibrated rest-contact predicate, and the support view is exactly that set viewed from the surface side.

## What they do NOT mean

- **Not a v1 benchmark improvement.** This is substrate/faithfulness work; the 10-query benchmark was not run or claimed.
- **Not table/chair support.** No furniture-top geometry exists in Replica room_0; those QA items are deferred, not answered as empty.
- **Not wall attachment.** Wall/ceiling contact is a different relation (`ATTACHED_TO` / `HANGS_FROM`), deferred to a later phase.
- **Not learned-backend readiness.** No learned components; rest-contact is a calibrated geometric predicate. `support_facts == on_surface_edges` is true by construction, not independent corroboration.
- **`penetration_tolerance_m = 0.03`** absorbs a measured Replica floor-plane fit bias; it is not a physical 3 cm penetration claim and should be revisited if the importer refits the floor.

## Validation

- **P4.06 exit gate: 9/9 blocking gates pass** — determinism (G1), subset ⊆ polygon-mode NEAR_SURFACE (G2), clean inverse (G3), zero materialized SUPPORTS (G4), smoke fixture + real F1 stool (G5), default-path preserved with ON_SURFACE absent from any default builder (G6), prior artifacts byte-untouched (G7), threshold guard enforced (G8), and graph serde v3 round-trip + v2 strict rejection.
- **479/479 tests across 31 suites.**
- **Prior artifacts untouched** — Phase 1/2/3 reports, telemetry, and eval tables byte-unchanged; ON_SURFACE wired into no default builder run.
- **Schema round-trip defect found and fixed** — the exit gate's schema check caught that `ON_SURFACE` edge evidence carried `up` as a tuple (round-tripping to a list, breaking `dump==load`); fixed at the evidence-assembly boundary (commit `085b8ef`, separate from the gate).

## Next

Phase 5 candidates, each honestly named and each needing its own geometry/data: `CONTACTS_SURFACE` / `ATTACHED_TO` (wall), `HANGS_FROM` (ceiling), `EntitySurface` (tabletop/seat) to unlock "what's on the table/chair?", and `IN` / `CONTAINED_BY`. None are started.

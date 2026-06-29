# Phase 7 plan — `ATTACHED_TO` (wall-mounted objects) as a gated view

Status: frozen scope (path A). Implement in small validated steps; the geometry
layer is validated in isolation before the shared compiler is touched.

## Decisions

**D1 — Relation.** `ATTACHED_TO(object → wall surface)`. Stored edge, same
entity→surface family as `CONTACTS_SURFACE`. NOT a derived view (the executor
reads the edge directly; no inversion needed, unlike P6's entity→entity support).

**D2 — Gate (no new thresholds).** Emit `ATTACHED_TO(o, w)` iff:
  - `wall_contact(o, w)` is true (reuse `CONTACTS_SURFACE` geometry + bands), AND
  - `o` does **not** `rest_contact` **any floor** surface (reuse `ON_SURFACE`
    geometry + bands).
All thresholds are inherited from the two existing predicates; Phase 7 invents
none. "Elevated" is emergent (= not floor-resting), recorded as evidence, not a
gate constant.

**D3 — Self-contained.** The `AttachedToExtractor` runs both predicates itself
(it has the floor and wall surfaces + the entity box). No dependency on
`ON_SURFACE`/`CONTACTS_SURFACE` having run first — order-independent.

**D4 — Evidence.** Each edge carries the full `wall_contact` evidence plus
`floor_supported: false`, `floor_rest_best_bottom_gap_m` (the closest floor
rest-gap that still failed), and `bottom_elevation_m` (bbox bottom above the
nearest floor plane). Invariant: every emitted edge has wall-contact true and
floor-rest false.

**D5 — Committed scene has zero positives (path A).** room_0 yields 0
`ATTACHED_TO` edges (the picture is elevated-but-not-wall-contacting; the lamp is
wall-contacting-but-floor-resting). Real positives exist only on apartment_0
(uncommitted, no answer key). Therefore:
  - room_0 exit gate asserts `ATTACHED_TO` = **empty** (honest negative).
  - real positives are proven by a **synthetic frozen smoke fixture**.
  - apartment_0's **3** validated plausibility positives — vent (2.43 m), wall
    sconce-lamp (1.89 m), wall sink (0.62 m) — stay in `demo/`, non-ground-truth.
    (A naive single-floor probe also flagged a curtain; the multi-floor-aware
    extractor correctly excludes it as upper-story floor-level, not high-wall.)

**D6 — P6 byte-freeze.** The committed P6 exit gate asserts
`attached.outcome == "abstain"` (`tools/phase6_exit_gate.py:259-260`). Advancing
the shared compiler to answer "attached" would break it. So a
`Phase6RulesCompiler` freeze (mirroring `Phase5RulesCompiler`) lands in the SAME
change, keeping "attached → defer" for P6 artifacts. Q6 defer→answer is a Phase 7
eval-definition change, isolated to `phase7_mixed_qa.json`.

## Known limitations (state up front)
- **Pictures missed (recall):** flush-mounted pictures are elevated but their AABB
  sits just off the wall plane → no wall contact → not ATTACHED. Disclosed, not
  discovered later.
- **Furniture-rest edge case:** v1 disqualifies floor-rest only. A tabletop object
  pushed flush against a wall could read as attached. Rare; documented; a v2 could
  add furniture-top rest disqualification (reusing P6 entity surfaces).
- **Floor-reaching wall objects missed (recall):** the "not floor-supported" gate
  excludes objects that reach floor level even if wall-mounted (a floor-length
  curtain, a tall mirror). Consistent with the gate definition; disclosed.
- **No ground truth for positives** — synthetic + plausibility only (D5).

## Insertion points (confirmed by code-map)
| Step | File | Change |
|---|---|---|
| 1 | `graph/relations/attached_to.py` (new) | `AttachedToExtractor` + `AttachedToConfig` |
| 2 | `graph/serde.py:23` | `CURRENT_SCHEMA_VERSION = 5 → 6` (ATTACHED_TO already in `EdgeType`) |
| 3 | `reasoner/compiler_rules.py:~153` | "attached to (the) wall" → `ATTACHED_TO(?x, SurfaceRef("wall"))` |
| 4 | `reasoner/executor.py:175` | add `"ATTACHED_TO"` to `_SURFACE_RELATION_TYPES` |
| 5 | `reasoner/compiler_rules.py` | `Phase6RulesCompiler` freeze ("attached"→defer) |
| 6 | `eval/questions/phase7_*.json` (new) | smoke fixture + mixed QA (Q6 defer→answer) |
| 7 | `tools/phase7_exit_gate.py` (new) | G1–G9 gate + demo command recording |

## Exit-gate checks (Phase 7)
G1 schema v6 round-trips ATTACHED_TO and rejects v5 · G2 synthetic attachment
smoke fixture passes (mounted positive, floor-standing negative, no-wall-contact
negative) · G3 room_0 ATTACHED_TO is an honest empty · G4 default compiler answers
while `Phase6RulesCompiler` still defers · G5 executor returns a synthetic
ATTACHED_TO binding · G6 committed P6 exit gate still passes · G7 ATTACHED_TO is
absent from the default build · G8 ATTACHED_TO build is deterministic ·
G9 apartment_0 demo/plausibility QA passes with `obj_176`, `obj_260`, `obj_309`.

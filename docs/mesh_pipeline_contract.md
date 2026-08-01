# Mesh-input evaluation ladder (A/B/C) and the frozen front-end contract

Status: A↔B repaired and frozen 2026-07-12. C1 measured on all four Phase 8
scenes with backend 1 (Mask3D class-agnostic mask stage via OpenMask3D)
2026-07-13 — see docs/c1_closeout.md. MIN_SCORE frozen at 0.2
(benchmark-definition choice; sweep-backed). Verdict: infrastructure
validated and reproducible; backend entity coverage NOT sufficient for
reliable raw-PLY question answering (answer recall vs B 0.39–0.51; dominant
failures: merging 42%, no viable proposal 28%).

## The ladder

Every variant feeds the SAME downstream path —
`build_graph(arts, demo.question_battery._runs(), density_policy="phase2_telemetry_only")`
→ Router — so each adjacent comparison attributes changes to exactly one
upstream stage.

| variant | instances/boxes | labels | structural surfaces | isolates |
|---|---|---|---|---|
| **A** JSON oracle | `info_semantic.json` boxes | oracle | oracle (JSON) | baseline |
| **B** semantic-mesh geometry | vertex bounds from `habitat/mesh_semantic.ply` per-face `object_id` | oracle | **inherited verbatim from A** | box source |
| **C1** raw-mesh segmentation | learned segmenter on `mesh.ply` | oracle via correspondence | oracle (from A) | instance extraction |
| **C2** raw-mesh semantics | same as C1 | learned | oracle (from A) | semantic classification |
| **C3** fully raw | learned | learned | mesh-derived | complete deployable pipeline |

Implementations: A = `demo/replica_habitat_import.py::import_habitat_room`,
B = `demo/replica_mesh_import.py::import_mesh_room`. C1 is implemented and
measured (segmenter package + `tools/c1_*` evaluators; Mask3D reference
backend frozen at MIN_SCORE=0.2, see `docs/c1_closeout.md`; the Segment3D
pilot failed its predeclared gate, see `docs/c1_m2_protocol.md`). C2 and C3
are not implemented.

## A↔B frame parity (what "repaired and frozen" means)

- Both importers obtain their rotation from ONE function:
  `replica_habitat_import._aligned_structural_surfaces` (gravity alignment +
  F1/F3 guarded yaw de-rotation, full-precision yaw).
- B takes its structural surfaces VERBATIM from A (including F2 floor
  calibration, whose voters are A's JSON boxes) — B never recalibrates from
  its own boxes, or its floors would drift from A's.
- B raises `AssertionError` at import time if its yaw disagrees with A's
  (frame drift is a bug, never a silent difference).
- Regression: `tests/importers/test_mesh_frame_parity.py` (per scene:
  identical yaw, identical floor calibration, byte-identical surfaces,
  identical entity uid sets, and boxes provably from the mesh).
- Data pinned: `tools/replica_scenes.lock.json` pins sha256+size of all
  three files per Phase 8 scene (`info_semantic.json`, `mesh_semantic.ply`,
  `mesh.ply`); `python3 tools/fetch_replica_scenes.py` verifies.

## Frozen front-end contract (what C1's segmenter must emit)

Two-stage isolation — the segmenter NEVER touches oracle files:

    raw mesh.ply
      -> MeshSegmenter (segmenter/base.py Protocol; local or remote GPU)
      -> immutable SegmentationOutput bundle:
           vertex_instance_ids.npy   int64 [n_vertices], -1 = unassigned
           instance_table.json       instance_id, n_vertices, confidence
           meta.json                 input mesh sha256, name/version/config,
                                     runtime + hardware, deterministic
                                     output_sha256 (assignment + table)
      -> ANONYMOUS candidate EntityArtifacts (segmenter/candidate.py):
           display_label="segment_<id>", semantic_hypotheses=[],
           structural_surfaces=[], candidate AABB + PCA-yaw OBB from the
           assigned vertices, geometry_handle=
           "<bundle>/vertex_instance_ids.npy#<id>" (dense membership sidecar
           — boxes alone cannot support exact per-vertex scoring),
           notes prove semantic_source="none" / surface_source="none";
           frame (rotation + z_translation) is CALLER-supplied, never read
           from info_semantic.json inside this path.
           Hard failures: input-hash mismatch, assignment-length mismatch,
           out-of-range ids, non-finite vertices, empty retained set.

Then a separate EVALUATION-ONLY step (C1.03, not yet implemented) derives
the C1 evaluation bundle: anonymous candidate + held-out mesh_semantic.ply /
info_semantic.json -> exact vertex-index correspondence -> oracle labels and
A's surfaces injected, with provenance `oracle_correspondence`, unmatched
predictions kept anonymous (never silently dropped), and a bundle hash
distinct from the anonymous candidate. The deployable anonymous candidate is
scored with `CompletenessProfile(source="unknown")`; reports on the enriched
bundle must state that labels and surfaces were injected for isolation.

EntityArtifacts requirements common to C-variants:

- `entities[*].bbox_aabb` + `centroid` in the shared canonical frame (the
  rotation from `_aligned_structural_surfaces` + scene `z_translation`).
- stable `obj_<n>` uids with `source_instance_ref="segmenter:<id>"`.
- C2 puts learned labels in `display_label` / `semantic_hypotheses`;
  C3 must additionally derive its own structural surfaces from the mesh.
- `notes` record `frame_source`, segmenter name/version/params, and (on
  enriched bundles) every oracle injection.

### Oracle correspondence for C1 is EXACT, not fuzzy

Measured 2026-07-12: `mesh.ply` and `habitat/mesh_semantic.ply` contain
**identical vertex arrays, in identical index order** (verified on all four
Phase 8 scenes: room_1, room_2, office_0, frl_apartment_0). So a C1 segmenter
that labels raw-mesh vertices/faces is scored against oracle instances by
*vertex index* — per-vertex oracle `object_id` transfers exactly, no IoU
matching or ICP needed. `tools/c1_exact_eval.py` implements this: oracle
vertex membership = majority incident-face `object_id` (ties -> smallest id,
untouched -> background), greedy max-overlap 1:1 matching, per-instance
vertex IoU, matched/unmatched counts, entity matches at IoU 0.25/0.5/0.75
(the honest detection numbers — the greedy match count alone is any-overlap
pairing incl. structural classes), per-oracle coverage (top covering
prediction + fraction, for merge-aware attribution), oracle recall
(informational IoU thresholds — no pass threshold by design),
over-/under-segmentation at the 10%-coverage rule, background/unassigned
fractions, and SUPPORT-OWNER recall (recovery of the supporting furniture
only — tables/chairs/shelves; NOT the items on them, whose recovery is
c1_run's per-question answer recall; null for scenes with zero owners —
room_1 has none). Companion tools: c1_run (B-relative per-question
precision/recall + merge-aware lost/gained attribution),
c1_failure_classes (zero-GPU four-way: recovered / merged /
lost_by_resolver / no_raw_proposal, with the merged-despite-viable-proposal
overlap counted separately), c1_resolve_sweep (MIN_SCORE measurement),
c1_reresolve (operating-point regeneration, recorded as a
benchmark-definition choice). The evaluator hard-fails (exit 1) on G1
violations, bundle-hash mismatches, and length mismatches — contract
violations, not low scores.

### Known representational deltas of the dense-assignment contract

Measured on the oracle-as-prediction plumbing run (office_0): (1) a DENSE
assignment gives each shared boundary vertex to exactly one instance, while
B's per-face boxes count shared vertices in every touching object — so C1
boxes shrink by millimeters at object boundaries (office_0 table bottom:
+6.6 mm), which band-edge cases can amplify into answer flips; (2) the
candidate `min_vertices` filter (default 20) drops tiny oracle objects —
office_0's only "against the wall" answer, other-leaf obj_65, is a
**1-vertex** object. Both effects are contract properties, not model errors;
`tools/c1_run.py` reports them explicitly.

## A/B baseline facts (2026-07-12, post F1/F2/F4)

Frame parity holds on all six scenes (room_0, room_1, room_2, office_0,
frl_apartment_0, apartment_0): identical yaw, floors, surfaces, uid sets.
Box deltas and battery-answer diffs (A vs B):

- **room_1** — center |Δ| median 0.007 m; extent |Δ| median 0.19 m,
  max **2.02 m (bed)**. The big extent deltas are A's per-object OBB→AABB
  inflation (the F3 residual): the JSON `oriented_bbox` of an object rotated
  relative to its room inflates when axis-aligned, while B's vertex bounds
  are tight. Consequence: **B does NOT produce the bed-ATTACHED_TO false
  positive** that A does, and comforter/lamp/nightstand drop out of
  near-wall (tight boxes no longer reach the band).
- **room_2** — center median 0.004 m; extent median 0.04 m, max 0.99 m
  (table, same OBB-inflation cause); table + picture leave near-wall under B.
- **office_0** — center median 0.002 m; extent median 0.016 m, max 0.45 m
  (rug). Under B the rug leaves "on the floor" (its true top-surface
  vertices sit above the 2 cm rest band — mesh measures the pile surface,
  JSON boxes the full slab), and two `anonymize_text` objects join
  "on the table".

Interpretation limit: neither A nor B is ground truth; B is generally the
tighter geometry (real vertex bounds), and several A-only answers were known
false positives. These are box-source differences, NOT model improvements —
key review (Phase 8 E2) stays the arbiter.

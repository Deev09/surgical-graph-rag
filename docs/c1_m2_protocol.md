# C1-M2 — Segment3D staged pilot (protocol, frozen BEFORE any run)

Written 2026-07-13, before any Segment3D inference. The stopping rule below
is predeclared so the backend comparison cannot be shaped after seeing
results.

## Why Segment3D, why staged

Mask3D C1 failure profile (docs/c1_closeout.md): merging 42% + no-proposal
28% dominate; direct resolver loss is 1.5% — threshold tuning is exhausted,
and soft-NMS cannot split a mask already spanning multiple objects.
Segment3D is designed for fine-grained class-agnostic masks (paper:
small-object AP50 15.9 vs Mask3D 4.5; overall Replica 18.7 vs 18.0 — a
targeted bet on the small-object tail, not an assumed overall win).
Caveats: same legacy py3.10/CUDA-11.3/MinkowskiEngine stack (Colab recipe
reusable), but its own-scene demo needs graph-based mesh segmentation
(Segmentator) as preprocessing — an adapter, not direct-PLY plug-and-play.

## Pins (filled 2026-07-13, BEFORE any inference; recorded in meta.json)

- Segment3D: `LeapLabTHU/Segment3D` @
  `c510d89a66c372c5358384d6d619f713506214db` (main, 2024-12-29)
- Checkpoint: Google Drive id `1Swq9d7rjV2Q1lTuXiKh1z0OZPt9V4sgO` (official
  README demo checkpoint); sha256 recorded at download time in meta.json
- Segmentator: vendored in-tree at the pinned repo commit (upstream ref
  `3e5726500896748521a6ceb81271b0f5b2c0e7d2`, ScanNet Felzenszwalb mesh
  segmentation); built with `make`; invocation exactly as upstream
  `process.sh`: `./segmentator mesh.ply 0.01 20` →
  `mesh.0.010000.segs.json` (kThresh=0.01, segMinVerts=20)
- Inference parameters: upstream `scripts/run_demo.sh` verbatim —
  num_queries=400, topk_per_image=-1, use_dbscan=true, dbscan_eps=0.05,
  dbscan_min_points=5, data.remove_small_group=15, train_on_segments=true;
  preprocessing (voxelization, color normalization) = upstream demo defaults
- Predeclared code deviations (both anchor-asserted patches, recorded in
  DEVIATIONS): (1) `cuml.cluster.DBSCAN` → `sklearn.cluster.DBSCAN` (CPU;
  same eps/min_samples semantics; avoids installing RAPIDS into the legacy
  CUDA-11.3 env); (2) demo.py's pyviz3d visualization call replaced by a
  masks/scores export (full-resolution `masks_binary` [K, N] bool in
  original vertex order + per-mask scores)
- Resolution: the SAME frozen contract — `segmenter/mask_resolve.py`,
  min_vertices=20, and **predeclared MIN_SCORE=0.2** for the headline run
  (a sweep may be REPORTED separately; the headline score is not chosen
  after inspecting the four evaluation scenes)

## Isolation (G2, unchanged)

The Colab runtime receives ONLY mesh.ply + the oracle-free segmenter files.
Preprocessing (Segmentator) must be shown to read only mesh.ply — no
info_semantic.json, no mesh_semantic.ply. Masks + confidences must be
exported at FULL resolution in ORIGINAL vertex order (undo any
voxelization/segment pooling before export).

## Staged execution + predeclared stopping rule

**Stage 1: room_2 only.** Compare against Mask3D's frozen-0.2 room_2 run
(`runs/phase8_c1/ms02/`). The pilot PASSES only if ALL of:

1. entity R@0.5 ≥ 0.42 (i.e. +0.08 absolute over 0.34)
2. answer recall vs B ≥ 0.44 (+0.05 over 0.39)
3. answer precision vs B ≥ 0.90
4. support-answer recall > 1/6
5. merged + no_raw_proposal count falls ≥ 20% (from 34)

**If room_2 fails the gate: STOP.** Do not spend runtime on frl.
**Stage 2 (only on pass): frl_apartment_0** — the multi-room merge stress
test. **Stage 3: office_0 + room_1** to complete the table.

All scoring through the unchanged evaluator stack (c1_exact_eval, c1_run,
c1_failure_classes, c1_resolve_sweep) — the comparison is backend-only.

## VERDICT (stage 1, room_2, 2026-07-13): GATE FAILED — STOPPED

Predeclared rule applied; no further scenes run. 4 of 5 criteria missed:

| criterion | needed | Mask3D @0.2 | Segment3D @0.2 | result |
|---|---|---|---|---|
| entity R@0.5 | ≥ 0.42 | 0.34 (18/53) | 0.32 (17/53) | FAIL |
| answer recall vs B | ≥ 0.44 | 0.39 | **0.53** | pass |
| answer precision vs B | ≥ 0.90 | 0.96 | **0.52** | FAIL |
| support-answer recall | > 1/6 | 1/6 | 1/6 | FAIL |
| merged + no_proposal | ≤ 27 | 34 | 30 | FAIL |

The diagnostic sweep (reported separately, headline unchanged): entity
R@0.5 is flat at 0.32 for min_score 0.0–0.3 and falls beyond — the failure
is threshold-robust. Support-OWNER recall is 0.60–0.70 across the whole
sweep vs Mask3D's 1.00: Segment3D fragments/merges the large furniture.

**The informative part — CORRECTED 2026-07-13** (the first version of this
paragraph over-read `no_raw_proposal=1` as "52/53 viable"; the failure
classifier assigns `merged` before checking raw viability, so its class
counts are composition-stage outcomes, not proposal-coverage statistics).
The orthogonal per-object viability cut gives:

| raw-mask result @ IoU 0.5 | Mask3D | Segment3D |
|---|---|---|
| entities with a viable individual raw mask | 20/53 | **30/53** |
| recovered after composition | 18 | 17 |
| viable raw mask but NOT recovered | 2 | **13** |
| merged entities without a viable individual mask | 21 | 22 |

So Segment3D genuinely raises the proposal ceiling (30 vs 20) — but not to
52/53. Its composition stage wastes **13 viable masks** (Mask3D wastes 2),
and of its 29 merged entities only 7 have a viable individual mask; the
other 22 would need objects CONSTRUCTED from multiple fragments, not merely
the right existing mask selected. Precision collapses because mis-composed
segments acquire wrong relations. The bottleneck is still composition, with
two distinct sub-problems: selection (13 winnable now) and construction
(fragment assembly, harder).

Implication for the roadmap: a composition experiment over Segment3D's
saved raw masks has a measurable selection-only ceiling of 30/53 (vs the
current 17) before any fragment assembly. That is a NEW experiment
requiring its own predeclared protocol — and it should be designed against
HUMAN-verified answer keys, not B-relative metrics (optimizing composition
against B risks teaching it B's known box artifacts). Mask3D @0.2 remains
the C1 reference backend.

**Addendum 2026-07-31 (stage-0 ceiling measurement, zero GPU):** greedy
oracle-guided unions of the saved masks add ZERO entity recall at IoU 0.5
on either backend — the fragment-ASSEMBLY half of the composition
hypothesis is dead for the saved room_2 masks; the entire winnable set (13
entities, 0.32 → 0.57) is pure mask SELECTION, mostly winner-takes-all
losses of near-perfect masks (a plate at raw IoU 0.999 delivered at 0.000).
See `docs/c1_composition_ceiling.md`.

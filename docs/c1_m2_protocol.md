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

## Pins (fill in at session start, BEFORE inference; record in meta.json)

- Segment3D commit: `________`
- Segment3D checkpoint URL + sha256: `________`
- Segmentator (graph mesh segmentation) commit + build hash: `________`
- Preprocessing parameters (voxel size, segmentator k/threshold): `________`
- Query count + DBSCAN settings: upstream defaults, recorded verbatim
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

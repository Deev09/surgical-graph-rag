# C1-P1 — multi-view 2D-to-3D proposal generation (draft protocol)

**Status: DRAFT — AWAITING PROJECT-OWNER SIGN-OFF. No model download,
scene inference, gate evaluation, or parameter tuning is authorized by this
document. Once signed off and frozen, failed gates may not be changed
mid-run.**

Written 2026-08-02 after the accepted and tagged MVP-v1.0 release. This is a
new experiment class. C1-M2C exhausted score, size, overlap, containment,
corroboration-count, and retained-fraction selection rules; this protocol
introduces RGB evidence at candidate generation instead of revisiting that
closed rule family.

## Decision this experiment answers

Can class-agnostic masks from multiple rendered RGB views of the raw
`mesh.ply`, fused through mesh adjacency alone, add viable small-object 3D
proposals that the frozen Mask3D backend never proposed?

This is deliberately a **proposal-ceiling experiment**, not a deployable C1
replacement. A pass only justifies a later, separately predeclared C1-P2
experiment for oracle-free scoring/composition. It does not authorize choosing
proposals with oracle labels or reporting an end-to-end QA improvement.

## Hypothesis

On development scene `room_2`, a fixed multi-view RGB proposal bank will raise
single-proposal entity coverage at IoU 0.5 from Mask3D's 20/53 viable raw masks
to at least 30/53 when the two banks are pooled, while adding at least four
entities that occur in positive human-key answers. The same frozen generator
will then add useful proposals on both prospective transfer scenes without any
scene-specific changes.

## Frozen anchors

| scene | oracle entities | Mask3D viable raw @0.5 | Mask3D delivered @0.5 | role |
|---|---:|---:|---:|---|
| room_2 | 53 | 20 | 18 | development |
| office_0 | 47 | 13 | 12 | transfer 1 |
| room_1 | 45 | 21 | 17 | transfer 2 |

For context only, Segment3D's room_2 raw/compatible ceiling reached 30/53,
but its delivered result was 17/53 and its predeclared backend and selection
experiments failed. C1 therefore remains frozen at Mask3D @ `MIN_SCORE=0.2`,
`min_vertices=20`. Existing answer keys, graph rules, and evaluators are not
changed by this experiment.

## Inputs and isolation boundary

Allowed at proposal-generation time:

- raw `mesh.ply` vertex positions, RGB values, and triangle indices;
- the already-frozen gravity/yaw transform used by the mesh pipeline;
- the model's own 2D mask quality filtering;
- raw-mesh adjacency and multi-view co-membership counts.

Forbidden at proposal-generation time:

- `mesh_semantic.ply`, `info_semantic.json`, oracle object ids or labels;
- human keys, variant A/B artifacts or answers, C1 matches, or C2 labels;
- class names, class prompts, per-scene thresholds, or hand-selected cameras.

Oracle membership and human keys may be opened only by the evaluator after the
proposal sidecar has been finalized and hash-stamped. An I/O isolation test
must fail the run if a forbidden semantic/key path is read by the generator.

## Model pin and inference configuration

- implementation: Meta `facebookresearch/sam2`, repository commit
  `2b90b9f5ceec907a1c18123530e92e794ad901a4`;
- checkpoint/config: SAM 2.1 Hiera Large,
  `sam2.1_hiera_large.pt` + `configs/sam2.1/sam2.1_hiera_l.yaml`;
- input: RGB `uint8`, 1024 × 1024;
- automatic-mask generator values: `points_per_side=32`,
  `points_per_batch=64`, `pred_iou_thresh=0.8`,
  `stability_score_thresh=0.95`, `stability_score_offset=1.0`,
  `mask_threshold=0.0`, `box_nms_thresh=0.7`, `crop_n_layers=0`,
  `crop_nms_thresh=0.7`, `min_mask_region_area=0`, `use_m2m=false`,
  `multimask_output=true`, `output_mode=uncompressed_rle`;
- numerical mode: `torch.inference_mode()`, CUDA bf16 autocast when supported;
- seeds: Python, NumPy, and Torch all `0`; deterministic render order;
- versions, CUDA device, driver, weights SHA-256, and elapsed time are written
  to every sidecar. The weights SHA-256 is filled by the freeze commit after
  download and before any scene inference.

These are the upstream defaults declared for the Hiera-L automatic-mask
generator at the pinned source commit. There is no threshold sweep and no
second checkpoint.

## Deterministic view contract

All coordinates first receive the frozen pipeline gravity/yaw transform, so
`+z` is physical up. Let `(cx, cy)` be the raw scene XY bounding-box center,
`sx, sy` its spans, and `z0` its minimum z.

- Five camera origins at height `z0 + 1.60 m`: `(cx,cy)`,
  `(cx ± 0.18*sx,cy)`, and `(cx,cy ± 0.18*sy)`.
- Eight yaw headings per origin: `0, 45, ..., 315` degrees in the canonical
  XY frame; pitch is `-10` degrees; roll is `0`.
- Perspective vertical field of view `90` degrees; near plane `0.05 m`; far
  plane is the scene bounding-box diagonal plus `1 m`.
- Exactly 40 views per scene. No view may be dropped, replaced, or inspected
  to choose a different camera. A camera inside geometry is an observed
  failure of this fixed design, not permission to move it.
- Raw vertices are rendered as depth-tested RGB point splats. Point diameter
  is 3 pixels. A paired integer id buffer records the frontmost source vertex
  at every covered pixel; uncovered pixels are `-1`. RGB and id images share
  exactly the same projection, depth test, draw order, and hash manifest.

## Fixed 2D-to-3D fusion

The generator creates an overlapping proposal **bank**, not a final dense
assignment:

1. Run the pinned automatic-mask generator independently on all 40 RGB views.
2. Lift each accepted 2D mask to the set of non-negative vertex ids under its
   pixels. Masks lifting to fewer than 20 unique vertices are discarded.
3. For every raw-mesh edge `(u,v)`, count views in which both endpoints are
   visible and the fraction of those views in which they share at least one
   accepted 2D mask. Zero-co-visible edges receive confidence `0`.
4. At each predeclared confidence cut `{0.25, 0.50, 0.75}`, form connected
   components using only mesh edges meeting the cut. Retain components with
   at least 20 vertices and at most 40% of all scene vertices.
5. Pool components across the three cuts. If two proposals have vertex IoU
   at least `0.95`, retain the one from the higher confidence cut; remaining
   ties use larger vertex count, then the lexicographically smallest vertex-id
   digest. No oracle signal participates in this ordering.

The three cuts are a predeclared multi-scale output of one generator run, not
three evaluated variants. They may not be removed or selected after seeing
oracle results.

## Stage 0 — implementation validity (zero scene inference)

Before the GPU run, synthetic tests must prove:

- RGB and id buffers agree under depth and occlusion;
- lifting returns exactly the visible source vertex ids under a known mask;
- co-membership separates two adjacent synthetic objects given distinct 2D
  masks and retains each object when its views agree;
- output is byte-identical across two CPU render/fusion runs;
- the generator cannot import or open semantic meshes, metadata, keys, or
  frozen answers;
- output sidecars conform to `SegmentationArtifact` vertex indexing without
  altering the frozen Mask3D bundle.

Failure to meet Stage 0 stops the experiment before model inference. Fixes are
allowed only while tests use synthetic geometry and no scene model output has
been generated; the final implementation and tests are then committed and
hash-pinned.

## Stage 1 — room_2 development run

Budget: one SAM inference over the 40 frozen views. Rendering/fusion may be
replayed from the same saved 2D masks only to fix a serialization bug; the
original masks and hashes must remain unchanged. No model or parameter rerun.

The evaluator reports P1-alone and pooled `Mask3D ∪ P1` proposal coverage.
For every oracle entity it records best single-mask IoU and source, but oracle
matches never feed back into the generator.

All gates must pass:

| gate | predeclared criterion |
|---|---|
| G1 | pooled viable entities @IoU0.5 ≥ **30/53** (Mask3D: 20/53) |
| G2 | P1 contributes ≥ **10** newly viable entities absent from Mask3D |
| G3 | ≥ **4** newly viable entities occur in at least one positive human-key citation set |
| G4 | ≥ **2** newly viable entities occur in a positive `on furniture` answer |
| G5 | at least **80%** of oracle entities have ≥10% of their vertices present in the id buffers across the 40 views (camera/evidence coverage guard) |
| G6 | final P1 bank contains ≤ **2,000** proposals and its serialized masks total ≤ **2 GiB** |

If any gate fails: **STOP**, commit the negative result, and spend no GPU on
the transfer scenes. Gates are not adjusted and the transfer scenes are not
used to redesign the camera set, fusion cuts, or model configuration.

## Stage 2 — frozen prospective transfer

Only after all development gates pass, run the identical committed generator
once on `office_0` and once on `room_1`, with no intervening code or parameter
change. Evaluate both only after both proposal sidecars are final.

All transfer gates must pass:

| gate | predeclared criterion |
|---|---|
| H1 | pooled viable @0.5 improves by ≥ **5** entities on office_0 (baseline 13/47) |
| H2 | pooled viable @0.5 improves by ≥ **5** entities on room_1 (baseline 21/45) |
| H3 | each scene adds ≥ **2** newly viable positive human-key citation entities |
| H4 | each scene passes the same G5 camera/evidence coverage guard |
| H5 | each scene stays within the same 2,000-proposal / 2-GiB cap |

These are **prospective transfer checks, not untouched scenes**: their human
keys and frozen baseline results already exist in the repository. A pass is
evidence that the proposal mechanism transfers under fixed settings, but a
publication must not call it sealed-holdout generalization.

## Budget, stopping rule, and decision after the run

- Maximum model inference: one development scene plus two transfer scenes;
  40 views each, one checkpoint, one parameterization.
- No rule versions, visual cherry-picking, prompt variants, threshold sweeps,
  or rescue run.
- Any run invalidated by a crash before all 40 views complete is rerun from
  scratch with the identical committed environment and is reported.
- Dev fail → negative result, P1 closed, no transfer spend.
- Dev pass + transfer fail → negative transfer result; P1 is not adopted.
- Dev + transfer pass → freeze the proposal artifacts and draft C1-P2 for an
  oracle-free scorer/composer with precision and human-QA gates. P1 proposals
  are still evaluation-only until P2 passes.

## Required artifacts and reporting

- committed generator, synthetic tests, and environment/install recipe;
- raw model checkpoint SHA-256 and upstream source pin;
- per-scene camera/RGB/id manifest with hashes;
- sanitized 2D-mask and 3D-proposal sidecars, or a reproducible download
  manifest if Git size limits prohibit tracking them;
- per-entity best IoU table for Mask3D, P1, and pooled banks;
- every G/H gate, proposal counts, visible-vertex coverage, elapsed time,
  peak VRAM, and failures;
- a dated verdict addendum here. No hand-computed headline metrics.

## Explicitly out of scope

- changing the Mask3D @0.2 reference, the human keys, graph semantics, Router,
  support allowlists, or evaluation definitions;
- C2 label tuning, C3, `frl_apartment_0`, NeRF/3DGS integration, or a live
  free-text demo;
- claiming QA improvement from an oracle-guided proposal ceiling;
- solving the AABB/support-representation ceiling already documented in C1.

## Preflight (2026-08-02, read-only — no gate, budget, or rule changed)

Verified against the current repository before requesting activation:

- **Protocol integrity:** this file is unmodified since its draft commit
  `7f04137` — hypothesis, gates G1–G6/H1–H5, budget (1 dev + 2 transfer
  scenes, 40 views each, one checkpoint, one parameterization), and the
  dev-fail stopping rule are exactly as drafted.
- **Frozen anchors re-verified** against the ms02 failure-class reports
  (schema v2): room_2 53 entities / viable 20 / delivered 18; office_0
  47 / 13 / 12; room_1 45 / 21 / 17 — all match this document's table.
- **Inputs on disk and pinned:** raw `mesh.ply` hash-locked for all
  three scenes (`tools/replica_scenes.lock.json`); frozen gravity/yaw
  frames present for all three (`eval/fixtures/c3_0_frames.json`);
  human keys `human_verified` for all three.
- **Post-draft context that does NOT alter this protocol:** the SR quad
  parser now exists (room_2/office_0 raw meshes are pure quads, room_1
  pure quads, room_0 pure triangles — the renderer must consume the
  SR-triangulated geometry); Stage 0m closed the mesh-plane surface
  family, making this protocol the active pivot target per the owner's
  decision rule.
- **Two notes for the freeze commit (cosmetic/logistical, not gate
  changes):** (1) the sidecar contract class is named
  `SegmentationOutput` in the repository (this document says
  "SegmentationArtifact"); (2) execution environment is not fixed by
  the protocol — recommendation: Colab CUDA (consistent with the
  Mask3D/Segment3D runs and the declared bf16-autocast mode) rather
  than local MPS/CPU; the environment actually used is recorded in
  every sidecar either way.
- **Not yet built (activation scope):** renderer + id buffers, SAM
  harness, fusion, evaluator, Stage-0 synthetic tests, isolation test.
  Stage 0 completes and the preparation freeze (with checkpoint sha)
  lands BEFORE any scene inference, per this protocol.

## Sign-off

- [x] Project owner approves the hypothesis, frozen generator, gates, budget,
      and transfer interpretation (2026-08-02, project owner / deevyaswain —
      "approved, activate C1-P1", following the recorded preflight).
- [x] Preparation-only freeze commit (this commit) records the checkpoint
      and environment BEFORE any scene inference:
      `sam2.1_hiera_large.pt` sha256
      `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`
      (downloaded from the official
      `dl.fbaipublicfiles.com/segment_anything_2/092824/` URL; stored
      locally outside the repo; the notebook re-downloads and HARD-FAILS
      on sha mismatch). Implementation frozen with Stage 0 green (6/6):
      `segmenter/view_render.py` (40-view contract, joint depth-sorted
      splats), `segmenter/proposal_fusion.py` (lift + co-membership +
      cuts + dedupe), `tools/c1p1_render.py` / `c1p1_fuse.py` /
      `c1p1_eval.py`, `notebooks/c1p1_sam2_colab.ipynb` (pinned commit,
      params, seeds; refuses a wrong checkpoint). Inference environment:
      Colab CUDA per the preflight recommendation.


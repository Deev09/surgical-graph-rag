# Reproduction manifest

One page answering: what data, what code pins, what environment, what
hardware, and what command reproduces each claimed result. Hashes are never
duplicated here — this page points at the files that pin them.

## 1. Datasets

- **Replica** (facebookresearch/Replica-Dataset). Scenes used:
  `room_0`, `room_1`, `room_2`, `office_0`, `frl_apartment_0`
  (plus `apartment_0` in earlier phases).
- Files per scene: `habitat/info_semantic.json` (variant A),
  `habitat/mesh_semantic.ply` (variant B), `mesh.ply` (variant C input).
- Fetch + verify: `python3 tools/fetch_replica_scenes.py` streams only the
  needed members from the upstream split tar; `--verify` checks every file
  against **`tools/replica_scenes.lock.json`** (sha256 + size, checked in).
- Local data root: any path; the tools take `room_dir` arguments. The lock
  records the root used for the committed runs.
- Raw Replica `mesh.ply` quirk: 9 float vertex properties + **quad** faces
  (`property list uint8 int`, count 4). The Colab notebooks rewrite it as a
  triangulated, color-preserving PLY before segmentation
  (`notebooks/c1_segment3d_colab.ipynb` cell [8a]); vertex order is preserved,
  which is what the exact-correspondence evaluator (gate G1) depends on.

## 2. Local environment

- Python ≥ 3.10 (developed on 3.13, macOS). Dependencies: `numpy`, `Pillow`
  (see `requirements.txt`; `openai`/`anthropic` are legacy-v1 only).
- Canonical test command: **`python3 tools/run_tests.py`** — runs all 57
  script-style test files, one subprocess each; dataset-guarded tests
  self-skip when the Replica data is absent. Exit code 0 = green.

## 3. GPU segmentation backends (Colab, not local)

Both backends run on Colab Pro (NVIDIA A100-SXM4-40GB for all committed
runs) inside a pinned legacy environment (Python 3.10, CUDA 11.3 subset via
Miniforge, torch 1.12.1+cu113, MinkowskiEngine 0.5.4). The complete,
cell-by-cell environment recipe IS the notebook — there is no separate
setup script:

- **Mask3D (C1 reference backend)** — `notebooks/c1_mask3d_colab.ipynb`.
  OpenMask3D mask stage @ commit `3bc3fc52`, arbitrary-scenes checkpoint
  (sha in each bundle's `meta.json`). Inference 55–86 s/scene.
- **Segment3D (failed M2 pilot)** — `notebooks/c1_segment3d_colab.ipynb`.
  All pins (repo commit, checkpoint Drive id + sha256, vendored Segmentator
  ref, run_demo.sh parameters, predeclared deviations) are frozen in
  **`docs/c1_m2_protocol.md`**, written before any inference ran.

Backend outputs are `SegmentationOutput` sidecar bundles
(`vertex_instance_ids.npy`, `instance_table.json`, `meta.json`,
`raw_masks.npz`). Bundles are NOT in git; authoritative copies live on
Google Drive and every hash (bundle output, input mesh, raw masks, hardware,
runtime) is pinned in **`docs/c1_artifact_manifest.json`**. Never rename or
hand-assemble a bundle — `meta.json` is its identity.

Everything downstream of the saved bundles is zero-GPU and local:
`tools/c1_run.py`, `tools/c1_exact_eval.py`, `tools/c1_failure_classes.py`,
`tools/c1_resolve_sweep.py`, `tools/c1_reresolve.py`.

## 4. Frozen decisions (benchmark definitions, not improvements)

- Mask resolution: highest-score-wins, `MIN_SCORE=0.2`, `min_vertices=20`
  (`segmenter/mask_resolve.py`; rationale in `docs/c1_closeout.md`).
- A↔B frame parity: one rotation source, B inherits A's surfaces verbatim
  (`docs/mesh_pipeline_contract.md`; regression
  `tests/importers/test_mesh_frame_parity.py`).
- F4 room-scale-flat exclusion: opt-in, ON only in the battery/review path
  (`eval/questions/phase8/REVIEW_GUIDE.md`, findings F1–F4).

## 5. Result → command map

| claim | source of truth | reproduce with |
|---|---|---|
| Phase 8 human-verified headline (56 Q, 4 scenes) | `runs/phase8_scorecard/aggregate.json` | `python3 tools/scene_scorecard.py` |
| Human answer keys | `eval/questions/phase8/*_qa.json` (`answer_key_type: human_verified`) | review process in `REVIEW_GUIDE.md` |
| C1 Mask3D four-scene table | `docs/c1_closeout.md`, reports in `runs/phase8_c1/ms02/` | `python3 tools/c1_run.py` on the pinned bundles |
| Segment3D gate verdict | `docs/c1_m2_protocol.md` (predeclared gate + verdict) | same evaluator stack on the s3d bundle |
| Negative results (expansion, uncertainty pool) | `docs/query_scoped_expansion_prototype.md`, `docs/uncertainty_policy_prototype.md` | `tools/query_scoped_expansion_demo.py`, `tools/uncertainty_policy_demo.py` |
| Per-phase gates (frozen behavior) | `tools/phase*_exit_gate.py` | run directly; all green as of 2026-07-31 |
| Legacy v1 benchmark | `manifest.json`, `evaluation_table.json` | `SKIP_LLM=1 python3 tiny_graph_demo.py --benchmark-only` |

## 6. Comparability rules

Phase 8 numbers are a separate track — never comparable to the legacy v1
benchmark or the per-phase scorecards. B-relative C1 metrics measure
agreement with B's boxes, not reality; only the human-verified keys support
real precision/recall claims. Any change to keys, thresholds, or relation
semantics is a benchmark-definition change and must be labeled as such.

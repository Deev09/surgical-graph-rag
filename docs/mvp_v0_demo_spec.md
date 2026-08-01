# MVP-v0 demo — specification (approved, implemented, and amended)

**Status: APPROVED 2026-08-01 with spec defaults (room_0 A/B-only row
kept; PNGs embedded; verbalized text included) and IMPLEMENTED —
`tools/mvp_demo.py` + `tools/mvp_report_html.py`, tests in
`tests/tools/test_mvp_demo.py`. Accepted by the owner before C2.0; the
current amended artifact includes C2 and office_0. Determinism and hard
reference checks are re-run after every amendment; after the office_0
amendment, determinism passed over six artifacts and 62/62 test files were
green.**

## What MVP-v0 is

One offline command that demonstrates the project's honest headline — *a
modular, queryable spatial-graph reasoner that exposes uncertainty and
isolates failures under imperfect 3D instance extraction* — by running the
SAME frozen graph + Router over the four current input variants (A oracle
boxes, B mesh-derived boxes, C1 learned instances, and evaluation-only C2
learned labels) and scoring them against
the human-verified answer keys, with every number carrying its provenance.

It is a packaging milestone, not an experiment: every metric it emits
already exists in committed reports; the demo's job is to reproduce them
from source, disclose their limits, and render them readable.

## Command

```
python3 tools/mvp_demo.py [--out-dir runs/mvp_v0] [--scene SCENE_ID]
                          [--check-determinism]
```

- Offline: no network, no GPU. Runtime target < 5 minutes total.
- Inputs (all verified before any work; hard-fail on mismatch):
  - scene data pinned by `tools/replica_scenes.lock.json`,
  - frozen C1 bundles `runs/phase8_c1/bundles_ms02/<scene>` hash-checked
    against `docs/c1_artifact_manifest.json`,
  - human keys `eval/questions/phase8/<scene>_qa.json`
    (`answer_key_type: human_verified` required, else the scene is
    refused — plausibility keys never enter MVP-v0).

## Scene set (predeclared)

| scene | variants | why |
|---|---|---|
| replica_room_1 | A, B, C1 | human key + frozen ms02 bundle |
| replica_room_2 | A, B, C1 | human key + frozen ms02 bundle |
| replica_room_0 | A, B only | human key exists; Mask3D was never run on room_0 and v0 spends no GPU — its C1 column renders as "not run", never as 0 |

office_0 and frl_apartment_0 are EXCLUDED (no human keys). Adding them
later is a data addition, not a spec change.

## Question set (predeclared)

The human-key questions VERBATIM — nothing added, removed, or reworded.
The keys are the predeclared question set; scoring semantics are exactly
`tools/c1_joint_ceiling.py::score_against_key` (micro-P/R over exhaustive
answer questions; membership-only otherwise; must_not violations tracked;
C1 pred-space uids translated via the exact-correspondence match table,
untranslatable uids prefixed `pred:` and rendered as "unlabeled segment").

## Outputs

### 1. Deterministic JSON — `runs/mvp_v0/<scene>_mvp.json` + `aggregate.json`

Schema `mvp_v0_report_v1`. Two consecutive runs on the same machine MUST
be byte-identical; `--check-determinism` runs everything twice and diffs
(exit non-zero on any difference). No timestamps, no absolute paths, no
dict-ordering nondeterminism inside the payload; run metadata that cannot
be deterministic (wall-clock, host) lives in a separate
`runs/mvp_v0/run_env.json` explicitly excluded from the determinism check.

Per scene, per variant (A / B / C1), per key question:
- `outcome` (answer / empty / defer / unknown), `cited` (uid + class
  label), `verbalized` (the Router's answer text),
- score vs key: hits, misses, must_not violations; P/R where the key is
  exhaustive,
- for C1: which citations come from matched instances (with IoU) vs
  unlabeled segments.

Per scene and aggregate: the standard table (micro-P / micro-R /
per-relation rollup / graph edges / entity count), plus a `provenance`
block: git commit, input hashes, bundle `output_sha256`, graph bundle
hash, scorer identity, key fixture ids, the C1 isolation statement
(labels + surfaces injected; only instance boundaries are learned), and
the completeness profile used (`source: oracle`).

### 2. Self-contained HTML — `runs/mvp_v0/report.html`

One file, no external requests (inline CSS/JS; the existing per-scene
question-sheet and UID-index PNGs embedded as data URIs). Sections:

1. **Headline table** — scenes × variants × (micro-P, micro-R, support
   hits, edges), with the honest one-line reading for each row.
2. **Per-question cards** — question text, the human answer (uids +
   labels), then A / B / C1 answers side by side with hits green, misses
   and false cites marked, C1 unlabeled segments visibly distinct.
3. **Disclosures** (fixed text, sourced from the docs): key semantics
   rulings; near-wall is membership-only; attachment recall is a
   downstream-semantics finding (even A scores ~1/14 on room_2); the 15
   representationally-unreachable support answers; C1 injection isolation;
   "empty ≠ no such object" under the oracle completeness profile.
4. **Provenance appendix** — the JSON provenance block rendered verbatim.

The HTML is generated from the deterministic JSON only (same determinism
requirement, embedded images being deterministic files).

## Implementation shape (est. one session)

- `tools/mvp_demo.py` (~350 LOC): orchestration + JSON; reuses
  `import_habitat_room` / `import_mesh_room` / `build_c1_eval_bundle` /
  `build_graph` / Router / `score_against_key` unchanged.
- `tools/mvp_report_html.py` (~250 LOC): JSON → HTML (pure rendering, no
  metric computation — a number appearing in HTML but not JSON is a bug).
- `tests/tools/test_mvp_demo.py`: synthetic-scene run (the existing
  two-cube fixtures), schema check, determinism check, and a consistency
  assertion that recomputed room_2 rows equal the committed reference
  values (A: P 0.95 / R 0.41; C1/Mask3D: P 1.00 / R 0.24).

## Acceptance criteria (what "accepted" means, testable)

1. Single offline command; < 5 min; no GPU, no network, no writes outside
   `--out-dir`.
2. `--check-determinism` passes (byte-identical JSON + HTML).
3. Recomputed metrics match the committed reference reports exactly
   (room_2 A/C1/C2 plus office_0 A/B/C1/C2; mismatch = hard fail, not a
   warning).
4. Full test suite green (`python3 tools/run_tests.py`); no frozen
   pipeline file, gate, key, or benchmark artifact modified.
5. HTML opens from disk in a browser with everything visible offline;
   owner reviews per-question cards for at least one scene and confirms
   the disclosures render.

## Original non-goals (v0; superseded where amendments say otherwise)

C2 (blocked on acceptance of this slice), any GPU inference, rule/
threshold tuning, key edits, new metrics, interactivity beyond static
HTML with inline JS (no server), publishing/hosting (the file is the
deliverable), office_0/frl rows (no human keys).

## Open decisions for sign-off

1. room_0 as A/B-only row: include (spec default) or drop to keep the
   table uniform?
2. Embed PNG sheets (report ~5–10 MB, fully self-contained — spec
   default) or link them as sibling files (small HTML, two-file bundle)?
3. Verbalized answer text in the per-question cards: include (spec
   default) or citations only?

## Sign-off

- [x] Project owner approves scene set, outputs, acceptance criteria,
      and the three open decisions — spec defaults on all three
      (date: 2026-08-01, by: project owner / deevyaswain — "approved
      with spec defaults, build the demo slice")

## Amendment 1 (2026-08-01, owner-instructed): C2 rows + semantic citation

After C2.0 was measured (`docs/c2_matched_labels_protocol.md`), the owner
instructed adding the corrected, explicitly EVALUATION-ONLY C2.0 row to
MVP-v0. Changes: room_1/room_2 gain a C2 variant whose labels come from
the COMMITTED prediction sidecars (`eval/predictions/phase8_c2/`,
hash-pinned to the frozen ms02 bundles; no torch in the demo path);
metric naming clarified — micro P/R are labeled **uid micro-P/R**
(UID/structural membership; the key cites uids, not names) and a
**semantic citation** metric is added (uid-correct citations that also
carry the canonical label; C1 = 1.0 by construction, a scorer
self-check); room_2 C2 reference row (1.0 / 0.2041) added to the hard
reference check. All original acceptance criteria re-verified after the
amendment (determinism PASS, references reproduced, suite green).

## Amendment 2 (2026-08-01, human-key addition): office_0 full ladder

The project owner reviewed office_0 against the raw RGB mesh and promoted a
human key. The scene therefore joins MVP-v0 as A/B/C1/C2 — a data addition
explicitly allowed by the original scene-set rule, not a scoring or pipeline
change. The frozen C2.0 labeler ran once with no prompt/vocabulary changes;
its sanitized sidecar is hash-pinned to the existing office_0 ms02 bundle.
Hard references now cover office A/B uid membership plus C1/C2 semantic
citation. The raw review exposed two limitations preserved in the key:
against-wall remains non-exhaustive under the 17° F3 residual, and a visible
purple floor object has no usable semantic uid.

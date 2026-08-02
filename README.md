# surgical-graph-rag

> A modular, queryable spatial-graph reasoner over real captured 3D scenes —
> typed relations, honest outcomes, and measured failure isolation under
> imperfect 3D instance extraction.

Given a captured indoor scene, the system builds a typed spatial scene graph
(support, wall contact/proximity, attachment, directional relations) and
answers natural-language structural questions ("what is on the table?",
"what is against the wall?") through a compile → execute → verbalize Router
that can answer, return a grounded empty, abstain, or say **unknown** — it
never fabricates certainty the graph cannot support.

What makes it unusual is the evaluation discipline: every input variant is
hash-pinned, every pipeline stage is isolated so failures attribute to exactly
one stage, experiments run under predeclared gates that are allowed to fail,
and negative results are committed as first-class documentation.

## What it is NOT (read this before citing numbers)

- It is **not** an end-to-end NeRF/3DGS system. The only real reconstruction
  adapter is Replica/oracle data.
- The learned-segmentation path (C1) starts from a raw `mesh.ply` but injects
  **oracle labels and structural surfaces** for controlled evaluation.
  Learned semantics (C2.0) is a measured **isolation result only** —
  zero-shot labels on C1's matched instances, closed to further
  optimization — and fully-raw operation (C3) is not implemented.
- The current headline is that the system **exposes and measures its own
  failures** — not that it solves raw-scene QA.

## The input ladder (stage isolation)

Each variant changes exactly one upstream stage, so adjacent comparisons
attribute differences to that stage (`docs/mesh_pipeline_contract.md`):

| variant | boxes | labels | status |
|---|---|---|---|
| **A** | `info_semantic.json` oracle | oracle | frozen baseline |
| **B** | derived from `mesh_semantic.ply` | oracle | frame parity with A frozen |
| **C1** | learned segmenter on raw `mesh.ply` | oracle via exact vertex correspondence | **measured** (Mask3D reference @ MIN_SCORE=0.2; Segment3D pilot failed its predeclared gate — see below) |
| **C2** | learned (= C1 frozen) | learned (CLIP zero-shot on matched instances) | **measured, evaluation-only** — labels are not the bottleneck; C2 optimization stopped (`docs/c2_matched_labels_protocol.md`) |
| **C3** | learned | learned, mesh-derived surfaces | fully raw path not implemented; C3.0 surface-source isolation drafted |

## Measured status (2026-08-01)

**Human-verified baseline (Phase 8).** Four Replica scenes
(room_0/room_1/room_2/office_0) have human-reviewed answer keys; the scorecard
(`runs/phase8_scorecard/`) reports against *reality*, not against the system's
own drafts: 56 questions → 4 fully-correct answers, 27 correct empties, 22
misses, 3 false answers. The misses are dominated by known representational
limits (whole-object AABBs cannot model sofa/chair seat surfaces; the 2 cm
wall-contact band is stricter than human "against the wall"; cabinet/nightstand
are missing from the support-class allowlist). A key the system fails is a
successful review — the keys record what is physically true.

**C1 (raw-mesh instances, oracle labels).** Mask3D backend, four scenes,
frozen operating point: entity recall@IoU0.5 0.25–0.38, answer recall vs B
0.39–0.51. Failure attribution: Mask3D's selection stage is near-optimal and
its ceiling is proposal coverage (~32% of oracle entities have a viable raw
mask); Segment3D raises the proposal ceiling (30/53 vs 20/53 on room_2) but
wastes 13 viable masks in composition and failed 4/5 predeclared gate criteria
— so the pilot stopped after one scene, per protocol
(`docs/c1_closeout.md`, `docs/c1_m2_protocol.md`).

**C2.0 (learned labels, isolation only).** Zero-shot CLIP on matched-
instance point-splats: support-owner labels were 9/10 on room_2 but only
2/7 on the later office_0 transfer; overall top-1 spans 0.25–0.57. One
shelf-label error erased room_2's two support answers, while office_0's
delivered support was already zero under C1. Semantic-citation fidelity
(uid-correct answers that also carry the canonical label) spans 0.31–0.62.
Labels are not the current binding constraint — C1 proposal coverage is —
but they are not robustly solved; C2 optimization remains stopped
(`docs/c2_matched_labels_protocol.md`).

**Committed negative results.** Query-scoped raw-proposal expansion recovers
zero additional support answers on saved proposals
(`docs/query_scoped_expansion_prototype.md`); the uncertainty-preserving
provisional pool is a policy prototype, not a tuned improvement
(`docs/uncertainty_policy_prototype.md`).

## Quickstart

```bash
git clone https://github.com/Deev09/surgical-graph-rag.git
cd surgical-graph-rag
pip install -r requirements.txt   # numpy + Pillow for the current pipeline

# Canonical test command (64 script-style test files, each in its own process;
# dataset-guarded tests self-skip without the Replica data)
python3 tools/run_tests.py

# With the Replica dataset on disk (see docs/reproduction.md):
python3 tools/fetch_replica_scenes.py                 # hash-check pinned inputs
python3 demo/question_battery.py /path/to/replica/room_0 replica_room_0
python3 tools/scene_scorecard.py                      # human-verified headline
python3 tools/mvp_demo.py                             # deterministic A/B/C1/C2 report
python3 tools/mvp_viewer.py                           # interactive 3D evidence viewer
#   -> open runs/mvp_v1/viewer.html (self-contained, offline)
python3 tools/mvp_captioned_demo.py                   # self-running captioned walkthrough
#   -> open runs/mvp_v1/captioned_demo.html (presentation-only derivative)
```

Public MVP walkthrough: **https://deev09.github.io/surgical-graph-rag/**

The immediate generalization experiment is drafted—but not yet authorized—in
[`docs/c3_0_mesh_surfaces_protocol.md`](docs/c3_0_mesh_surfaces_protocol.md).
It replaces only oracle structural surfaces while holding the frame fixed, so
it is C3.0 isolation rather than a fully raw C3 claim. The separate performance
protocol, [`docs/c1_p1_multiview_proposals_protocol.md`](docs/c1_p1_multiview_proposals_protocol.md),
remains parked and unrun.

## Repo layout (current system)

```
common/ extractors/ geometry/   # EntityArtifacts contract, frame, surfaces
graph/                          # typed relation extractors + graph builder
reasoner/                       # RulesCompiler -> RulesExecutor -> Verbalizer (Router)
segmenter/                      # C1: segmentation sidecar contract, mask resolution,
                                #     anonymous candidates, derived eval bundles
demo/                           # Replica importers (A/B), question battery, review sheets
eval/                           # router QA scoring + Phase 8 answer keys (human_verified)
tools/                          # evaluators, scorecard, sweeps, dataset fetch, run_tests
notebooks/                      # Colab GPU backends (Mask3D, Segment3D) — full env recipes
docs/                           # contracts, closeouts, protocols, phase records
tests/                          # 62 script-style test files (tools/run_tests.py)
```

Reproduction (datasets, checkpoints, environments, hardware):
`docs/reproduction.md`.

## Legacy v1 (graffiti_bathroom)

The original prototype — a hand-authored 12-object scene graph with
relation-aware retrieval and an optional LLM answering step — lives on in
`tiny_graph_demo.py`, `scenes/`, `benchmark/`, `baselines/`, `scoring/`,
`relations/`, `parsers/`. Its 10-query benchmark is saturated
(top-1 accuracy 1.0) and is **not comparable** to the Phase 8 track:

```bash
SKIP_LLM=1 python3 tiny_graph_demo.py --benchmark-only
```

## License

MIT — see [LICENSE](LICENSE).

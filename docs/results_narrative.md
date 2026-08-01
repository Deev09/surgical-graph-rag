# Results narrative (publication draft, MVP release)

Status: DRAFT for the owner's editing pass — assembled 2026-08-01 at the
MVP freeze. Every number is a committed, hash-pinned value; sources are
cited per claim. Positioning is the agreed honest one:

> **A modular, queryable spatial-graph reasoner over real captured 3D
> scenes that exposes uncertainty and isolates failures under imperfect
> 3D instance extraction.** It does not claim to solve raw-scene QA, and
> a strong end-to-end NeRF/3DGS claim would require a real reconstruction
> adapter plus learned semantics — neither is claimed.

## Summary

Given a captured indoor scene, the system builds a typed spatial scene
graph (support, wall contact/proximity, attachment, directional) and
answers structural questions through a compile → execute → verbalize
Router with four honest outcomes: answer, grounded empty, defer, unknown.
The contribution is not headline accuracy — it is the evaluation
architecture: an input ladder in which each variant changes exactly one
upstream stage, human-verified answer keys that record reality rather
than system output, predeclared experimental protocols that are allowed
to fail (and did), and end-to-end provenance from every cited answer back
to hash-pinned inputs. Across four human-keyed Replica scenes, the ladder
attributes every recall drop to a specific stage, and an interactive 3D
viewer makes each attribution visible on the actual mesh.

## The ladder and headline numbers

| variant | changes | labels |
|---|---|---|
| A | oracle boxes (`info_semantic.json`) | oracle |
| B | boxes re-derived from the semantic mesh | oracle |
| C1 | learned instances (Mask3D, frozen @0.2) on raw `mesh.ply` | oracle via exact vertex correspondence (evaluation-only) |
| C2 | = C1 instances | zero-shot CLIP on instance point-splats (evaluation-only) |

![UID micro-recall by scene and variant](assets/fig_ladder_recall.svg)

*(Figure: uid micro-recall vs human-verified keys; `runs/mvp_v0/
aggregate.json`. room_0's C1/C2 are absent — not run, never rendered as
zero; office_0's C1/C2 are true zeros.)*

![room_2 recall attribution](assets/fig_room2_attribution.svg)

*(Figure: room_2, each ladder step isolates one stage: box source costs
nothing, instance extraction costs −0.163, labels −0.041.)*

Companion metrics (both defined in `docs/c2_matched_labels_protocol.md`):
**uid micro-P/R** score UID/structural membership (the key cites uids);
**semantic citation** scores whether uid-correct citations also carry the
canonical label — C1 rows score 1.0 by construction (a scorer
self-check), C2 rows score 0.31–0.62.

## Findings (each with its evidence source)

1. **Instance proposal coverage is the binding constraint.** Only ~32%
   of oracle entities have any viable raw Mask3D mask (IoU ≥ 0.5), and
   Mask3D's selection stage is near-optimal (2/20 viable masks wasted on
   room_2) — its ceiling is proposals, not composition
   (`docs/c1_closeout.md`).
2. **Fragment assembly of saved masks is measurably dead.** Oracle-guided
   greedy unions of 2–8 masks add ZERO entity recall at IoU 0.5 on
   either backend; the achievable ceiling is pure mask selection
   (`docs/c1_composition_ceiling.md`).
3. **The selection ceiling is real but unreachable with structural
   signals.** The 30/53 jointly-compatible ceiling (uid-P 0.93 / R 0.29
   vs the human key) exists — but three predeclared oracle-free rule
   versions (scores, sizes, overlap structure, corroboration, retained
   fractions) all failed their gates; backend confidence is
   anti-correlated with mask quality (23/30 winning masks score below
   the frozen threshold). Closing the gap needs new evidence classes
   (`docs/c1_m2c_protocol.md`).
4. **Backend choice beat every repair attempt.** Mask3D @0.2 (uid-P 1.00
   / R 0.24 on room_2's human key) outperformed the Segment3D swap
   (failed 4/5 gate criteria) and all three selection-repair rules
   (`docs/c1_m2_protocol.md`, `docs/c1_m2c_protocol.md`).
5. **Labels are learnable for anchors but not robustly.** Zero-shot CLIP
   reached 9/10 on room_2's support-owner labels yet 2/7 on the office_0
   transfer (overall top-1 0.25–0.57); a single shelf→vent error erased
   room_2's support answers. Labels are not the current bottleneck —
   but only because C1 coverage already dominates
   (`docs/c2_matched_labels_protocol.md`).
6. **Representation limits cap even perfect inputs.** Variant A with
   oracle boxes reaches only R 0.29–0.41: whole-object AABBs cannot
   model seat surfaces (15/20 room_2 support answers unreachable), the
   2 cm attachment band cannot see human "attached" (≈1/14 for every
   variant), and the support-class allowlist misses cabinet/nightstand
   (`eval/questions/phase8/REVIEW_GUIDE.md`, scorecard).
7. **Honest outcomes are load-bearing.** "Empty ≠ absent" under the
   oracle completeness profile, "not run ≠ 0" in every table, and keys
   that the system fails are recorded as successful reviews.

## Negative results (committed as first-class artifacts)

Segment3D pilot (predeclared gate, stopped after one scene); three
selection-repair rule versions (stopped at budget, holdout unspent);
query-scoped proposal expansion (zero recovered answers, 645→1360 edge
inflation); uncertainty-preserving proposal pool (policy prototype only);
C2 vocabulary hygiene explicitly rejected as post-hoc tuning.

## Limitations

Four human-keyed scenes, one dataset (Replica), one reconstruction
source; C1/C2 inject oracle labels/surfaces respectively as declared
evaluation scaffolding; the C2 vocabulary is closed-set (declared leak);
near-wall questions are membership-only; frl_apartment_0 remains
plausibility-only; the compiler is a rules engine with a known
paraphrase-brittleness surface (measured separately).

## Artifacts

- `python3 tools/mvp_demo.py` — deterministic A/B/C1/C2 report vs human
  keys (byte-identical, hard reference checks).
- `python3 tools/mvp_viewer.py` — self-contained 3D evidence viewer
  (office_0 + room_2), owner-accepted; walkthrough script in
  `docs/mvp_walkthrough.md`.
- Reproduction manifest: `docs/reproduction.md`; artifact pins:
  `docs/c1_artifact_manifest.json`, `tools/replica_scenes.lock.json`,
  `eval/predictions/phase8_c2/`.
- Release tag: `mvp-v1.0` (this freeze).
- Canonical tests: `python3 tools/run_tests.py` (63/63 files).

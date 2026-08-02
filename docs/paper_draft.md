# Where Failure Moves: An Evidence-First Spatial-Graph Reasoner for Captured 3D Scenes

*Paper draft converted from `docs/results_narrative.md` at
`paper-results-v1.0`. Structure follows a standard short-paper layout;
port into the venue template once one is chosen. Every number is
committed and hash-pinned in the repository; section 8 (related work)
is deliberately left to the author — no citations are invented here.*

## Abstract

We present a spatial question-answering system over captured indoor
scenes whose primary contribution is an evidence architecture rather
than a headline accuracy claim. The system builds a typed spatial scene
graph (support, wall contact/proximity, attachment, directional) and
answers structural questions through a compile–execute–verbalize
pipeline with four honest outcomes: answer, grounded empty, defer, and
unknown. Evaluation is organized as an input ladder in which each
variant changes exactly one upstream stage, scored against
human-verified answer keys that describe physical reality rather than
system output, under predeclared experimental protocols that are
allowed to fail. Across four human-keyed Replica scenes, this
discipline yields a complete causal account of failure: instance
proposal coverage is initially binding; a multi-view 2D-to-3D proposal
generator removes that bottleneck under predeclared gates (entity
viability 20/53 → 33/53, transferring to two further scenes); with
proposals fixed, an oracle-guided ceiling shows the bottleneck has
moved to relation semantics (recall 0.245 → 0.265 despite 31/53
entities at precision 1.00); and a frozen semantics revision shows
support relations are representable (5/20 → 16/20 at precision 0.94 on
the development scene) while attachment is unrecoverable from the
dataset's annotation-box geometry (11/14 keyed fixtures lie behind the
annotated wall planes). Six negative results are committed as
first-class artifacts. The system identifies where failure moves —
across perception, relation semantics, and annotation quality — without
disguising negative results as improvements.

## 1. Introduction

Spatial question answering over captured 3D scenes fails for many
reasons at once: reconstruction error, instance segmentation, label
assignment, relation geometry, language grounding, and the quality of
the evaluation targets themselves. A single end-to-end score cannot say
which. This work asks a narrower question with a stricter method: **can
a system be built so that every failure is attributable to exactly one
stage — and can that attribution survive contact with human-verified
ground truth?**

Contributions:

1. **A stage-isolated input ladder** (oracle boxes → mesh-derived boxes
   → learned instances → learned labels) over one frozen graph and
   reasoner, in which adjacent comparisons attribute answer changes to
   exactly one upstream stage (§3, §4).
2. **An evaluation discipline** — human-verified keys recording
   physical reality, predeclared gates with hard stopping rules,
   hash-pinned inputs, deterministic byte-identical reports, and
   negative results committed as artifacts (§3, §6).
3. **A gate-passing multi-view proposal generator**: SAM 2.1 masks over
   40 deterministic renders of the raw mesh, lifted through per-pixel
   vertex-id buffers and fused by cross-view co-membership, recover
   exactly the small/wall-mounted objects 3D geometry backends never
   propose (§5.1).
4. **A measured causal chain of bottleneck relocation** — from
   perception to relation semantics to annotation geometry — with each
   hand-off measured rather than assumed, including the demonstration
   that relation-specific gates prevented a misleading aggregate pass
   (§5).

## 2. System

Given a captured scene, importers produce a typed artifact bundle
(entities with boxes and labels; structural floor/wall/ceiling
surfaces; a gravity-canonical frame). Relation extractors emit typed
edges (support on floor and furniture, wall contact and proximity,
attachment, directional). A rules compiler maps natural-language
structural questions to graph queries; an executor evaluates them; a
verbalizer renders answers with citations. The reasoner never
fabricates certainty: outcomes are answer / grounded-empty / defer /
unknown, and "empty" explicitly does not claim absence in the scene.
All downstream components are frozen across every experiment reported
here.

## 3. Evaluation methodology

**The ladder.** Variant A uses the dataset's oracle boxes and labels;
B re-derives boxes from the semantic mesh (frame parity with A frozen
and regression-tested); C1 uses learned class-agnostic instances from
the raw mesh with labels injected through exact vertex correspondence
(declared evaluation scaffolding); C2 replaces those labels with
zero-shot CLIP predictions on rendered instance views. Each transition
holds everything else fixed.

**Keys.** Four scenes carry human-verified answer keys reviewed against
the raw RGB mesh. Keys record what is physically true — a key the
system fails is a successful review. Recall is computed only over
questions the reviewer marked exhaustive; membership-only questions are
excluded from micro-metrics by construction.

**Protocol discipline.** Every experiment predeclares its gates,
budget, and stopping rule before any number exists; failed gates stop
the experiment with remaining budget unspent; constants are never
adjusted after contact with results. Inputs are hash-locked, model
checkpoints sha-pinned, and the headline reports are byte-identical
across runs with hard consistency checks against committed reference
values.

## 4. Ladder results

![UID micro-recall by scene and variant](assets/fig_ladder_recall.svg)

Precision stays high down the ladder (C1 rows reach precision 1.00
where they cite at all); recall attributes cleanly:

![room_2 recall attribution](assets/fig_room2_attribution.svg)

On room_2, box source costs nothing (0.408 → 0.408), instance
extraction costs −0.163, and label learning costs a further −0.041 —
with the label loss traceable to a single support-anchor
misclassification (shelf → "vent"). Even variant A with perfect input
reaches only recall 0.29–0.41: whole-object AABBs cannot represent
seat surfaces, the 2 cm attachment band cannot express what humans call
"attached", and the support-class allowlist has measured gaps. That
representational ceiling motivates everything in §5.

## 5. The measured causal chain

### 5.1 Perception fixed: multi-view 2D evidence (C1-P1)

The measured C1 bottleneck was proposal coverage: only ~32% of oracle
entities had any viable 3D mask, selection over existing masks was
near-optimal, fragment unions added zero recall, and three oracle-free
selection-repair rules failed their gates. Introducing a new evidence
class — SAM 2.1 automatic masks over 40 deterministic point-splat
renders, lifted through per-pixel vertex-id buffers and fused by
cross-view co-membership over mesh edges — raised pooled entity
viability from 20/53 to 33/53 on the development scene and transferred
to both prospective scenes (+6 and +5 entities) with zero
scene-specific settings, under predeclared gates, at ~30 s of GPU per
scene. The recovered population is precisely the small and wall-mounted
clutter every 3D geometry backend missed: blinds, vents, switches,
wall-plugs, plates, lamps, vases at IoU 0.61–0.95.

### 5.2 The bottleneck moves: semantics become binding (C1-P2.0)

A predeclared ceiling measurement then asked what those proposals are
worth downstream: oracle-guided best-of-pool composition materializes
31/53 entities at QA precision 1.00 — and lifts human-keyed recall only
0.245 → 0.265. Of 13 newly recoverable entities, exactly one becomes a
citable answer; seven are attachment-key positives the frozen 2 cm
attachment semantics cannot cite even from perfect geometry (variant A
itself scores 1/14). The predeclared proceed rule stopped the composer
before a single parameter existed. Perception is no longer binding;
the representation layer is.

### 5.3 Semantics revisited: representable support, annotation-blocked attachment

A separately labeled semantics-v2 track (definitions frozen before
scoring; every key and frozen row preserved and hash-guarded) re-ran
variant A first. Its contained-rest support definition works on the
development scene — support answers 5/20 → 16/20 at precision 0.94 —
but is poorly calibrated across scenes (precision 0.36/0.58 on the two
transfer scenes). Attachment scored zero: a census of all 14 keyed
attached objects shows 11 lie *behind* the annotated wall planes
(signed gaps −0.16 to −1.23 m), the three contact-passing vents carry
0.37–0.62 m-deep annotation boxes, and one keyed vent sits 1.26 m from
every wall. "Attached" in the keys is a semantic judgment about wall
fixtures, not a box-proximity property recoverable from these
annotations. The frozen relation-specific gates stopped the track —
and notably, the aggregate gates alone (recall 0.612 at precision
0.94) would have passed; only the per-relation gates caught the
failure.

![room_2 per-relation hits, frozen A vs semantics-v2 A](assets/fig_semantics_v2_room2.svg)

## 6. Negative results as first-class artifacts

Committed with their predeclared protocols and unspent budgets: a
backend swap (failed 4/5 gates, stopped after one scene); three
oracle-free selection-repair rules (backend confidence measured
anti-correlated with mask quality); query-scoped proposal expansion
(zero recovered answers); an uncertainty-preserving proposal pool
(policy prototype only); the mesh-plane surface-estimator family
(input-contract failure, then acceptance-constant collapse, then a
read-only measurement proving no constant could work); and the
semantics-v2 track itself. Each failure narrowed the search space the
next experiment had to cover.

## 7. Limitations

Four human-keyed scenes from one dataset and one reconstruction source;
C1/C2 use declared evaluation-only oracle injections; no cross-dataset
or NeRF/3DGS claim. Annotation quality is itself a measured limitation:
the oracle wall planes disagree with captured geometry by >5 cm on 2 of
4 room_2 walls, and the keyed attachment answers are unrecoverable from
the annotation boxes — keys and scores were left unrevised to preserve
comparability, so those metrics carry stated target uncertainty. The
semantics-v2 support definition is reported with its cross-scene
precision collapse, not just its development-scene success.

## 8. Related work

*(Left to the author: 3D scene graphs, open-vocabulary 3D
segmentation, SAM-based lifting, benchmark-honesty/preregistration in
ML evaluation. No citations are fabricated in this draft.)*

## 9. Conclusion

By freezing every stage but one, scoring against human-verified
reality, and letting predeclared gates fail, the system localizes
failure precisely as it moves: proposal coverage (fixed by multi-view
2D evidence), then relation semantics (partly representable), then the
annotations themselves (partly not). We argue this style of measured
bottleneck relocation — including its six committed negative results
and one demonstrated case of relation-specific gates preventing a
misleading aggregate pass — is a reusable template for evaluating
compound 3D-language systems honestly.

## Reproducibility statement

All inputs are hash-locked; model checkpoints are sha-pinned; the
headline reports are byte-identical across runs and hard-checked
against committed reference values; the full pipeline, evaluators,
protocols with sign-off history, and an interactive 3D evidence viewer
are in the repository at tag `paper-results-v1.0` (reproduction
manifest: `docs/reproduction.md`; canonical test command:
`python3 tools/run_tests.py`, 71 test files at the freeze).

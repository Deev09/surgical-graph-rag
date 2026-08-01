# C2.0 — learned labels on matched instances (predeclared protocol)

Written 2026-08-01, BEFORE any label is computed. Per the C1 precedent this
is a measurement-first protocol: metric definitions and isolation rules are
frozen here; NO accuracy thresholds are set before the first numbers exist.
C2.0 starts only because the MVP-v0 demo slice was accepted (2026-08-01).

## Question

Holding instance boundaries fixed at the frozen C1 reference (Mask3D @0.2)
and structural surfaces fixed at variant A's, can a learned labeler replace
the oracle labels — and what does label error alone cost downstream?

## Scope: matched instances only

The evaluated set is exactly C1's matched entity instances (greedy
IoU-based correspondence, the same match table the C1 evaluator emits).
This scope choice uses oracle GEOMETRY to decide which instances are
evaluated — declared scaffolding: it isolates classification from
detection, which is the point of C2.0. Unmatched instances stay anonymous;
the structural-removal set is kept IDENTICAL to C1's (removal is by oracle
class — scaffolding, not a label decision). Consequence: the C1 and C2
bundles contain the SAME entities with the SAME geometry and surfaces;
the ONLY difference is `display_label` on the matched subset. Any
downstream answer delta is attributable to labels alone.

## Labeler (pinned)

- **Model**: OpenCLIP `ViT-B-32`, pretrained tag `openai`, run on CPU in
  eval mode under `no_grad` (deterministic); weights sha256 recorded in
  the run report at download time.
- **Input per instance**: the instance's vertices from raw `mesh.ply`
  (positions + vertex RGB), gravity-aligned with the same rotation the C1
  candidate builder uses. Three orthographic point-splat renders (top XY,
  front XZ, side YZ), 224×224, white background, depth-sorted 2 px
  splats, 8 % margin. Instance-only crops (no scene context) — declared.
- **Classification**: cosine similarity between the mean of the three
  image embeddings and per-class text embeddings averaged over three
  prompt templates: "a photo of a {c}", "a photo of a {c} in a room",
  "a 3D render of a {c}". Top-1 wins; full ranking recorded.
- **Vocabulary**: the sorted set of object `class_name`s of the SCENE
  (from `info_semantic.json`). This is closed-set zero-shot
  classification; giving the labeler the scene's class LIST (not any
  per-instance label) is a declared vocabulary leak, standard for
  zero-shot evaluation. Open-vocabulary C2.x would remove it.

## Isolation

The labeler sees ONLY: raw `mesh.ply` vertices/colors, the C1 bundle's
per-vertex instance ids, and the class vocabulary. It never sees
`mesh_semantic.ply`, per-instance oracle labels, answer keys, or any A/B
artifact. Oracle labels are used exclusively AFTER labeling, to score.

## Measurements (definitions frozen; no pass thresholds)

1. **Label accuracy** on matched entity instances, per scene: top-1 and
   top-3, plus a support-class confusion slice (table/desk/shelf/counter/
   stool/bench/sofa/chair/plant-stand vs everything else — anchor classes
   drive support questions downstream).
2. **Downstream QA delta**: the frozen MVP scorer over the human keys,
   C1 row (oracle labels) vs C2 row (learned labels), same scenes as
   MVP-v0 (room_1, room_2). Attribution is exact by construction (labels
   are the only difference).
3. **Support-anchor integrity**: for each support question, whether the
   anchor class resolves to the same entity set under learned labels.

## Scenes and order

room_2 first (best-characterized), then room_1. office_0/frl only as
label-accuracy rows (no human keys — no QA claims). Zero GPU: everything
runs locally on the saved bundles; the CLIP forward passes are CPU.

## Deliverables

- `segmenter/instance_render.py` (point-splat views), `segmenter/
  clip_labeler.py` (pinned backend), `tools/c2_run.py` (labels + accuracy
  + QA rows), `label_override` injection path in `segmenter/derived.py`
  (opt-in parameter; the frozen C1 path is byte-identical when unused).
- Report JSONs under `runs/phase8_c2/`; results recorded in this file.
- torch + open_clip_torch are OPTIONAL dependencies (labeler only), noted
  in requirements.txt; the frozen pipeline never imports them.

## Results (2026-08-01; reports in `runs/phase8_c2/`)

Pins as declared; weights sha256 `e6d1bd7789aa4519…` recorded in the
reports. One correction before recording: the first run used the
non-QuickGELU ViT-B-32 config against the openai weights (open_clip
warned); the correct `ViT-B-32-quickgelu` architecture is what these
numbers use (the mismatch had been accidentally favorable, 0.619 vs
0.571 top-1 on room_2).

| scene | matched entities | top-1 | top-3 | support-slice top-1 |
|---|---|---|---|---|
| room_2 | 21 | 0.571 (12/21) | 0.714 | **0.90 (9/10)** |
| room_1 | 22 | 0.500 (11/22) | 0.500 | n/a (0 owners) |

Downstream QA vs the human keys (identical entities/geometry/surfaces —
labels are the only difference). Metric naming corrected 2026-08-01:
**uid micro-P/R score UID/structural MEMBERSHIP** (the key cites uids, not
names); **semantic citation** scores whether uid-correct citations also
carry the canonical label — a uid-correct answer can verbalize the wrong
learned label, and this metric is where that shows (C1 = 1.0 by
construction, which doubles as a scorer self-check):

| scene | row | uid micro-P | uid micro-R | support hits | semantic citation |
|---|---|---|---|---|---|
| room_2 | C1 oracle labels | 1.00 | 0.245 | 2 | 1.00 (23/23) |
| room_2 | C2 learned labels | 1.00 | **0.204** | **0** | **0.619 (13/21)** |
| room_1 | C1 oracle labels | 0.571 | 0.114 | 0 | 1.00 (12/12) |
| room_1 | C2 learned labels | 0.571 | 0.114 | 0 | **0.500 (6/12)** |

room_1's "zero downstream delta" is therefore a MEMBERSHIP statement
only: half of its uid-correct citations verbalize a wrong label.

### Findings

1. **Support furniture classifies well; clutter does not.** The
   support-class slice is 9/10 on room_2 (tables/chairs/shelf-class
   objects have distinctive point-splat silhouettes); overall top-1 is
   dragged down by small/flat objects.
2. **The entire downstream cost came from ONE support-anchor error**:
   room_2's shelf misread as "vent" removed the shelf as a support
   anchor, erasing both support hits (R 0.245 → 0.204). room_1's eleven
   label errors cost exactly nothing — its support answers were already
   blocked by the allowlist gap, and no anchor changed. Downstream
   sensitivity is concentrated in the ~10 support-owner labels per
   scene, not the long tail.
3. **Vocabulary hygiene is a real, measured issue**: error mass flows to
   Replica's junk classes — 4 of room_2's 9 errors predicted "vent" and
   3 predicted "anonymize_picture" (an anonymization artifact class).
   Per the measurement-first rule the vocabulary was NOT cleaned after
   seeing results; a vocabulary-hygiene variant (drop non-object classes
   from the prompt set) is declared here as C2.1, to be run as its own
   labeled variant.
4. Precision was unharmed in both scenes (label errors on non-anchor
   objects change what an entity is CALLED, not which uids relations
   cite; the key scores uids).

### Honest summary (narrowed 2026-08-01)

Support-owner classification reached 9/10 on room_2 — but C2.0 did NOT
preserve room_2's support QA: the one shelf-label error removed both
existing support answers, and semantic-citation fidelity is 0.50–0.62
(half the uid-correct answers verbalize a wrong label). Overall error
remains dominated by C1 proposal coverage, consistent with the C1
closeout's prediction: label learning is not the binding constraint,
but it is not solved either.

### Status: C2 optimization STOPPED (owner decision 2026-08-01)

C2.1 (vocabulary hygiene) is NOT run: removing the junk classes would
not fix the shelf error (CLIP ranked "vent" above "shelf"; dropping
OTHER vocabulary entries does not change that ordering, and dropping
"vent" itself would be post-hoc tuning — vents are legitimate
attached-to-wall objects in the human key). C2.0 is integrated into
MVP-v0 as an explicitly EVALUATION-ONLY row via the committed prediction
sidecars (`eval/predictions/phase8_c2/`, pinned to the frozen ms02
bundles by output hash). Any subsequent performance experiment should
return to C1 proposal generation or new visual/geometric evidence — not
C3 and not vocabulary tuning.

### Frozen transfer extension: office_0 (2026-08-01)

After C2 optimization was stopped, office_0 received a human-verified key.
The unchanged C2.0 model, prompts, vocabulary construction, render path, and
ms02 bundle were therefore applied once as a transfer measurement — no
iteration and no reopening of C2.1.

| scene | matched | top-1 | top-3 | support-slice top-1 |
|---|---|---|---|---|
| office_0 | 16 | **0.250 (4/16)** | 0.500 | **0.286 (2/7)** |

Against the office human key, C1 and C2 both have uid micro-R 0.0 and zero
support hits on the exhaustive support rows: C1 coverage/downstream support
already supplies no answer for learned labels to preserve. Semantic citation
drops from C1's construction check 1.0 (16/16) to **0.3125 (5/16)** under C2.
Thus the earlier 9/10 room_2 support-owner result does not generalize to
office_0. Label learning remains non-binding for the delivered office QA only
because C1 has already removed the relevant support answers; it is plainly
not robustly solved. Optimization remains STOPPED.

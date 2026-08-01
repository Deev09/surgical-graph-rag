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
labels are the only difference):

| scene | row | micro-P | micro-R | support hits |
|---|---|---|---|---|
| room_2 | C1 oracle labels | 1.00 | 0.245 | 2 |
| room_2 | C2 learned labels | 1.00 | **0.204** | **0** |
| room_1 | C1 oracle labels | 0.571 | 0.114 | 0 |
| room_1 | C2 learned labels | 0.571 | 0.114 | 0 |

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

### Honest summary

Zero-shot CLIP on instance point-splats can carry the support-anchor
labels that drive this system's QA (9/10) but not the clutter tail
(0.5–0.57 overall). Since C1's coverage ceiling already dominates
end-to-end raw-PLY error, label learning is NOT currently the binding
constraint — consistent with the C1 closeout's prediction. C2.1
(vocabulary hygiene) and per-class prompt work are cheap declared
follow-ups; a C3 attempt remains blocked on C1 coverage, not on labels.

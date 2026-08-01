# MVP-v1 — interactive 3D viewer (owner-specified, implementation decisions recorded)

Owner instruction (2026-08-01, verbatim requirements): orbitable raw
mesh.ply; raw-RGB / semantic-instance overlay toggle; click any object →
actual Replica obj_N, label, and provenance; enter a question → highlight
cited mesh regions and relation evidence; A/B/C1/C2 selector showing how
answers change; human answer versus system answer; clear answer / empty /
unknown / defer states; explicit C1/C2 evaluation-only disclosures; one
offline command; no GPU compute, no model download. Initial scenes:
office_0 (exposes real generalization failures) and room_2 (strongest
C1/C2 contrast). Motivation: the office review showed the static PNGs can
obscure or misrepresent the scene; the actual RGB mesh must be the visual
foundation.

## Command

```
python3 tools/mvp_viewer.py            # -> runs/mvp_v1/viewer.html
```

Reads ONLY existing verified artifacts — the raw `mesh.ply`, the semantic
mesh (for per-vertex oracle ids), the frozen ms02 C1 bundles (hash-checked
against `docs/c1_artifact_manifest.json`), the committed C2 sidecars, and
the MVP-v0 deterministic JSONs (`runs/mvp_v0/<scene>_mvp.json` — run
`tools/mvp_demo.py` first). It computes NO new metrics: every answer,
citation, status, and verbalization shown is taken verbatim from the
MVP-v0 reports (the "a number in the viewer but not in the JSON is a bug"
rule carries over).

## Implementation decisions (recorded, not silently chosen)

1. **Point-based rendering of the raw mesh vertices, full resolution.**
   All 589k (office_0) / 722k (room_2) colored vertices are embedded and
   rendered as depth-attenuated points via a ~400-line hand-written WebGL
   renderer — no three.js, no CDN, no downloaded code; the single HTML
   file makes zero external requests. Faces are not rendered (Replica
   quad topology + no need: at this density the point cloud reads as a
   surface). Positions are uint16-quantized over the scene bbox
   (~0.1 mm resolution); colors are the raw vertex RGB.
2. **Question entry = the predeclared human-key question set** (picker
   with type-ahead). Free-text questions would require the Router at
   view time; MVP-v1 is a viewer over frozen, scored answers, not a live
   QA server. This mirrors the MVP-v0 rule that keys are the question
   set.
3. **Relation evidence** = for support questions, the anchor-class
   entities under the ACTIVE variant's labels are emphasized in blue
   (C2's learned labels change which objects are "the table" — visible
   directly), alongside the citation highlighting (hit green / wrong
   red / missed orange, everything else dimmed).
4. **Click-to-inspect** uses GPU id-picking; the info panel shows the
   Replica `obj_N`, oracle class, C1 match (pred id + IoU) or
   "unlabeled segment", the C2 learned label where present, and the
   provenance line for the active variant.
5. **Overlay modes**: raw RGB / oracle semantic instances / C1 learned
   instances (hashed instance colors; unassigned vertices grey).
6. **Outcome states** render as explicit badges with their honest
   meanings ("empty = graph holds no such relation — NOT proof of
   absence"; "defer = compiler/schema abstention").
7. **Disclosures** panel carries the MVP-v0 disclosure set plus the C2
   evaluation-only statement; the C1/C2 rows are labeled evaluation-only
   wherever they appear.

## Determinism & size

The generator is deterministic (byte-identical HTML for identical
inputs; no timestamps). Expected size ~25–30 MB (two embedded full-
resolution point clouds, base64) — a deliberate trade for a single
self-contained file that opens from disk with no server and no CORS
issues.

## Acceptance (owner)

Open `runs/mvp_v1/viewer.html`, orbit both scenes, toggle overlays,
click objects, walk the question set across A/B/C1/C2 vs the human
answer, and confirm the office_0 failures and room_2 C1-vs-C2 contrast
are legible. After acceptance: record the walkthrough, freeze/tag the
MVP release, prepare the results narrative — then (and only then) a new
performance protocol targeting C1 proposal generation.

## Sign-off

- [ ] Owner accepts the viewer (date: ______, by: ______)

# Phase 8 answer-key review guide

What to check, per scene and per panel, when promoting a draft key in
`eval/questions/phase8/drafts/` to a `human_verified` key. The mechanical edit
steps live in the `tools/draft_answer_key.py` docstring; this file is the
*judgment* side: what "correct" looks like in each scene, and which scenes are
worth your time.

**A slanted or absurd review PNG is usually a real finding, not a broken
picture.** The renderer draws exactly what the graph believes. Four of the
five scenes originally failed for measured geometry reasons — recorded as
findings F1–F4 below. F1, F2, and F4 are fixed and all drafts/PNGs are
regenerated; what remains open is the per-object half of F3 (see the
office_0 row).

## Scene status (measured 2026-07-12, after F1/F2/F4)

All drafts and PNGs regenerated 2026-07-12 with every fix in — the JSONs now
record exactly what the current system answers.

| scene | status after fixes | verdict |
|---|---|---|
| replica_room_0 | human key promoted 2026-07-29; first real review exposed missed floor furniture, broader wall attachment, seat support, and cabinet support | **HUMAN VERIFIED** |
| replica_room_1 | human key promoted 2026-07-31; bed attachment rejected, missing floor bed and wall furniture recorded, bed/nightstand/cabinet support added | **HUMAN VERIFIED** |
| replica_room_2 | human key promoted 2026-07-31; window 60 moved from floor to wall contact/attachment, dining-table and cabinet/shelf memberships expanded | **HUMAN VERIFIED** |
| replica_office_0 | human key promoted 2026-08-01 from raw RGB mesh review; floating table/display wall contact, missed floor furniture, complete table contents, and plant-stand support recorded | **HUMAN VERIFIED, PARTLY NON-EXHAUSTIVE** — against-wall preserves the F3 residual and a visible purple floor object has no usable semantic uid |
| replica_frl_apartment_0 | floor snapped −0.28 m; sheet + draft regenerated | PARTIAL — floor/attached now sane; "near the wall" (178 objects) is trivially true for a multi-room scene — record membership only, never exhaustive |

Bottom line: **room_0, room_1, room_2, and office_0 have promoted human
keys; office_0 keeps floor/against/attachment non-exhaustive where the raw
mesh exposed missing annotation or F3 ambiguity. frl remains reviewable
except near-wall.** Pre-fix drafts are preserved outside
the repo as evidence of what F1/F2/F4 changed.

The metric for "is a sheet reviewable," if you want one: for each panel, could
you personally list what SHOULD be colored, and does the colored set roughly
match it? room_0 passes; the other four fail on at least two panels each.

### Findings behind the "DO NOT VERIFY" rows

- **F1 — rotated scenes break AABB geometry. FIXED 2026-07-12** (jointly with
  the room-level half of F3) via guarded yaw de-rotation in
  `demo/replica_habitat_import.py` (`_dominant_yaw_deg`, tests in
  `tests/importers/test_yaw_derotation.py`): the dominant room yaw is
  estimated from wall normals (length-weighted, 90°-symmetric stats) and
  folded into the import rotation only beyond a 5° guard. room_1 (+26.6°) and
  room_2 (−7.2°) de-rotate; room_0/office_0/frl/apartment_0 import
  bit-identically and all frozen gates pass. After the fix, room_2's "rug
  against the wall" false positive is gone (its attached/against answers
  collapse to a vent — plausible), and room_1's sheets render axis-aligned.
  New review case surfaced: room_1 claims the *bed* is ATTACHED_TO the wall
  (its box bottom sits ~4.5 cm above the calibrated floor, above the 2 cm
  contact band, so the floor-support disqualifier doesn't fire).
- **F2 — floor plane height off in room_1 (~10 cm) and frl_apartment_0
  (~30 cm). FIXED 2026-07-12** via guarded floor calibration in
  `demo/replica_habitat_import.py` (`_calibrate_floor_planes`, tests in
  `tests/importers/test_floor_calibration.py`): the plane snaps to the median
  of the lowest objects over it, but ONLY when they penetrate beyond a 0.10 m
  guard, so room_0/room_2/office_0/apartment_0 import bit-identically and all
  frozen gates still pass. After the fix: room_1 "on the floor" 1 → 7 sensible
  objects (the *window* false positive is gone); frl 2 → 38, and the *table*
  "attached to the wall" false positive is gone. The *book* ATTACHED_TO cases
  remain (documented furniture-rest limitation, not F2). Drafts/PNGs are NOT
  regenerated yet — regenerate once F3/F4 also land, so the diff shows all
  fixes at once.
- **F3 — the importer discards object orientation. ROOM-LEVEL HALF FIXED
  2026-07-12** by the F1 yaw de-rotation (objects and walls now share an
  axis-aligned frame, so room-aligned furniture gets tight AABBs — room_2's
  rug shrank from 3.45 × 2.97 m to its true footprint). The per-object half
  remains open: `bbox_obb` is still `None` from this importer, so an
  individual object angled relative to its room (or office_0's genuinely
  non-rectangular 17° wall) still inflates. Full fix would thread OBB support
  through the wall-contact/proximity geometry — a wide change to frozen
  phase 5/6 code (the NEAR⊇CONTACTS subset guard assumes one geometry), so it
  should be its own carefully-gated phase, not a quick patch.
- **F4 — room-scale flat objects (rugs) poison structural questions.
  FIXED 2026-07-12** via an opt-in exclusion in the two wall-relation
  extractors (`exclude_room_scale_flat` on `ContactsSurfaceConfig` and
  `SurfaceProximityConfig`; predicate `room_scale_flat` in
  `graph/relations/base.py`, tests in `tests/relations/test_room_scale_flat.py`).
  An entity whose footprint is ≥ 0.60 of the wall-bounded room XY extent AND
  whose box height is ≤ 0.20 m is never an against/near-WALL candidate
  (rejected as `room_scale_flat_excluded`); floor relations are untouched —
  a rug IS on the floor. The room denominator comes from the walls, not the
  floor label, because office_0's `floor` instance covers only ~25% of the
  real floor. Thresholds sit in a wide measured gap: the only object that
  trips across all six scenes is office_0's rug (frac **0.975**); the
  next-largest flat object anywhere is room_1's rug at 0.483, and room_1's
  bed (frac 0.911 but 1.27 m tall) is kept by the height gate. The flag is
  OFF by default (frozen P2/P5 behavior, config hashes byte-identical) and
  ON in the battery/review path (`demo/question_battery._runs()` +
  `demo/visualize_questions.py`); phase gates build default configs and are
  unaffected. Note `tools/threshold_sweep.py` still probes the default
  (exclusion-off) extractor semantics.

## How to identify which object is which

Three artifacts per scene, used together:

1. **UID index sheet** — `demo/<scene_id>_uid_index.png` (regenerate with
   `python3 demo/visualize_uid_index.py <room_dir> <scene_id>`): the same
   top-down map with EVERY object annotated by its obj number, plus a
   legend column of `obj_N → label → (x, y) centroid`. This is your
   ground-truth map of which box is which.
2. **Question sheet** — `demo/<scene_id>_questions.png`: the colored
   panels. Boxes here are unlabeled; match them to the index sheet by
   position (same projection, same orientation).
3. **Draft JSON** — each question's `candidate_labels` maps the cited
   `obj_N → class label`, so you can read an answer list in words without
   the map.

Review loop per question: read the cited uids + labels from the draft JSON
→ find each number on the index sheet to see WHERE it is → check the
colored panel for anything colored that shouldn't be (false positive → its
uid goes to `expected_must_not_contain`) or an obvious member left
uncolored (miss → add to `expected_must_contain`). When two same-class
objects are ambiguous (two chairs), the index sheet's number placement and
the legend centroid settle it.

## What each PNG panel means and what to check

Each panel is a top-down view; grey outlines are all objects, colored fills
are the objects the system cited for that question. Cross-reference UIDs with
the `candidate_labels` block in the draft JSON (it maps `obj_N → label`).

1. **"what is on the floor?" (green)** — every large piece of furniture you'd
   expect (sofas, tables, chairs, rug, bed) should be green. Check both
   directions: a green box that is actually elevated (wall art, a window) is a
   false positive → move its UID to `expected_must_not_contain`; an obvious
   floor-stander that is NOT green is a miss → add its UID to
   `expected_must_contain`. Only after you've decided for *every* plausible
   object, set `"exhaustive": true`.
2. **"what is against the wall?" (red)** — red boxes should visually touch the
   black wall lines. This band is 2 cm, so expect FEW answers; a red box in
   the middle of the room is wrong.
3. **"what is near the wall?" (yellow)** — 0.5 m band; expect many answers.
   Don't try to make this exhaustive (too many candidates); just scan for
   absurd ones (something in the room center) and record membership only.
4. **"what is on furniture?" (orange, blue outline = supporting furniture)** —
   orange items should sit inside/over a blue outline. Classic false positive:
   an object *beside* the furniture at the same height.
5. **Attached-to (in the draft JSON, not always a panel)** — should be only
   genuinely wall-mounted things (vents, blinds, sconces, pictures). Books
   and tables attached to walls are false positives.

## Per-scene things to look hard at

Current draft answers in words (full UID→label maps in each question's
`candidate_labels`; regenerated 2026-07-12 post-F1/F2/F4):

- **room_0** — human verified 2026-07-29. The reviewer excludes boundary
  door/pillars `obj_36`, `obj_46`, `obj_84` from floor support and adds cabinet
  `obj_2` plus sofa `obj_77`; adds both to against-wall. Wall attachment includes
  all nine windows plus planter/vase `obj_45` and picture `obj_63`. Furniture
  support uses gravity-resting physical contact, including sofa/chair seats,
  table members whose imported AABBs disagree, and cabinet `obj_2` as an owner.
  The promoted key adds a scene-specific cabinet question because the current
  Router defers that support class; this is a measured capability gap, not a key
  error.
- **room_1** — human verified 2026-07-31. Door `obj_6` is excluded from floor
  support and bed `obj_32` is added. Against-wall keeps basket `obj_4`, blinds
  `obj_27`, and bed `obj_32`, and adds both nightstands (`obj_8`, `obj_28`) plus
  cabinet `obj_36`; door `obj_6` and rug `obj_20` are explicit negatives.
  Attachment includes all four windows, all three blinds, and picture `obj_12`,
  while bed `obj_32` is the confirmed false positive. Scene-specific bed,
  nightstand, and cabinet support questions preserve the human furniture review
  even though those owner classes currently defer in the Router.
- **room_2** — human verified 2026-07-31. Window `obj_60` is explicitly not
  floor-supported; it is against and attached to the wall. Attachment includes
  all windows, blinds, vents, and picture `obj_58`. Dining-table support expands
  beyond vase `obj_14` and plate `obj_18` to lamps `obj_39`, `obj_55` and plant
  `obj_56`; the plant is rejected from the chair answer. Dataset `shelf`
  `obj_4` was visually identified as the cabinet, and its exhaustive supported
  membership expands from three vases to fifteen objects.
- **office_0** — human verified 2026-08-01 against the raw RGB mesh rather
  than the box sheet alone. Floor adds chair `obj_4`, sofa `obj_9`, and
  plant-stand `obj_23`; floating wall-mounted table `obj_12` and boundary
  door `obj_16` are explicit floor negatives. The visible purple floor
  object has no usable semantic instance, so floor stays non-exhaustive.
  Against/attachment add floating table `obj_12` and blue projector/display
  `obj_66`, reject degenerate `other-leaf obj_65`, and stay non-exhaustive
  because the 17° wall remains an F3 residual. Table support expands to
  `obj_1/2/6/13/28/44/64`; plant `obj_22` rests on stand `obj_23`.
- **frl_apartment_0** — floor: 38 (plausible sweep — spot-check *shoe*,
  *bike*, *stair*); against wall: tissue-paper/book/table/picture; attached:
  tissue-paper/book/picture — the *book* and *tissue-paper* are the
  documented furniture-rest limitation (resting on furniture flush with the
  wall, not floor-supported, so not disqualified). "Near the wall" cites 178
  objects — membership-only, never exhaustive, in a multi-room scene.

## Honesty rules (unchanged)

- A key measures *reality*, not the system: if the system is wrong, the key
  should disagree with it. A reviewed key that the system fails is a
  successful review.
- `exhaustive: true` per question only after checking every candidate object;
  that is what unlocks recall/P-R in `tools/scene_scorecard.py`.
- Promote by saving as `eval/questions/phase8/<scene_id>_qa.json` with
  `answer_key_type: "human_verified"`; never edit the draft in place to claim
  verification.

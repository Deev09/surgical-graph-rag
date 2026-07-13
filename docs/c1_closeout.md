# C1 closeout — OpenMask3D/Mask3D class-agnostic backend, four Replica scenes

Date: 2026-07-13. Backend: Mask3D mask stage via OpenMask3D `3bc3fc52`,
arbitrary-scenes checkpoint (`da4b68cb…`), A100, 55–86 s/scene inference.
All reports under `runs/phase8_c1/` (`ms02/` = frozen operating point).

## Verdict, stated precisely

**The C1 infrastructure is validated and reproducible; the Mask3D backend's
entity coverage is not sufficient for reliable raw-PLY question answering.**
Only the first half of that sentence is a "ready for C2" claim. Learned
labels (C2) cannot recover absent entities, so any end-to-end C2 score
would remain dominated by C1's coverage ceiling. C2 may still be prototyped
on matched instances to isolate semantic classification.

## Headline numbers (frozen MIN_SCORE=0.2, min_vertices=20)

| scene | entity matches @IoU0.5 | answer recall vs B | support-answer recall | support-OWNER R@0.5 |
|---|---|---|---|---|
| office_0 | 12/47 (0.26) | 0.40 | 0.00 (0/5) | 1.00 (7) |
| room_2 | 18/53 (0.34) | 0.39 | 0.17 (1/6) | 1.00 (10) |
| room_1 | 17/45 (0.38) | 0.42 | n/a (0 owners) | n/a |
| frl_apartment_0 | 49/194 (0.25) | 0.51 | 1.00 (2/2) | 0.79 (19) |

Reading rules (the corrections that produced these columns):
- *entity matches @IoU0.5* is detection-style recall over entity classes —
  NOT the greedy any-overlap match count, which includes structural classes.
- *support-OWNER recall* measures recovery of the supporting furniture
  (tables/chairs/shelves), not the items on them. Supported-item recovery is
  the *support-answer recall* column — near zero in the dense single rooms.
- Matched-instance boundary quality is high (median matched IoU 0.80–0.89)
  but describes only the found objects; it does not offset the missing tail.

## Failure classification (zero-GPU, frozen 0.2; 339 oracle entities total)

| scene | recovered | merged | lost_by_resolver | no_raw_proposal |
|---|---|---|---|---|
| office_0 | 12 | 13 | 0 | 22 |
| room_2 | 18 | 22 | 1 | 12 |
| room_1 | 17 | 20 | 0 | 8 |
| frl_apartment_0 | 49 | 87 | 4 | 54 |
| **total** | **96 (28%)** | **142 (42%)** | **5 (1.5%)** | **96 (28%)** |

NOTE (added 2026-07-13): failure_class is the COMPOSITION-stage outcome —
`merged` is assigned before raw viability is checked, so these counts are
not proposal-coverage statistics. The orthogonal cut (report fields
`raw_proposal_recall_at_iou` / `n_viable_raw_at_05`, schema v2): Mask3D has
a viable individual raw mask for 13/47, 20/53, 21/45, 53/194 entities
(107/339 ≈ 32%), of which only 11 total are lost by composition — its
selection stage is nearly optimal; its ceiling is proposal coverage.

At the frozen operating point the dominant failure is **merging** — objects
absorbed into a neighbor's winning mask (worst in the multi-room frl, 87/194)
— with **no viable proposal** second. Direct resolver loss is small (~1.5%),
and six merged objects DID have viable raw masks (office_0's desk organizer:
raw IoU 0.824 → dense 0.078; counted as
`n_merged_with_viable_raw_proposal`). Precisely stated: the current
**model-plus-winner-takes-all-postprocessing** produces these merges. Score
thresholding cannot recover them, but a smarter resolution (soft assignment,
geometric splitting of multi-object masks) might recover the
viable-proposal subset and possibly more — what it cannot manufacture is
the 28% of objects with no viable proposal at all.

## Operating point: MIN_SCORE frozen at 0.2 (benchmark-definition choice)

Cross-scene sweep evidence (`*_resolve_sweep.json`): 0.1 and 0.2 identical
in office_0/room_1/room_2; support-owner recall flat across 0.0–0.5 in all
scenes (office/room_2 at 1.00 until their cliff, frl at 0.79); 0.3 loses a
room_1 entity; 0.4 loses a room_2 support owner. frl pays exactly one
entity@0.5 at 0.2 vs 0.1, in exchange for 14 fewer noise segments. Frozen
via `tools/c1_reresolve.py` (bundles in `runs/phase8_c1/bundles_ms02/`,
provenance-stamped, `instance_confidence` populated from raw scores).
This is an operating-point/benchmark-definition decision, **not** a model
improvement; the 0.4 Colab-side reports are retained for comparison.

## What C1 answered

Holding labels and surfaces oracle-correct, a class-agnostic learned
segmenter preserves furniture-level relational structure (support owners,
floor layout) with tight boundaries, while small-object relations (tabletop
contents, wall clutter) are lost predominantly by absence — merged or never
proposed — and to a smaller degree by distortion: **10 of the lost answer
memberships across the four scenes came from RECOVERED entities whose
learned box changed the relation** (1/2/1/6 per scene; e.g. office_0's
chair at IoU 0.93 leaving "on the floor"). Answer recall vs B sits at
0.39–0.51 per scene.

## Next decision — DECIDED 2026-07-13: Segment3D staged pilot (C1-M2)

Not another Mask3D tuning sweep: soft-NMS cannot split a mask that already
spans multiple objects, and only 5/339 entities are direct resolver losses —
thresholding is exhausted. Segment3D targets exactly the observed failure
(fine-grained class-agnostic masks; paper reports small-object AP50 15.9 vs
Mask3D's 4.5, though overall Replica gain is modest, 18.7 vs 18.0 — hence a
gated pilot, not an assumed win). Protocol, pins, and the PREDECLARED
room_2 stopping rule are frozen in docs/c1_m2_protocol.md — room_2 first,
frl only if room_2 passes the gate.

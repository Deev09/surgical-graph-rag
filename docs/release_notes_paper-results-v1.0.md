# Release notes — `paper-results-v1.0`

*Prepared for the GitHub release page (publish with
`gh release create paper-results-v1.0 -F docs/release_notes_paper-results-v1.0.md`
when ready). The `mvp-v1.0` tag marks the earlier demo freeze; this tag
closes the experimental arc.*

## An evidence-first spatial-graph reasoner for captured 3D scenes

This release freezes a complete, measured causal account of where
spatial-QA failure lives — and where it moves — across four human-keyed
Replica scenes:

- **Perception was binding → fixed.** A multi-view proposal generator
  (SAM 2.1 over 40 deterministic renders of the raw mesh, lifted through
  per-pixel vertex-id buffers, fused by cross-view co-membership) raised
  entity viability 20/53 → 33/53 and transferred to two further scenes
  (+6, +5) under predeclared gates — the project's first gate-passing
  performance experiment. Evaluation-only: no QA headline was claimed.
- **Then semantics were binding.** With proposals fixed, an
  oracle-guided ceiling showed composition could deliver 31/53 entities
  at precision 1.00 while lifting human-keyed recall only
  0.245 → 0.265. The predeclared rule stopped the composer before a
  single parameter existed.
- **Then the annotations themselves.** A frozen semantics revision
  showed support relations are representable (5/20 → 16/20 answers at
  precision 0.94 on the development scene, miscalibrated across scenes)
  while attachment is unrecoverable from the dataset's annotation boxes:
  11 of 14 human-keyed attached fixtures lie *behind* the annotated
  wall planes. Relation-specific gates stopped the track at the exact
  point where aggregate gates would have passed.

Six negative results ship as first-class artifacts with their
predeclared protocols and unspent budgets. Nothing here disguises a
negative result as an improvement.

## What's in the box

- The full pipeline: importers, typed relation extractors, graph
  builder, compile–execute–verbalize reasoner with honest outcomes
  (answer / grounded-empty / defer / unknown).
- Four human-verified answer keys and the review kit that produced them.
- Deterministic evaluation: `python3 tools/mvp_demo.py` (byte-identical
  reports, hard reference checks) and a self-contained interactive 3D
  evidence viewer (`python3 tools/mvp_viewer.py`).
- Every experimental protocol with its sign-off history and dated
  verdict; all inputs hash-locked; model checkpoints sha-pinned.
- Results narrative (`docs/results_narrative.md`), paper draft
  (`docs/paper_draft.md`), figures (`docs/assets/`), reproduction
  manifest (`docs/reproduction.md`).

## Verification at freeze

71/71 test files green · MVP determinism byte-identical over 6
artifacts · figures regenerate deterministically · all pinned Replica
inputs verified against the lock.

## Known limitations (measured, stated)

One dataset, one reconstruction source; C1/C2 rows use declared
evaluation-only oracle injections; oracle wall-plane annotations
disagree with captured geometry by >5 cm on 2/4 room_2 walls, and keyed
attachment answers are unrecoverable from annotation-box proximity —
keys were left unrevised to preserve comparability. Future work (D2
precision hardening; annotation-aware attachment) is identified and
deliberately unopened.

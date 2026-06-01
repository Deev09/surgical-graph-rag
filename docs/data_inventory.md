---
title: Data inventory
status: living document
date: 2026-05-31
tags: [data, phase-2, provenance]
---

# Data inventory

Phase 2 introduces a dependency on raw Replica data that is **not** checked into the repository. This document is the canonical record of:

- where Phase 2 expects raw inputs to live on disk,
- where the inputs come from,
- how to verify them,
- and which downstream tasks consume them.

The data payload lives under `data/` which is gitignored (alongside `runs/`). The pin record `tools/replica_inputs.lock.json` IS checked in — that file is how Phase 2 detects a drift between the data on your disk and the data the project was developed against.

---

## Replica room_0 (Phase 2 P2.01)

### Expected on-disk layout

```
data/
  replica/
    room_0/
      habitat/
        info_semantic.json      # Habitat-format semantic instance metadata
        mesh_semantic.ply       # Semantic mesh (vertex-colored by instance id)
```

The two files above are what `tools/verify_replica_inputs.py` checks by default. If your local Replica copy uses a different layout, place the files at the paths above or pass `--root` to the verifier.

### Provenance

- **Source dataset:** Replica Dataset by Facebook Reality Labs (Straub et al., 2019).
- **Distribution channel:** [facebookresearch/Replica-Dataset](https://github.com/facebookresearch/Replica-Dataset) — follow the upstream instructions for fetching. The dataset license sits at the upstream repo; do not redistribute by checking copies into this project.
- **Scene:** `room_0` (a small living-room scene with sofa, table, chairs, etc.). Phase 1 already used a pre-imported reduction of this scene under `scenes/replica_room_0/`.
- **Habitat split:** Phase 2 uses the Habitat-formatted assets (`habitat/info_semantic.json` + `habitat/mesh_semantic.ply`). The non-Habitat OBJ / mesh is not required.

### Acquisition workflow

1. Obtain the Replica dataset by following the upstream instructions.
2. Place the two expected files at `data/replica/room_0/habitat/...` (see layout above).
3. Run the verifier with `--init` once to pin sha256 + size:

   ```
   python tools/verify_replica_inputs.py --init
   ```

   This writes `tools/replica_inputs.lock.json` with the recorded facts and prints a short summary. The lock file gets committed to the repo and becomes the canonical Phase 2 provenance pin.
4. Every subsequent run uses the default mode to verify:

   ```
   python tools/verify_replica_inputs.py
   ```

   Exit 0 means "the data on disk matches the pinned record." Exit 1 means a missing file, a size mismatch, or a sha256 mismatch — Phase 2 is gated until resolved.

### What changes when you re-pin

`tools/replica_inputs.lock.json` is the dataset-version reference for Phase 2. If you intentionally update the underlying Replica copy (e.g. switching to a newer release), re-run `--init` to overwrite the pin and commit the change as a recorded provenance step. Phase 2 evaluation artifacts produced before vs after a re-pin are **not** byte-comparable; treat such a re-pin as a Phase 2 dataset-version bump and re-run the gates from scratch.

### Downstream consumers

| Task | Consumer | Path used |
|---|---|---|
| P2.02 | `importers/replica.py` (extended) — emits `bbox_obb` to `scenes/replica_room_0/enriched/v2/` | `data/replica/room_0/habitat/info_semantic.json`, `mesh_semantic.ply` |
| P2.03 | `importers/replica.py` (extended) — emits `structural_surfaces` with `source` provenance | same |
| P2.06 | `extractors/oracle_replica.py` (with `enriched_path=<v2 dir>`) | reads from `scenes/replica_room_0/enriched/v2/`, NOT from `data/` directly |
| All Phase 2 gates | `tools/phase2_gates.py`, `tools/phase2_exit_gate.py` | indirectly, via the v2 enriched bundle |

### Failure-mode notes

- **Verifier fails because the lock file is missing.** Run with `--init` after placing the data.
- **Verifier fails because of a sha256 mismatch.** Either the data was re-downloaded from a different release, or it was modified in place. Investigate before re-pinning — a silent re-pin would erase the project's record of which Replica release the Phase 2 artifacts came from.
- **The raw data cannot be acquired.** Per Q1 in `docs/phase2_plan.md`, there is no silent canonical fallback. Either pause Phase 2 or run the labeled experimental path (Q3 option C, `source="synth_bbox_fallback"`) — every artifact produced that way is tagged and excluded from the canonical Phase 2 exit gates.

---

## Notes for future entries

When Phase 3 or Phase 4 introduces additional raw datasets (e.g. ScanNet, Hypersim, a learned-backend training set), append a section here following the same shape:

- expected on-disk layout
- provenance (source, license)
- acquisition workflow + verifier
- downstream consumers
- failure-mode notes

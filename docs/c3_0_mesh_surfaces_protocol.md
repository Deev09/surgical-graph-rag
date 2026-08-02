# C3.0-S — raw-mesh structural surfaces (frozen protocol)

**Status: PREDECLARED AND FROZEN 2026-08-02 at preparation commit
`e5f77c7`. No estimator had run on a real scene at that commit. Gates,
parameters, and the implementation may not be changed mid-run.**

Written 2026-08-02 after MVP-v1.0 and before any raw-mesh surface-estimator
implementation.

## Why this is the next experiment

C1 and C2 still inherit floors, walls, and ceilings from the Replica oracle.
That dependency blocks any non-Replica or unlabeled capture even when the
`EntityArtifacts` contract and downstream graph/Router are dataset-agnostic.
This experiment replaces **only the structural-surface source** and measures
the result using data already present locally, with no GPU.

This is intentionally named **C3.0-S**, not C3. The input mesh is transformed
by the frozen Replica gravity/yaw frame, and boxes plus labels remain variant
B's oracle-isolated inputs. Therefore it measures surface estimation cleanly
but does not claim a fully raw or deployable pipeline. Raw gravity/frame
estimation would be a later C3.1 stage.

The previously drafted C1-P1 proposal experiment remains frozen and unrun
while C3.0-S is evaluated. This is a sequencing decision, not a cancellation
of C1-P1.

## Decision this experiment answers

Can one deterministic, class-agnostic geometry rule recover usable floor,
wall, and ceiling polygons from `mesh.ply` closely enough to preserve the
surface-dependent behavior of variant B across keyed Replica rooms?

Success means the oracle **surface source** can be removed under a held-fixed
frame. It does not mean raw-scene QA is good, that C1 coverage improved, or
that the current relation thresholds generalize to noisy phone/3DGS meshes.

## Baselines and scene roles

Every row uses the unchanged variant-B entity boxes and oracle labels. Only
`EntityArtifacts.structural_surfaces` changes.

| scene | oracle surfaces (floor/wall/ceiling) | role |
|---|---:|---|
| room_2 | 1 / 4 / 1 | development |
| room_1 | 1 / 4 / 1 | prospective transfer 1 |
| office_0 | 1 / 5 / 1 | prospective transfer 2 |
| room_0 | 1 / 5 / 1 | prospective transfer 3 |

Room_2's frozen variant-B human-key anchors are:

- `SUPPORTS_FLOOR`: 13 hits / 14 citations / 13 expected;
- `CONTACTS_SURFACE`: 1 / 1 / 2;
- `ATTACHED_TO`: 1 / 1 / 14;
- exhaustive surface-question micro-P = 15/16 = 0.9375;
- exhaustive surface-question micro-R = 15/29 = 0.5172.

`NEAR_SURFACE(wall)` is reported but excluded from human-key P/R because the
near-wall key is explicitly non-exhaustive. Furniture-support questions are
also excluded from this experiment's gate because their geometry does not use
structural surfaces.

`frl_apartment_0` is not part of this protocol: it has no human key and its
multi-room surface semantics are a separate test. `apartment_0` cannot be used
without new data because no raw `mesh.ply` is present locally. Claims that all
six Replica scenes can be tested without new data are therefore out of scope.

## Isolation boundary

Allowed at estimator runtime:

- raw `mesh.ply` vertices, RGB values, and triangle indices;
- the already-frozen gravity/yaw transform and z translation, supplied as an
  input value rather than recovered from semantic metadata;
- raw triangle normals, areas, edge adjacency, component geometry, and the
  fixed constants below.

Forbidden at estimator runtime:

- `mesh_semantic.ply`, `info_semantic.json`, object ids, class labels, and
  oracle structural surfaces;
- entity boxes, C1/C2 outputs, human keys, questions, graph edges, or answers;
- per-scene constants, visually selected patches, or scene-specific retries.

Oracle surfaces and human keys may be read only by the evaluator after the
estimated-surface artifact is finalized and hash-stamped. The generator must
have an I/O isolation test that rejects forbidden reads.

## Fixed estimator (`mesh_region_fit_v1`)

All calculations use float64 in the supplied canonical frame (`+z` is up).
Triangle winding is not trusted, so angular comparisons use the unsigned
normal axis `n ≡ -n`.

1. Parse every raw triangle. Drop degenerate faces with area below `1e-8 m²`.
2. Candidate orientation:
   - horizontal when `abs(n_z) >= cos(12°)`;
   - vertical when `abs(n_z) <= sin(12°)`;
   - all other faces are ignored.
3. Build face adjacency from shared undirected mesh edges. Deterministic region
   growth joins adjacent candidate faces of the same orientation family when:
   - unsigned normal-axis disagreement is at most `8°`; and
   - every new face vertex is within `0.02 m` of the component seed plane.
   Seeds are processed by ascending face index; neighbors by ascending index.
4. Refit each component plane by area-weighted PCA over its vertices. Reject a
   component when RMS point-to-plane residual exceeds `0.015 m`.
5. Horizontal components must have mesh area at least `1.5 m²` and projected
   area at least `1.0 m²`. Keep the lowest qualifying component as the floor
   and the highest qualifying component as the ceiling. They must be separated
   vertically by at least `1.8 m`; otherwise emit an explicit invalid result.
6. Vertical components must have mesh area at least `1.5 m²`. For each fitted
   normal axis, project all scene vertices onto that axis. Retain the component
   only when its plane offset lies within `0.15 m` of the 2nd or 98th percentile
   scene boundary along that axis. This boundary gate is intended to reject
   large cabinet and furniture sides.
7. A surface polygon is the largest deterministic boundary loop of the
   accepted component, projected to its fitted plane and simplified only by
   removing consecutive vertices within `0.01 m`. Holes are not filled. A
   component with no valid simple boundary loop is rejected, not replaced by
   a scene-sized rectangle or convex hull.
8. Deduplicate same-type surfaces when unsigned normal disagreement is at most
   `3°`, plane offset differs by at most `0.02 m`, and polygon IoU is at least
   `0.80`; retain the larger-area component, then lower seed-face index on a
   tie. Polygon IoU uses deterministic `0.01 m` plane-local cell centers (the
   same resolution already fixed for boundary simplification), with a hard
   five-million-cell safety failure rather than adaptive coarsening.
9. Emit deterministic UIDs ordered floor, walls by `(normal azimuth, offset,
   polygon digest)`, then ceiling. Provenance is
   `source="mesh_region_fit"`, estimator version `1`, input mesh SHA-256,
   frame-input hash, parameter hash, and output hash.

There is one parameterization and no sweep. These constants may be challenged
before sign-off; afterward they are frozen findings even if poorly chosen.

## Metric definitions

Surface geometry is compared after the estimated artifact is frozen:

- **compatible plane:** same type, unsigned normal error ≤ `10°`, and absolute
  plane offset error ≤ `0.05 m`;
- **oracle area coverage:** fraction of each oracle polygon covered by the
  union of compatible estimated polygons in that plane;
- **estimated spill:** fraction of estimated polygon area outside the union of
  compatible oracle polygons;
- floor/ceiling coverage is reported separately; wall coverage and spill are
  area-weighted across all wall polygons;
- split predicted patches may jointly cover one oracle wall; one giant patch
  spanning incompatible planes cannot.

The downstream row is named `B+S_mesh`. It is built from byte-identical B
entities/labels/frame plus only the estimated surfaces. Graph configs, Router,
human keys, scorer semantics, thresholds, and question wording remain frozen.
Older A/B/C1/C2 results remain comparable because no existing artifact or
evaluation definition is changed.

## Stage 0 — implementation validity (synthetic only)

Before any real-scene estimator run, tests must prove:

- ASCII and binary raw-triangle PLY fixtures parse to identical geometry;
- triangle normals, adjacency, region growth, PCA fit, boundary extraction,
  deduplication, and UID ordering are deterministic;
- a synthetic room with a floor, four walls, ceiling, table, and cabinet emits
  only the six structural surfaces;
- a tilted-wall fixture is retained and a large interior cabinet side is not;
- malformed/nonmanifold boundaries fail explicitly rather than becoming a
  bounding rectangle;
- the estimator cannot read semantic meshes, metadata, keys, entities, or
  answers;
- two identical runs are byte-identical and carry all required hashes.

Implementation bugs may be fixed while only synthetic fixtures have run. The
complete estimator, tests, parameters, environment, and room_0 input hashes
must then be committed before Stage 1.

## Stage 1 — room_2 development run

One estimator run is allowed. The raw output is finalized before oracle or key
evaluation. All gates must pass:

| gate | predeclared criterion |
|---|---|
| G1 | exactly 1 floor and 1 ceiling emitted; each compatible with its oracle plane |
| G2 | floor and ceiling oracle area coverage ≥ **0.85** each; spill ≤ **0.10** each |
| G3 | wall oracle area coverage ≥ **0.75**; wall spill ≤ **0.15** |
| G4 | median compatible-plane angular error ≤ **5°** and median offset error ≤ **0.03 m** |
| G5 | exhaustive surface-question micro-P vs the human key ≥ **0.90** (B: 0.9375) |
| G6 | exhaustive surface-question micro-R vs the human key ≥ **0.48** (B: 0.5172) |
| G7 | `NEAR_SURFACE(wall)` entity-membership F1 versus B ≥ **0.85** (diagnostic oracle-behavior parity, not human truth) |

If any gate fails: **STOP**, commit the negative result, do not run the three
transfer scenes, and do not alter constants to rescue the experiment.

## Stage 2 — frozen prospective transfer

Only after all room_2 gates pass, run the identical committed estimator once
on `room_1`, `office_0`, and `room_0`. Finalize all three artifacts before
opening any transfer oracle comparison. No code or parameter change between
scenes.

All gates must pass on **each** scene:

| gate | predeclared criterion |
|---|---|
| H1 | exactly 1 compatible floor and 1 compatible ceiling |
| H2 | floor/ceiling oracle area coverage ≥ **0.80** each; spill ≤ **0.15** each |
| H3 | wall oracle area coverage ≥ **0.70**; spill ≤ **0.20** |
| H4 | exhaustive surface-question human-key micro-P no more than **0.05** below that scene's frozen B row |
| H5 | exhaustive surface-question human-key micro-R no more than **0.05** below that scene's frozen B row |
| H6 | `NEAR_SURFACE(wall)` entity-membership F1 versus B ≥ **0.80** |

These are prospective estimator transfers, not untouched datasets: their
oracle surfaces, B results, and human keys already exist. Publication language
must not call them sealed cross-dataset generalization.

## Budget and stopping rule

- Zero GPU. One estimator parameterization.
- Maximum real-scene runs: one development + three transfer.
- No parameter versions, threshold sweeps, manual patch selection, or rescue
  run. A serialization-only replay may use the immutable intermediate surface
  artifact and must retain its hashes.
- Dev failure closes C3.0-S as a negative result with transfer unspent.
- Transfer failure is recorded; the surface estimator is not adopted.
- Full pass freezes `mesh_region_fit_v1` as the C3.0-S candidate and permits a
  separately predeclared C3.1 experiment for raw gravity/frame estimation.

Passing does **not** unblock a claim of good raw-scene QA: C1 proposal coverage
and the graph's AABB/contact/allowlist ceilings remain measured limitations.

## Required implementation and artifacts

- `geometry/mesh_surfaces.py`: pure deterministic estimator;
- raw-mesh parser local to the new path or a tested shared parser;
- `tools/c3_surface_run.py`: isolation runner and evaluator;
- synthetic tests plus a test that frozen B artifacts are unchanged;
- input/output/provenance manifest and per-scene estimated-surface sidecar;
- geometry coverage/spill table, plane errors, downstream question rows, all
  G/H gate verdicts, runtime, and peak memory;
- a dated verdict appended here. No hand-computed headline metrics.

Adding `mesh_region_fit` to the `StructuralSurface.source` validator is an
opt-in provenance-enum extension, not a benchmark-definition change. Existing
surface sources and bundles must remain byte-identical.

## Second-dataset sequencing

ScanNet A/B remains valuable after C3.0-S, but it is not a zero-friction first
step. The official dataset requires a signed terms-of-use agreement and access
request; then this project needs a ScanNet importer, label mapping, input
manifest, and a new human key. No ScanNet work or download is authorized by
this protocol.

If C3.0-S passes, the next generalization protocol should use one licensed
ScanNet validation scene to test importer/frame/threshold transfer before any
cross-dataset learned perception claim. If C3.0-S fails, its surface failure
class should inform that protocol rather than tuning on ScanNet.

## Sign-off

- [x] Project owner approves the staged C3.0-S interpretation, fixed
      estimator, development/transfer scenes, gates, and stopping rule
      (2026-08-02, project owner / deevyaswain — "approved—freeze C3.0-S").
- [x] Preparation-only freeze commit `e5f77c7` pins room_0 inputs and the
      complete implementation/environment before the first room_2 estimator
      run (66/66 canonical test files passed; no `runs/phase8_c3` existed).

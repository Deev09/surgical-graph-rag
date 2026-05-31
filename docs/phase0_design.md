# Phase 0 — Frozen Contracts

Status: draft for review
Owner: Deev
Last updated: 2026-05-30

This document pins the contracts the rest of the project depends on.
Phase 1+ is downstream of these choices; we over-think them now so we
under-touch them later.

If a contract here is wrong, the cost shows up later as either (a) modules
leaking across boundaries, (b) artifact-schema thrash after cached bundles
exist, or (c) a "swap" claim that doesn't survive a second backend.

---

## 0. Goals & non-goals

**Goal.** A pipeline whose middle is a *queryable spatial graph*, fed by a
swappable reconstruction + instance-extraction front end and consumed by an
AST-based reasoner. Relational questions are answered by structure over
explicit edges, not by label retrieval.

**Diagnosability is a first-class goal.** When the pipeline gets a question
wrong, we must be able to record one *primary* failure stage plus any
*contributing* stages, drawn from: reconstruction / representation / instance
extraction / graph construction / query compilation / execution /
verbalization. Real failures cascade — a missed instance can propagate into a
missing edge and then into wrong bindings. The attribution model captures
this; it does not pretend each wrong answer has exactly one cause.

**Non-goals for this doc.**

- Picking a specific reconstruction backend or instance extractor
  (decided at Phase 4 spike, gated as in §10)
- Picking an LLM provider for the L3 AST compiler (Phase 4)
- Fixing the surface syntax of the AST (Phase 4)
- Designing model training; we use pre-trained extractors only

---

## 1. Hard constraints

| Constraint | Implication |
|---|---|
| No local CUDA GPU | Any GPU-requiring stage (reconstruction, lifted instance extraction) runs on a remote pinned image and emits an immutable bundle, addressed by content hash. Local dev only loads bundles; never re-runs them. |
| Single developer | No distributed orchestration. A backend run is one script that emits one directory. |
| Production-shaped | Strict module boundaries. No `_internal` imports across stages. Every interface has a fake / oracle implementation usable in unit tests. Every stage emits structured diagnostics, never logs that the next stage parses. |
| Extend the existing benchmark | `benchmark/schema.py` and `benchmark/runner.py` are kept and fixed in place. We do not rebuild the question/runner/scoring trio from scratch. |
| Retire v1 as the primary path | `tiny_graph_demo.py`, `EXPECTED_ANSWERS`, and the hand-authored graffiti bathroom remain in git as a frozen regression fixture. They are *not* the oracle adapter's expected output. They are not the new benchmark either. |

---

## 2. Architecture: five stages, one query path

```
CaptureBundle
  │
  ▼
ReconstructionAdapter ──► SceneRepresentationBundle
  │                        (immutable artifact; loaded into runtime
  │                         SceneRepresentation which exposes render_view)
  ▼
InstanceExtractor ──────► EntityArtifacts
  │                        (object_uid, geometry, semantic hypotheses)
  ▼
GraphBuilder ──────────► SceneGraphBundle + BuildDiagnostics
  │                        (sparse typed edges, per-extractor evidence)
  ▼
QueryCompiler ──────────► AST
  │
  ▼
ASTExecutor ────────────► Bindings | empty | unknown | abstain | parser_failure
  │ (reads CompletenessProfile via ExecutionContext to choose empty vs unknown)
  ▼
Verbalizer ─────────────► Answer (NL string + structured trace)
```

Each arrow is a contract. Each stage emits diagnostics distinct from its
output so the next stage cannot accidentally depend on them.

**What each stage owns the right to fail at:**

| Stage | Owns | Cannot blame |
|---|---|---|
| ReconstructionAdapter | Geometry quality, pose accuracy, coverage | Missing labels (that's extraction) |
| SceneRepresentation | Rendering correctness, frame consistency | Object identity (that's extraction) |
| InstanceExtractor | What is an object, label hypotheses, geometry per instance | Edge correctness (that's the graph builder) |
| GraphBuilder | Which typed edges exist, their evidence | Whether a question is answerable (that's the compiler) |
| QueryCompiler | NL → AST, including parser-failure signaling | Graph contents |
| ASTExecutor | Faithful execution of the AST against the graph | Whether the AST was the right plan |
| Verbalizer | Bindings → NL string | Anything semantic |

`eval/` is the only place that converts these signals into pass/fail metrics.
No stage scores itself.

---

## 3. Identity model

The identity bugs in v1 (`tiny_graph_demo.py:155` re-keys edge targets to
labels, collapsing duplicates) trace to a single missing concept: an immutable
per-bundle object id distinct from display string. Fixed as follows.

```python
@dataclass(frozen=True)
class EntityIdentity:
    object_uid: str          # immutable within this EntityArtifacts bundle only
    display_label: str       # human-readable, may collide ("chair" x 3)
    aliases: list[str]       # alt strings the parser may match against
    source_instance_ref: str # the upstream backend's native id (debug only)
```

**Scoping rule.** `object_uid` is **immutable within the bundle that minted
it**. We do not promise cross-bundle identity (oracle vs learned, run #1 vs
run #2). Comparing two bundles requires an explicit correspondence:

```python
@dataclass(frozen=True)
class BundleCorrespondence:
    source_bundle_hash: str
    target_bundle_hash: str
    entity_pairs: list[tuple[str, str]]     # (source_object_uid, target_object_uid)
    surface_pairs: list[tuple[str, str]]    # (source_surface_uid, target_surface_uid)
    method: str                             # "iou_match" | "manual" | "shared_source_ref"
    score: dict[str, float]                 # per-pair match confidence; key = "kind:src->tgt"
    unmatched_source_entities: list[str]
    unmatched_target_entities: list[str]
    unmatched_source_surfaces: list[str]
    unmatched_target_surfaces: list[str]
```

Generated by tools in `eval/` (not by extractors). Required to run any
cross-bundle benchmark comparison.

**Where identity flows.** Edges reference `object_uid` only. `display_label`
appears in NL output and in `eval/` for the human-readable fallback when a
benchmark target's uid doesn't exist in the candidate bundle.

Benchmark schema already supports this: `ExpectedTarget` in
`benchmark/schema.py:51` has `canonical_id` + `display_label` + `aliases`. The
fix is on the consumer side — the new GraphBuilder and ASTExecutor honor
uids; the runner already does.

---

## 4. Spatial frame model

Replica's import notes (`capture_meta.json`) record `gravity_dir` because
axes matter. v1 silently assumed world axes for every relation. Fixed:

```python
@dataclass(frozen=True)
class SceneFrame:
    gravity: Vec3                    # unit vector, world frame
    canonical_forward: Vec3 | None   # room-relative +Y, if defined
    canonical_right: Vec3 | None     # room-relative +X, if defined
    units: Literal["meters"]
    notes: str
```

**Per-edge-type frame policy.**

| Edge type | Frame | Where derived |
|---|---|---|
| `ABOVE`, `BELOW` | world (gravity) | GraphBuilder, materialized |
| `ON_TOP_OF`, `SUPPORTS`, `INSIDE`, `CONTAINS`, `ATTACHED_TO` | world (gravity + contact) | GraphBuilder, materialized |
| `NEAR` | world (metric distance) | GraphBuilder, materialized |
| `FAR` | world | **query-time operator over an index, never stored** |
| `LEFT_OF`, `RIGHT_OF`, `IN_FRONT_OF`, `BEHIND` (canonical) | scene canonical axes, if defined | GraphBuilder, materialized only if `canonical_forward` and `canonical_right` are set |
| `LEFT_OF`, `RIGHT_OF`, `IN_FRONT_OF`, `BEHIND` (viewpoint) | viewpoint-relative | **derived at query time from a `Camera`; never stored** |

Default is world-frame, materialized. View-conditioned graphs are an opt-in
artifact for use cases that specifically need them (e.g. "from where I'm
standing, what's to my left"). The reasoner picks frame based on whether the
question's parse carries a camera-pose anchor.

Every materialized edge carries its frame in its evidence so a reader can
tell which axis convention produced it.

---

## 5. Per-stage contracts

### 5.1 `ReconstructionAdapter`

```python
class ReconstructionAdapter(Protocol):
    name: str
    version: str

    def reconstruct(
        self,
        capture: CaptureBundle,
        config: ReconstructionConfig,
    ) -> SceneRepresentationBundle: ...

    def capabilities(self) -> ReconstructionCapabilities: ...
```

```python
@dataclass(frozen=True)
class CaptureBundle:
    bundle_hash: str               # content hash of inputs
    scene_id: str
    images_dir: Path | None        # for image-based reconstruction
    poses: list[CameraPose] | None # COLMAP-style; None if to-be-estimated
    rgbd_dir: Path | None          # for RGB-D backends
    mesh_path: Path | None         # for oracle / pre-meshed paths (Replica)
    semantic_export: Path | None   # for oracle backends only
    notes: dict[str, JSON]
```

```python
@dataclass(frozen=True)
class ReconstructionCapabilities:
    produces_mesh: bool
    produces_pointcloud: bool
    produces_gaussian_splat: bool
    produces_nerf_field: bool
    estimates_poses: bool
    requires_gpu: bool
    typical_runtime_minutes: int   # for cost-aware orchestration
```

The adapter's output is a `SceneRepresentationBundle` (the on-disk artifact).
The runtime `SceneRepresentation` is a separate object that wraps a bundle and
exposes behavior (`render_view`, `query_geometry`). Diagnostics live on the
bundle; methods do not.

### 5.2 `SceneRepresentationBundle` and `SceneRepresentation`

**Why the split.** The bundle is content-addressed, immutable, and
serializable. Methods can't live on an immutable on-disk artifact without
inviting accidental drift between the serialized form and the runtime form.
The runtime wrapper exposes capabilities; the bundle is pure data.

```python
@dataclass(frozen=True)
class SceneRepresentationBundle:
    """Immutable on-disk artifact. Serializable as JSON + sidecar blobs."""
    representation_hash: str       # content hash; stable for (adapter, config, capture)
    scene_id: str
    frame: SceneFrame
    capabilities: RepresentationCapabilities
    geometry_handle: GeometryHandle   # opaque; deref via repr-specific loader
    poses: list[CameraPose]           # known cameras (training views or imported)
    diagnostics: ReconstructionDiagnostics
    notes: dict[str, JSON]


class SceneRepresentation(Protocol):
    """Runtime wrapper around a bundle. Not serialized. Constructed by a
    representation-specific loader given a bundle path."""
    bundle: SceneRepresentationBundle

    def render_view(self, request: RenderRequest) -> ViewBundle: ...
    def query_geometry(self, query: GeometryQuery) -> GeometryResult: ...
```

Downstream stages may accept either form: `InstanceExtractor.extract`
takes the runtime `SceneRepresentation` (because it may need to render);
pure-analysis tools that don't need rendering can read the bundle directly.

```python
@dataclass(frozen=True)
class RenderRequest:
    request_hash: str              # deterministic from fields below
    camera: CameraPose
    width: int
    height: int
    channels: list[Channel]        # which channels the caller wants
    feature_extractor: str | None  # required when channels include features

Channel = Literal["rgb", "depth", "normals", "semantic_features", "instance_features"]

@dataclass(frozen=True)
class ViewBundle:
    request: RenderRequest
    camera: CameraPose
    rgb: np.ndarray | None         # HxWx3 uint8
    depth: np.ndarray | None       # HxW float32, meters
    normals: np.ndarray | None     # HxWx3 float32
    semantic_features: np.ndarray | None   # HxWxF float32
    instance_features: np.ndarray | None   # HxWxF float32
    feature_extractor: str | None
    cache_key: str                 # f"{repr_hash}:{request_hash}"
```

```python
@dataclass(frozen=True)
class RepresentationCapabilities:
    renderable_channels: frozenset[Channel]
    supports_arbitrary_pose: bool   # vs only training views
    deterministic: bool             # MUST be True for caching
    typical_render_ms: int
```

**Caching.** Render results are content-addressable by `cache_key`. A
`ViewBundleCache` lives in `scenes/<scene_id>/views/<repr_hash>/<request_hash>.npz`.
Identical requests hit cache; non-deterministic representations (which we
forbid) would invalidate this and so MUST set `deterministic=False`, which
makes them unusable for cached extractors.

### 5.3 `InstanceExtractor`

```python
class InstanceExtractor(Protocol):
    name: str
    version: str
    required_channels: frozenset[Channel]   # validated against repr.capabilities

    def extract(
        self,
        repr: SceneRepresentation,
        config: InstanceExtractorConfig,
    ) -> EntityArtifacts: ...

    def capabilities(self) -> InstanceExtractorCapabilities: ...
```

```python
@dataclass(frozen=True)
class EntityArtifact:
    identity: EntityIdentity
    bbox_aabb: tuple[Vec3, Vec3]   # world frame, gravity-aligned
    bbox_obb: OrientedBBox | None
    centroid: Vec3
    geometry_handle: str | None    # path under EntityArtifacts.geometry_store
    semantic_hypotheses: list[SemanticHypothesis]  # ranked label candidates w/ confidence
    embedding: np.ndarray | None
    extraction_diagnostics: dict[str, JSON]  # e.g. coverage_score, multi-view consistency

@dataclass(frozen=True)
class SemanticHypothesis:
    label: str
    confidence: float
    source: str                    # "clip_text_match" | "open_vocab_seg" | "habitat_oracle" | ...

@dataclass(frozen=True)
class EntityArtifacts:
    bundle_hash: str               # f(repr_hash, extractor_name, version, config)
    scene_id: str
    frame: SceneFrame              # inherited from representation
    representation_hash: str       # what we extracted from
    extractor_name: str
    extractor_version: str
    entities: list[EntityArtifact]
    geometry_store_path: Path | None
    structural_surfaces: list[StructuralSurface]  # floors, walls, ceiling planes
    diagnostics: ExtractionDiagnostics
    notes: dict[str, JSON]
```

```python
@dataclass(frozen=True)
class StructuralSurface:
    surface_uid: str               # stable within this EntityArtifacts bundle; target of GraphRef(kind="surface")
    surface_type: Literal["floor", "wall", "ceiling"]
    plane: Plane                   # ax+by+cz+d=0
    polygon: list[Vec3] | None     # optional bounded extent
    confidence: float
```

Structural surfaces are first-class outputs (not entities), so support and
attachment relations can reference "the floor" and "the wall" without
inventing fake entities. The current importer drops these
(`importers/replica.py:88`); the new oracle extractor must re-emit them.

```python
@dataclass(frozen=True)
class InstanceExtractorCapabilities:
    label_vocab: list[str] | None  # None = open vocab
    provides_embeddings: bool
    provides_oriented_bboxes: bool
    provides_structural_surfaces: bool
    extractor_class_hint: Literal["furniture_only", "small_objects", "all"]
    # NOTE: extractor_class_hint is observational. It does NOT authorize the
    # executor to call a missing result "empty". That decision uses an
    # externally calibrated CompletenessProfile (§5.5) passed via ExecutionContext.
```

### 5.4 `GraphBuilder`

```python
class GraphBuilder(Protocol):
    name: str
    version: str
    extractors: list[RelationExtractor]   # one per relation family

    def build(
        self,
        entities: EntityArtifacts,
        config: GraphBuilderConfig,
    ) -> tuple[SceneGraphBundle, BuildDiagnostics]: ...
```

```python
class RelationExtractor(Protocol):
    name: str
    version: str
    edge_types: frozenset[EdgeType]

    def extract(
        self,
        entities: EntityArtifacts,
        config: RelationExtractorConfig,
    ) -> tuple[list[Edge], RelationExtractorDiagnostics]: ...
```

```python
@dataclass(frozen=True)
class GraphRef:
    """Typed reference into a SceneGraphBundle. Edges may terminate on either
    an entity (object_uid) or a structural surface (surface_uid). Treating
    surfaces as a distinct kind avoids fake "floor entity" hacks and makes
    ATTACHED_TO / ON_TOP_OF(floor) type-safe."""
    kind: Literal["entity", "surface"]
    uid: str

@dataclass(frozen=True)
class Edge:
    edge_id: str                   # f(extractor, source, type, target, version)
    source: GraphRef
    type: EdgeType
    target: GraphRef
    frame: Literal["world", "viewpoint", "scene_canonical"]
    weight: float                  # in [0, 1], extractor-normalized
    confidence: float              # in [0, 1]
    extractor: str
    extractor_version: str
    evidence: dict[str, JSON]      # the numbers the extractor used
    rejected_reason: None          # always None on emitted edges

@dataclass(frozen=True)
class EdgeRejection:
    source: GraphRef
    type: EdgeType
    target: GraphRef
    extractor: str
    rejected_reason: str           # "below_overlap_threshold" | "gap_too_large" | ...
    evidence: dict[str, JSON]

@dataclass(frozen=True)
class SceneGraphBundle:
    bundle_hash: str
    scene_id: str
    frame: SceneFrame
    entity_bundle_hash: str
    nodes: list[Node]              # one per EntityArtifact (selected attributes copied)
    edges: list[Edge]
    structural_surface_refs: list[str]

@dataclass(frozen=True)
class BuildDiagnostics:
    extractor_versions: dict[str, str]
    edges_emitted_per_type: dict[EdgeType, int]
    rejections_per_type: dict[EdgeType, int]
    rejection_samples: list[EdgeRejection]   # bounded sample for debugging
    runtime_ms_per_extractor: dict[str, int]
```

**Edge sparsity is structural, not advisory.** Each extractor declares its
own sparsity predicate (e.g. directional only emits when one axis dominates
by ≥ threshold AND inter-object distance < threshold). Diagnostics report
emitted/rejected counts so we can see density drift as code changes.

**FAR is not an extractor.** It lives in `ASTExecutor` as a query-time
operator backed by a spatial index over centroids.

### 5.5 `QueryCompiler`, `AST`, `ASTExecutor`, `Verbalizer`

The same executor runs single-hop templates and multi-hop compositions; the
only difference is AST complexity.

```python
class QueryCompiler(Protocol):
    def compile(self, question: str, scene: SceneGraphBundle) -> CompileResult: ...

@dataclass(frozen=True)
class CompileResult:
    ast: QueryAST | None
    outcome: Literal["compiled", "parser_failure", "out_of_schema"]
    compiler_name: str             # "rules_v1" | "llm_v1"
    notes: str
```

`QueryAST` (sketch — full grammar deferred to Phase 4):

```python
# A pattern is a conjunction of typed-edge constraints with variables,
# plus an aggregation over a bound variable.
QueryAST = Aggregation(
    op: Literal["ANY", "ALL", "COUNT", "EXISTS", "ENUMERATE"],
    bind: Variable,
    where: list[EdgeConstraint | NotConstraint | DistanceConstraint],
    frame_hint: Literal["world", "viewpoint"] | None,
    camera_anchor: CameraPose | None,
)
```

```python
@dataclass(frozen=True)
class CompletenessProfile:
    """Externally calibrated coverage priors for a pipeline configuration.
    Lives in eval/, attached to a (backend, extractor, scene) triple by a
    calibration run. Never produced by an extractor or builder."""
    source: Literal["oracle", "measured", "unknown"]
    entity_recall_by_class: dict[str, float]   # e.g. {"furniture": 0.92, "small": 0.41}
    edge_recall_by_type: dict[EdgeType, float]
    calibration_dataset: str | None            # provenance for the priors

@dataclass(frozen=True)
class ExecutionContext:
    """Everything the executor needs beyond the AST and graph."""
    completeness: CompletenessProfile
    empty_recall_threshold: float = 0.95       # below this, empty -> unknown


class ASTExecutor(Protocol):
    def execute(
        self,
        ast: QueryAST,
        graph: SceneGraphBundle,
        ctx: ExecutionContext,
    ) -> ExecutionResult: ...

@dataclass(frozen=True)
class ExecutionResult:
    outcome: Literal["bindings", "empty", "unknown", "abstain", "execution_error"]
    bindings: list[dict[str, GraphRef]]  # var -> entity or surface ref
    evidence: list[Edge]                 # edges that supported the bindings
    coverage_floor: float                # min(recall priors) across entity classes and edge types touched
    notes: str
```

**`empty` vs `unknown` is a load-bearing distinction.**

- `empty` = the graph executed the query and confidently asserts "no
  bindings satisfy this." An *oracle* graph can return `empty` and mean it
  ("there is no chair in this room").
- `unknown` = the executor cannot distinguish between "no match exists" and
  "the relevant entity or edge was likely missed by an upstream stage." A
  *learned* pipeline with recall < threshold for the touched classes /
  relations should return `unknown`, not `empty`, when no bindings are found
  — absence of evidence is not evidence of absence.

**The decision rule.** The executor reads `ctx.completeness` and consults
the recall priors for every entity class and edge type touched by the AST.
If the minimum prior across those is ≥ `ctx.empty_recall_threshold` (default
0.95), a no-bindings result is reported as `empty`. Otherwise it is reported
as `unknown`. `CompletenessProfile.source == "oracle"` short-circuits to
`empty`; `CompletenessProfile.source == "unknown"` short-circuits to
`unknown`. This keeps the rule explicit, auditable, and outside the
extractor's authority — per the §2 rule that stages do not score themselves.

```python
class Verbalizer(Protocol):
    def verbalize(
        self,
        question: str,
        compile_result: CompileResult,
        exec_result: ExecutionResult | None,
        scene: SceneGraphBundle,
    ) -> Answer: ...

@dataclass(frozen=True)
class Answer:
    text: str
    answered_by: Literal["rules_compiler", "llm_compiler", "verbalizer_abstain"]
    outcome: Literal["bindings", "empty", "unknown", "abstain", "parser_failure"]
    cited_uids: list[str]              # GraphRef.uid values (entity or surface)
    cited_edges: list[str]             # edge_ids
```

**Routing policy (final for Phase 0).**

```
QueryCompiler(rules) →
    compiled →                  ASTExecutor → (bindings|empty|unknown) → Verbalizer
    parser_failure →            QueryCompiler(llm) →
                                    compiled →           ASTExecutor → Verbalizer
                                    parser_failure →     Verbalizer abstains
                                    out_of_schema →      Verbalizer abstains
    out_of_schema →             Verbalizer abstains
```

Neither `empty` nor `unknown` triggers escalation. The LLM is invoked only to
compile NL into an AST when rules fail. The LLM never sees the graph as JSON
and never produces a free-text answer. Verbalizer maps `empty` to "nothing
matches" and `unknown` to "I don't have enough evidence to say" — these are
distinct user-facing answers.

---

## 6. Edge schema and relation semantics

Frozen meanings. Changing one of these requires bumping the bundle schema
version and rebuilding all cached graphs.

| Type | Meaning | Frame | Stored? |
|---|---|---|---|
| `LEFT_OF(a,b)` | a is to the left of b along scene-canonical right axis | scene_canonical or viewpoint | yes (canonical), no (viewpoint) |
| `RIGHT_OF(a,b)` | inverse of LEFT_OF | scene_canonical or viewpoint | yes/no as above |
| `IN_FRONT_OF(a,b)` | a is forward of b along scene-canonical forward axis | scene_canonical or viewpoint | yes/no as above |
| `BEHIND(a,b)` | inverse of IN_FRONT_OF | scene_canonical or viewpoint | yes/no as above |
| `ABOVE(a,b)` | a is above b along gravity, with optional xy-salience weight | world | yes |
| `BELOW(a,b)` | inverse of ABOVE | world | yes |
| `ON_TOP_OF(a,b)` | a's bottom face has horizontal footprint overlap with b's top face AND vertical surface gap ≤ ε | world | yes |
| `SUPPORTS(b,a)` | inverse of ON_TOP_OF | world | yes |
| `INSIDE(a,b)` | a's OBB is enclosed by b's OBB with margin ≥ ε | world | yes |
| `CONTAINS(b,a)` | inverse of INSIDE | world | yes |
| `ATTACHED_TO(a, s)` | a is mounted on structural surface s; one bbox dim ≈ 0 and contact-adjacent to s. `target` is `GraphRef(kind="surface")`. | world | yes |
| `NEAR_SURFACE(a, s)` | a's bbox is within near_threshold of structural surface s. `target` is `GraphRef(kind="surface")`. | world | yes |
| `NEAR(a,b)` | surface-to-surface distance ≤ near_threshold; symmetric | world | yes |
| `FAR(a,b)` | surface-to-surface distance ≥ far_threshold; symmetric | world | **no — query-time operator** |

`ABOVE` and `ON_TOP_OF` coexist as distinct semantics. A lamp can be ABOVE
a chair without being ON_TOP_OF it (no contact). The AST distinguishes them
explicitly.

---

## 7. Benchmark — fixes to apply in place

We keep `benchmark/schema.py` and `benchmark/runner.py`. The known bugs we
fix as part of Phase 1:

### 7.1 `count` and `yes_no` answer types are unscored

`benchmark/runner.py:128` (`score_output`) only compares
`output.answer_entity_ids` against expected targets. `RunnerOutput` carries
`answer_count` and `answer_yes_no` (lines 67–68) but they are never read.

Fix: when `Question.answer_type == "count"`, score by integer equality
against an `expected_count` field on the question (to be added to schema). When
`answer_type == "yes_no"`, score by boolean equality against an `expected_yes_no`
field. Schema bump: `SCHEMA_VERSION` "v0.1" → "v0.2".

### 7.2 `any_of_subset` is identical to `one_of`

`benchmark/runner.py:119–122`:

```python
if policy == "one_of":
    return len(expected_covered) >= 1
if policy == "any_of_subset":
    return len(expected_covered) >= 1
```

`any_of_subset` should additionally require that every answered entity is
in the expected set — i.e. no false positives among returned answers. Fix:

```python
if policy == "any_of_subset":
    return len(expected_covered) >= 1 and false_positives == 0
```

This requires threading `false_positives` (currently computed lower in
`score_output`) into `_policy_satisfied`.

### 7.3 Identity-on-target and Replica zones

Targets already use `canonical_id`; the new graph and reasoner honor it.
The runner needs no change here.

**Replica zones are deferred.** Replica has no zones or structural nodes
today (`scenes/replica_room_0/capture_meta.json:25` records `object_count: 73`
and `authored_relation_count: 0`; the importer sets `zone: null` on every
object). Phase 1 authors Replica questions **without zone targets**. The
`target_kind="zone"` path in the runner stays present for the frozen
graffiti-bathroom fixture only.

After structural surfaces land in Phase 2, the *zone use case* is served by
surface-aware relations (`ATTACHED_TO`, `NEAR_SURFACE`) — "what's on the
right wall?" becomes a query against `ATTACHED_TO(?x, wall_right)`, not
against a zone string. A versioned region layer may be added later as an
optional artifact for true zone semantics (kitchen / bedroom / etc), but
that's out of scope for Phase 1–3.

### 7.4 Comparability with the existing baselines under `baselines/`

Frozen pre-fix runs in `baselines/v1/`, `baselines/v1_paraphrase/`,
`baselines/v1_computed_relations/`, and scene-level evals in
`scenes/replica_room_0/eval/evaluation_table.v1.*.json` remain readable. Any
metric whose semantics change post-fix gets a new column name; old columns
are not silently reinterpreted.

---

## 8. GraphBuilder evaluation

Graph construction is a first-class evaluable component. Metrics live in
`eval/`, not in the builder.

```python
@dataclass(frozen=True)
class RelationGroundTruth:
    scene_id: str
    sampling_protocol: SamplingProtocol     # see §8.1
    labels: list[GroundTruthEdge]

@dataclass(frozen=True)
class GroundTruthEdge:
    source: GraphRef               # in the *oracle* bundle; entity or surface
    target: GraphRef               # in the *oracle* bundle; entity or surface
    type: EdgeType
    label: Literal["positive", "negative", "unknown"]
    stratum: str
    inclusion_probability: float
    rater_notes: str

class RelationEvaluator(Protocol):
    def evaluate(
        self,
        graph: SceneGraphBundle,
        truth: RelationGroundTruth,
        correspondence: BundleCorrespondence | None,
    ) -> RelationMetrics: ...

@dataclass(frozen=True)
class RelationMetrics:
    per_type: dict[EdgeType, PerTypeMetrics]
    overall: PerTypeMetrics

@dataclass(frozen=True)
class PerTypeMetrics:
    n_labeled: int
    n_positive: int
    n_negative: int
    n_unknown: int
    weighted_precision: float      # Horvitz-Thompson estimator over strata
    weighted_recall: float
    edge_density: float            # emitted edges / max possible pairs of that type
```

`unknown` labels are excluded from both numerator and denominator. Estimates
are stratified by `(stratum, inclusion_probability)` so candidate-heavy
sampling does not bias aggregates.

### 8.1 Sampling protocol (per relation family)

Recorded explicitly on each `RelationGroundTruth`:

| Family | Candidate predicate | Negative sample |
|---|---|---|
| Support (`ON_TOP_OF` / `SUPPORTS`) | horizontal footprint overlap > 0 AND vertical surface gap ≤ 0.20 m | uniform sample of pairs with no footprint overlap |
| Containment (`INSIDE` / `CONTAINS`) | OBB-A's centroid lies inside OBB-B AND volume ratio < 0.5 | uniform sample of pairs with no centroid containment |
| Proximity (`NEAR`) | surface-to-surface distance ≤ near_threshold | stratified sample of pairs at 2x, 4x, 8x threshold |
| Directional (world / scene-canonical) | stratified by (distance band, axis-dominance band); meaningful pairs may be > 1 m | within-band uniform |
| Attached (`ATTACHED_TO`) | min bbox dim ≤ 0.10 m AND adjacent to a structural surface | uniform sample of free-standing entities |

Each `GroundTruthEdge` carries its stratum and the inclusion probability of
its candidate set. Aggregate precision/recall reported with bootstrap CIs.

Hand-labeling effort target for Replica room_0: a few hundred labeled pairs
total, not exhaustive over the 73·72 ordered pairs.

---

## 9. Content-addressed remote artifact workflow

The "no local GPU" constraint isn't a workaround; it's part of the
architecture.

```
remote (Colab / RunPod, pinned image)
  ▼
ReconstructionAdapter ──► SceneRepresentation
  └── serialize to bundle_dir/<bundle_hash>/
        ├── manifest.json   (scene_id, adapter, version, config_hash, capabilities)
        ├── geometry/...
        └── poses.json
  ▼
upload bundle_dir/<bundle_hash>/ to artifact store (S3 / GCS / git-lfs)
  ▼
local
  ▼
download bundle by hash, mount under scenes/<scene_id>/artifacts/<bundle_hash>/
  ▼
InstanceExtractor (CPU-friendly variants) or InstanceExtractor (remote) →
  EntityArtifacts (also content-addressed, also uploadable)
  ▼
GraphBuilder / Reasoner / Verbalizer run locally on cached bundles
```

Bundle directories are immutable. Re-running a pipeline stage produces a new
hash; old hashes remain valid for old graphs. Pruning policy is out of
scope for Phase 0.

**`manifest.json` per bundle** (mandatory):

```json
{
  "bundle_hash": "...",
  "scene_id": "replica_room_0",
  "stage": "reconstruction" | "extraction" | "graph",
  "producer_name": "...",
  "producer_version": "...",
  "config_hash": "...",
  "inputs": {"prior_bundle_hashes": ["..."]},
  "capabilities": {...},
  "diagnostics_path": "diagnostics.json",
  "schema_version": 1
}
```

---

## 10. Backend selection — gates, not picks

We do not pick a reconstruction adapter or instance extractor in Phase 0.
We define what a candidate must demonstrate to be adopted.

All thresholds below are **provisional feasibility floors**, not final
adoption thresholds. The first spike measures against them; the actual
adoption thresholds are set after we see distributions on real data.

**Match rule for G2 and dependent gates.** A candidate entity matches an
oracle entity when *either* AABB IoU ≥ 0.3 *or* the candidate centroid lies
within a scale-aware distance of the oracle centroid (default:
`min(0.5 × oracle_bbox_diag, 0.30 m)`). The disjunction exists because
small-object AABB IoU is unstable at typical 3DGS extraction quality, and a
small detection that's clearly the right object should not be penalized for
imperfect bbox fit.

**Adoption gates (run on Replica room_0; oracle adapter as ground truth via `BundleCorrespondence`).**

| Gate | Metric | Provisional floor |
|---|---|---|
| G1 reconstruction coverage | fraction of oracle entity centroids within k nearest geometry samples | ≥ 0.90 |
| G2a furniture instance quality | recall AND precision over oracle entities in the furniture class | both ≥ 0.60 |
| G2b small-object instance quality | recall AND precision over oracle entities in the small-object class | **reported only, does not block the first spike** |
| G2c task-critical coverage | recall over the subset of entities referenced by the selected QA demo question set | **= 1.0 (strict)** |
| G3 label hypothesis quality | top-1 label match for matched entities | ≥ 0.50 |
| G4 graph reproduction | per-type edge precision and recall on the hand-labeled relation set | TBD when ground-truth labeling exists (Phase 3) |
| G5 end-to-end QA gap | top1 accuracy on directional category ≥ 0.7 × oracle adapter's | ≥ 0.7 |
| G6 runtime budget | per-scene reconstruction + extraction ≤ 60 min on one A10 / equiv | ≤ 60 min |

G2c is the gate that actually protects the demo: if any entity that a
benchmark question names is missed by the extractor, the question is
unanswerable and the spike result is uninterpretable. G2a is the
generalization floor; G2b is informational so we can see whether
small-object recall is the bottleneck without forcing a kill on the first
spike.

**Pre-registration requirement for G2c.** The exact question set used to
define "task-critical coverage" must be frozen and committed to git
**before** the first backend spike runs. The pre-registered set lives at
`eval/questions/g2c_demo.json` and includes the union of entities (by
`object_uid` in the oracle bundle) that those questions name. After the
spike, G2c is computed against exactly that pre-registered set. This
prevents post-hoc selection of easier questions to make a candidate look
better than it is.

Candidates we spike against gates (no commitment yet):

- Nerfstudio Splatfacto (reconstruction) + render-and-OpenMask3D (extraction)
- OpenSplat3D (reconstruction + extraction; instance features clustered in-place)
- Replica oracle adapter (reference; trivially passes all gates by construction)

We commit to one trio after the spike measures it.

---

## 11. Phase plan with measurable exit gates

| Phase | Scope | Exit gate |
|---|---|---|
| 0 (this doc) | Contracts | Review checklist (§14) signed off |
| 1 | Skeleton + Replica oracle adapter + GraphBuilder (two modes: `compat` and `sparse`) + ASTExecutor + rules QueryCompiler + Verbalizer + benchmark fixes from §7 | See Phase 1 detail below. |
| 2 | Importer extension: structural surfaces, world-frame OBBs, floor / wall / ceiling planes | Replica scene exports ≥ 1 floor, ≥ 2 wall, 1 ceiling planes; per-entity OBB in world frame; existing directional + proximity edges unchanged. |
| 3 | Support / containment / attached extractors + hand-labeled relation GT + RelationEvaluator | Per-type precision and recall reported with bootstrap CIs on the labeled set. No claim of "good" or "bad" — just measured. |
| 4 | Real reconstruction + extraction spike trio | At least one trio clears gates G1–G3 in §10 on Replica. Decision recorded. |
| 5 | Compositional AST + rules→LLM compiler fallback (D-policy from §5.5) | At least 5 multi-hop questions added to Replica benchmark; ≥ 70% of single-hop questions previously L1-answered are still answered after migration to the unified executor. |
| 6 | Per-category, per-backend benchmark with per-stage failure attribution | Failure attribution histogram populated; each scene-runner-backend combo emits `eval_table.json` + `summary.json`. |

### 11.1 Phase 1 detail — two GraphBuilder modes

Phase 1 ships the GraphBuilder with two explicit modes. Both run from the
same EntityArtifacts; they differ only in extractor configuration.

**`compat` mode** — legacy reproduction target.

- Extractors run with the same thresholds as `relations/compute.py` today
  (`MIN_DELTA = 0.3`, `NEAR_THRESHOLD = 1.0`, NEAR emitted symmetrically as
  both directions).
- Exit gate: the produced edge set is **exactly equal** (by edge key:
  `(source.kind, source.uid, type, target.kind, target.uid)`) to the union
  of `scenes/replica_room_0/computed_relations/scene_graph.json` and
  `baselines/v1_computed_relations/scene_graph.json`. No tolerance, no
  symmetric-difference allowance — this is a pure port of legacy logic and
  any drift indicates a bug in the port. The diff is written to
  `scenes/replica_room_0/eval/oracle_adapter_repro_diff.json` and must be
  empty for the gate to pass.
- **This 5,414-edge graph is a legacy reproduction target, not the desired
  graph.** Its purpose is solely to prove the new pipeline preserves prior
  behavior bit-for-bit when configured to do so — i.e. no silent regressions
  hiding inside the refactor.

**`sparse` mode** — the desired graph going forward.

- Extractors use relation-family-specific sparsity predicates from §5.4
  (directional only when one axis dominates by a stricter margin AND
  inter-object distance below a per-extractor threshold).
- **NEAR uses centroid-to-centroid distance provisionally in Phase 1.**
  Surface-to-surface distance depends on the world-frame OBBs and
  structural geometry that land in Phase 2; once that geometry is
  available, NEAR is recomputed against surface distance and the sparse
  graph is rebuilt.
- Symmetric edges (NEAR / FAR) are stored **once** with an evidence flag
  `symmetric: true`, not duplicated as both directions. Inverse pairs
  (LEFT_OF/RIGHT_OF, ABOVE/BELOW, ON_TOP_OF/SUPPORTS, INSIDE/CONTAINS) are
  stored **once** in a canonical direction; the executor derives the
  inverse at query time. Per-family edge counts are reported in
  `BuildDiagnostics`.
- Exit gate is the conjunction of:
  - **Determinism**: identical EntityArtifacts → identical SceneGraphBundle
    by `bundle_hash`.
  - **Scale-aware density smoke test**:
    `logical_edges / entity_count ≤ 14`, where `logical_edges` counts
    symmetric edges once and inverse pairs once. For Replica room_0
    (73 entities) this is ≤ 1,022 edges — roughly the < 1,000 figure but
    now scale-aware so it generalizes to other scenes.
  - **Per-family counts emitted** in `BuildDiagnostics.edges_emitted_per_type`
    for inspection. No per-family threshold is enforced in Phase 1; that
    is a Phase 3 deliverable once hand-labeled relation ground truth
    exists.
- The density rule is a **smoke-test guardrail**, not a quality metric. It
  will be replaced in Phase 3 by per-type precision and recall from
  `RelationEvaluator`.
- Sparse mode is the default for everything downstream of Phase 1.

**Shared Phase 1 exit conditions (both modes).**

- All schema round-trips (write → read → equal) pass for every dataclass
  in §3–§5.
- Benchmark fixes in §7.1–7.2 land; re-running existing Replica question
  sets emits cleanly under the new runner.
- No stage imports another stage's `_internal` module.
- The frozen graffiti bathroom v1 is loaded via a thin wrapper that
  populates a `SceneGraphBundle` directly from the hand-authored graph.
  **It is not produced by the oracle adapter** and is not part of the
  reproduction diff. It exists as a separate regression fixture that
  Phase 1+ pipelines must continue to answer correctly.

---

## 12. Retirement / preservation map

| Asset | Disposition |
|---|---|
| `tiny_graph_demo.py` | Frozen; no longer a CLI entry point. Wrapped by a thin loader that exposes `GRAFFITI_BATHROOM` as a `SceneGraphBundle` fixture for regression tests. |
| `EXPECTED_ANSWERS` (in `tiny_graph_demo.py`) | Frozen; not used in new benchmark. May seed a small `eval/questions/graffiti_bathroom.json` if useful for the fixture. |
| `baselines/v1/`, `baselines/v1_paraphrase/`, `baselines/v1_computed_relations/` | Kept in git, read-only. New eval can load them for historical comparison panels. |
| `scenes/replica_room_0/eval/evaluation_table.v1.*.json` | Kept in git, read-only. |
| `importers/replica.py` | Survives; wrapped by `extractors/oracle_replica.py`. Phase 2 work extends it (structural surfaces, world-frame OBBs). |
| `relations/compute.py` | Logic survives, moves under `graph/relations/directional.py` and `graph/relations/proximity.py`. Density and sparsity rules tightened (§5.4). |
| `scoring/spatial.py`, `scoring/v1.py`, `scoring/v2.py` | Spatial-XY salience logic moves into the directional / proximity relation extractors where it actually belongs. The "scoring" framing was a misnomer. |
| `parsers/llm_parser.py`, `parsers/dispatch.py`, `parsers/vocab.py` | Inputs to the new `QueryCompiler`. Retained verbatim until Phase 4 rewrites the routing. |
| `benchmark/schema.py`, `benchmark/runner.py`, `benchmark/categories.py` | **Kept and fixed in place** (§7). No replacement. |
| `eval_scene.py`, `eval_graph.py`, `eval_vlm.py`, `eval_paraphrase.py`, `strict_eval.py` | Retained for now; superseded one-by-one as `eval/runner.py` (new) takes over per-runner orchestration. Removal happens only after equivalence is demonstrated. |

---

## 13. Module layout (locks file paths for Phase 1)

```
adapters/
  base.py                              # ReconstructionAdapter Protocol, CaptureBundle, capabilities
  oracle_replica.py                    # Replica habitat semantic → SceneRepresentation
  splatfacto.py                        # Phase 4
  __init__.py

representations/
  base.py                              # SceneRepresentation Protocol, RenderRequest, ViewBundle, cache
  mesh.py                              # mesh-backed representation (Replica oracle path)
  gaussian_splat.py                    # Phase 4
  view_cache.py                        # content-addressed render cache
  __init__.py

extractors/
  base.py                              # InstanceExtractor Protocol, EntityArtifacts, StructuralSurface
  oracle_replica.py                    # Replica oracle entities + structural planes
  openmask3d.py                        # Phase 4
  open_splat3d.py                      # Phase 4
  __init__.py

graph/
  schema.py                            # SceneGraphBundle, Node, Edge, EdgeRejection, BuildDiagnostics
  builder.py                           # GraphBuilder
  relations/
    base.py                            # RelationExtractor Protocol
    directional.py                     # LEFT/RIGHT/ABOVE/BELOW/IN_FRONT/BEHIND
    proximity.py                       # NEAR (FAR is query-time, lives in reasoner)
    support.py                         # Phase 3: ON_TOP_OF, SUPPORTS, ATTACHED_TO
    containment.py                     # Phase 3: INSIDE, CONTAINS
  __init__.py

reasoner/
  base.py                              # QueryCompiler, ASTExecutor, Verbalizer Protocols
  ast.py                               # QueryAST, EdgeConstraint, Aggregation
  compiler_rules.py                    # rules-based NL → AST
  compiler_llm.py                      # Phase 4: LLM-as-AST-compiler
  executor.py                          # single executor for L1 templates + L2 compositions
  verbalizer.py                        # Bindings → NL
  router.py                            # the routing in §5.5
  __init__.py

eval/
  bundle_correspondence.py             # build BundleCorrespondence by IoU / shared_source_ref
  relation_ground_truth.py             # author/load RelationGroundTruth
  relation_evaluator.py                # RelationEvaluator → RelationMetrics
  runner.py                            # benchmark/runner.py wrapper that scores Reasoner outputs
  questions/<scene_id>.json            # Question payloads (existing schema.py format)
  __init__.py

scenes/<scene_id>/
  artifacts/
    <bundle_hash>/                     # immutable bundles per stage
  views/<repr_hash>/<request_hash>.npz # cached renders
  graph/<graph_hash>.json
  eval/
    oracle_adapter_repro_diff.json     # Phase 1 exit artifact
```

---

## 14. Review checklist — sign off before Phase 1 starts

Please push back on any of these before I write code:

1. **Five-stage decomposition** with `SceneRepresentation` owning the
   `render_view` capability (not a sixth stage). Right boundaries?
2. **`EntityIdentity` scoped per-bundle**, with explicit `BundleCorrespondence`
   required for cross-bundle comparison. OK?
3. **Spatial frame policy**: world-frame materialized; viewpoint-relative
   derived at query time; FAR is a query-time operator, never stored. Any
   case you want stored that I'm dropping?
4. **`EdgeType` set and semantics in §6.** Especially: ABOVE and ON_TOP_OF
   coexist as distinct relations. Any missing edge types you want present in
   the schema from day one even if extractors land in Phase 3?
5. **AST-only L3** (no free-text LLM answers; LLM compiles to AST or
   abstains). `empty` is a valid answer, not a trigger for fallback.
6. **Benchmark fixes in §7** (count/yes_no scoring, `any_of_subset` ≠
   `one_of`). Anything else broken in the existing runner you've already
   noticed?
7. **GraphBuilder emits `BuildDiagnostics`; metrics live in `eval/`.** No
   builder ever scores itself.
8. **Sampling protocol in §8.1** — stratified by relation family, inclusion
   probability recorded, `unknown` allowed. Any family you want sampled
   differently?
9. **Adoption gates G1–G6 in §10.** Are the thresholds right or are they
   strawmen?
10. **Phase 1 exit gate**: oracle adapter reproduces existing Replica
    `computed_relations` artifacts within a bounded diff; bathroom v1 is a
    separate regression fixture loaded via thin wrapper, not the oracle
    adapter's output. OK?
11. **Module layout in §13.** Anything you want renamed before files exist?
12. **Retirement map in §12.** Anything I marked "frozen" you actually want
    deleted, or anything I marked for retirement you want kept?

When all 12 are signed off (or amended), Phase 1 starts with:

- (a) Create `adapters/base.py`, `representations/base.py`,
  `extractors/base.py`, `graph/schema.py`, `reasoner/base.py` containing the
  dataclasses and Protocols from §3–§5.
- (b) Implement `adapters/oracle_replica.py`, `representations/mesh.py`,
  `extractors/oracle_replica.py` as the thinnest possible wrappers around
  `importers/replica.py`, with the explicit goal of byte-for-byte
  reproducing existing Replica artifacts.
- (c) Port `relations/compute.py` to `graph/relations/directional.py` and
  `graph/relations/proximity.py`, adding sparsity diagnostics.
- (d) Implement `reasoner/executor.py` and `reasoner/compiler_rules.py` for
  the existing single-hop question categories.
- (e) Apply benchmark fixes in §7 with `SCHEMA_VERSION` bump to "v0.2".
- (f) End-to-end test that proves the data flow with no stage importing
  another stage's internals, and that the Phase 1 exit gate passes.

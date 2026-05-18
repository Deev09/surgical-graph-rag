# surgical-graph-rag

> Scene-graph RAG for spatial reasoning — typed-relation pruning over a real captured scene.

LLMs waste context when reasoning about real scenes: they're handed every object, every coordinate, every label, when the question only needs the few entities in the right spatial relation. `surgical-graph-rag` builds a structured scene graph with typed spatial relations and prunes it *before* retrieval, keeping only the relevant subgraph for the question. Compact context goes to the LLM; the answer comes back grounded.

## What it does

The pipeline:

1. **Scene graph construction.** Each scene is captured as a set of objects with labels, zones, xyz positions, and attributes. Pairwise typed relations are computed: `LEFT_OF`, `RIGHT_OF`, `BELOW`, `ABOVE`, `IN_FRONT_OF`, `BEHIND`, `NEAR`, `ATTACHED_TO`.
2. **Question → relation pattern.** A natural-language question is parsed into the relation(s) it implies (e.g. "What is left of the sink?" → `LEFT_OF` anchored on `sink`).
3. **Typed-relation pruning + zone-aware retrieval.** The scoring pass walks graph edges matching the relation type, applies a spatial-XY salience term for `BELOW`/`ABOVE` (`lambda_xy=0.38`, `floor=0.05`), and prunes everything else.
4. **Compact JSON to the LLM (optional).** The pruned subgraph is serialized to a compact JSON payload — orders of magnitude smaller than the full scene — and passed to a local OpenAI-compatible model for natural-language answering. Set `SKIP_LLM=1` to skip the LLM call and return the pruned answer directly.

Two scenes ship in v1:

- `graffiti_bathroom` — 12 labeled objects, hand-authored relations, 10 benchmarked queries.
- `replica_room_0` — Replica room reconstruction with scene graph, computed relations, and visualization tooling.

Three baseline variants live under `baselines/`: `v1` (hand-authored relations), `v1_computed_relations` (relations derived from geometry), and `v1_paraphrase` (robustness under paraphrased queries).

## Example query

```
QUERY: What is left of the sink?

Pruned subgraph (excerpt):
{
  "anchor":     {"id": "obj_2", "label": "sink", "zone": "front_right"},
  "candidates": [
    {"id": "obj_1", "label": "toilet", "rel": "LEFT_OF", "score": 0.91}
  ]
}

Answer:   toilet
Expected: toilet   ✓
```

The structured question schema (`benchmark/questions/graffiti_bathroom.json`) records the relation category (`relative_position`, `proximity`, …), the answer type (`entity` vs `entity_list`), the ambiguity policy (`one_of` vs `all_of`), and aliases for each canonical target — so a paraphrased answer like "bowl" still resolves to "toilet."

## Quickstart

```bash
git clone https://github.com/Deev09/surgical-graph-rag.git
cd surgical-graph-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run the v1 benchmark with no LLM required
SKIP_LLM=1 python tiny_graph_demo.py --benchmark-only

# Full demo: prune subgraphs and call a local LLM through Docker Model Runner
#   export CONTEXT_SURGEON_OPENAI_BASE_URL=http://localhost:12434/engines/vllm/v1
#   export CONTEXT_SURGEON_MODEL=ai/llama3.1
python tiny_graph_demo.py
```

## v1 baseline

| scene             | n_queries | top1_accuracy | topk_recall | avg_false_positives_per_query |
| ----------------- | --------- | ------------- | ----------- | ----------------------------- |
| graffiti_bathroom | 10        | 1.0           | 1.0         | 0.0                           |

The 10-query benchmark is saturated. v1 demonstrates the pipeline works on a clean scene with hand-authored relations; the open question is how it holds up under harder cases.

## What's next (candidate experiments)

The `baselines/` directory already scaffolds two of these:

- **Paraphrase robustness** (`baselines/v1_paraphrase/`, `eval_paraphrase.py`) — does pruning quality degrade under reworded queries?
- **Computed relations** (`baselines/v1_computed_relations/`, `relations/compute.py`) — replace hand-authored relations with geometry-derived ones; measure the gap.
- **VLM grounding** (`eval_vlm.py`) — feed image patches into the candidate-scoring pass for scenes that don't have clean object labels.
- **Harder + more diverse scenes.** Currently 2; the scene schema is designed to ingest more.
- **Additional typed relations** — `INSIDE_OF`, `SUPPORTS`, `FACING`, etc.
- **Context-cost ablations.** Token savings vs. a no-pruning RAG baseline at equivalent accuracy.

## Repo layout

```
surgical-graph-rag/
├── tiny_graph_demo.py         # Main demo: scene graph → prune → compact JSON → optional LLM
├── eval_graph.py              # Graph-level eval
├── eval_paraphrase.py         # Robustness under paraphrased queries
├── eval_scene.py              # Scene-level eval
├── eval_vlm.py                # VLM-grounded eval (experimental)
├── strict_eval.py             # Strict-match scoring
├── scenes/                    # Per-scene assets (scene_graph.json, expected_answers.json, …)
├── benchmark/                 # Questions, runner, schema, categories, VLM client
├── baselines/                 # v1, v1_computed_relations, v1_paraphrase artifacts
├── scoring/                   # Scoring (v1, v2, spatial, geom, topk, filter)
├── relations/                 # Relation compute + compare
├── parsers/                   # Question → relation pattern
├── importers/                 # Scene ingestion
└── requirements.txt
```

## License

MIT — see [LICENSE](LICENSE).

## Status

v1 demo. The graffiti_bathroom 10-query benchmark saturates; harder evals are in progress (paraphrase, computed-relations, additional scenes). Issues and PRs welcome.

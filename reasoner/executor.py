"""ASTExecutor — phase0_design.md §5.5.

Phase 1: handles single-Aggregation ASTs with a single EdgeConstraint
whose operands are one Variable and one EntityRef. Returns bindings
plus the empty/unknown decision per the CompletenessProfile.

Storage-vs-query indirection:
  The graph may store edges in only the canonical direction (sparse
  mode) or in both (compat mode). The executor handles either by
  consulting the inverse mapping: a question for RIGHT_OF(?x, sink) is
  satisfied by stored LEFT_OF(sink, ?x). Symmetric types (NEAR) are
  matched on both directions.

empty vs unknown decision (per §5.5):
  - completeness.source == 'oracle'   → empty when no bindings
  - completeness.source == 'unknown'  → unknown when no bindings
  - completeness.source == 'measured' → empty iff min(touched recall priors)
                                        >= ctx.empty_recall_threshold,
                                        else unknown
"""
from __future__ import annotations

import re

from graph.relations.base import (
    CANONICAL_INVERSE_PAIRS, INVERSE_TO_CANONICAL, SYMMETRIC_EDGE_TYPES,
)
from graph.schema import Edge, EdgeType, GraphRef, Node, SceneGraphBundle
from reasoner.ast import (
    Aggregation, EdgeConstraint, EntityRef, Operand, QueryAST, Variable,
)
from reasoner.base import ExecutionContext, ExecutionResult


def _norm_label(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).replace("_", " ").strip().lower())


def _node_match_labels(node: Node) -> list[str]:
    """All labels under which the node can be matched: `node.label`,
    `attributes.display_label`, plus any aliases."""
    cands: list[str] = [str(node.label)]
    display = node.attributes.get("display_label")
    if display:
        cands.append(str(display))
    for a in node.attributes.get("aliases", []) or []:
        cands.append(str(a))
    return cands


def _resolve_entity_ref(ref: EntityRef, graph: SceneGraphBundle) -> list[GraphRef]:
    target = _norm_label(ref.label)
    matches: list[GraphRef] = []
    for node in graph.nodes:
        for cand in _node_match_labels(node):
            if _norm_label(cand) == target:
                matches.append(GraphRef(kind="entity", uid=node.id))
                break
    return matches


def _edges_matching(
    graph: SceneGraphBundle,
    *,
    edge_type: EdgeType,
    source_ref: GraphRef | None,
    target_ref: GraphRef | None,
) -> list[Edge]:
    """Find edges matching (source_ref, edge_type, target_ref), handling
    canonical-vs-inverse storage and symmetric types."""
    out: list[Edge] = []
    seen_ids: set[str] = set()

    def _push(e: Edge) -> None:
        if e.edge_id not in seen_ids:
            out.append(e)
            seen_ids.add(e.edge_id)

    def _scan(*, want_type: EdgeType, src: GraphRef | None, tgt: GraphRef | None) -> None:
        for e in graph.edges:
            if e.type != want_type:
                continue
            if src is not None and e.source != src:
                continue
            if tgt is not None and e.target != tgt:
                continue
            _push(e)

    # Direct storage of the requested type (compat mode has both; sparse
    # has only canonical).
    _scan(want_type=edge_type, src=source_ref, tgt=target_ref)

    # If the requested type is non-canonical, the canonical inverse may
    # be stored with endpoints swapped.
    if edge_type in INVERSE_TO_CANONICAL:
        canonical = INVERSE_TO_CANONICAL[edge_type]
        _scan(want_type=canonical, src=target_ref, tgt=source_ref)
    # If the requested type IS canonical, its non-canonical inverse may
    # also be present (compat mode).
    if edge_type in CANONICAL_INVERSE_PAIRS:
        inverse = CANONICAL_INVERSE_PAIRS[edge_type]
        _scan(want_type=inverse, src=target_ref, tgt=source_ref)

    # Symmetric types: also try swapped endpoints with the same type.
    if edge_type in SYMMETRIC_EDGE_TYPES:
        _scan(want_type=edge_type, src=target_ref, tgt=source_ref)

    return out


def _coverage_floor_for_query(ctx: ExecutionContext, edge_type: EdgeType) -> float:
    """Min of touched-class entity recall and the edge-type recall.
    Phase 1 approximation: treat all entity classes as touched because
    we cannot know which class would have answered the query."""
    cp = ctx.completeness
    if cp.source == "oracle":
        return 1.0
    if cp.source == "unknown":
        return 0.0
    # measured
    entity_recalls = list(cp.entity_recall_by_class.values())
    entity_floor = min(entity_recalls) if entity_recalls else 1.0
    edge_floor = cp.edge_recall_by_type.get(edge_type, 1.0)
    return min(entity_floor, edge_floor)


def _empty_or_unknown(ctx: ExecutionContext, edge_type: EdgeType, floor: float) -> str:
    cp = ctx.completeness
    if cp.source == "oracle":
        return "empty"
    if cp.source == "unknown":
        return "unknown"
    # measured
    return "empty" if floor >= ctx.empty_recall_threshold else "unknown"


def _operand_role(constraint: EdgeConstraint) -> tuple[Variable, EntityRef, str]:
    """Identify which operand is the Variable and which is the EntityRef,
    plus a 'role' tag: 'var_is_source' or 'var_is_target'. Raises if
    the constraint shape is unsupported in Phase 1."""
    if isinstance(constraint.source, Variable) and isinstance(constraint.target, EntityRef):
        return constraint.source, constraint.target, "var_is_source"
    if isinstance(constraint.target, Variable) and isinstance(constraint.source, EntityRef):
        return constraint.target, constraint.source, "var_is_target"
    raise ValueError(
        "Phase 1 executor requires exactly one Variable and one EntityRef "
        f"in the EdgeConstraint; got source={type(constraint.source).__name__}, "
        f"target={type(constraint.target).__name__}"
    )


class RulesExecutor:
    """Implements the ASTExecutor Protocol."""
    name: str = "executor_v1"
    version: str = "0.1"

    def execute(
        self,
        ast: QueryAST,
        graph: SceneGraphBundle,
        ctx: ExecutionContext,
    ) -> ExecutionResult:
        if not isinstance(ast, Aggregation):
            return ExecutionResult(
                outcome="execution_error", bindings=[], evidence=[],
                coverage_floor=0.0,
                notes=f"expected Aggregation, got {type(ast).__name__}",
            )
        if len(ast.where) != 1 or not isinstance(ast.where[0], EdgeConstraint):
            return ExecutionResult(
                outcome="execution_error", bindings=[], evidence=[],
                coverage_floor=0.0,
                notes="Phase 1 executor handles exactly one EdgeConstraint",
            )

        constraint: EdgeConstraint = ast.where[0]
        try:
            var, anchor_ref, role = _operand_role(constraint)
        except ValueError as e:
            return ExecutionResult(
                outcome="execution_error", bindings=[], evidence=[],
                coverage_floor=0.0, notes=str(e),
            )

        floor = _coverage_floor_for_query(ctx, constraint.type)
        anchor_refs = _resolve_entity_ref(anchor_ref, graph)
        if not anchor_refs:
            outcome = _empty_or_unknown(ctx, constraint.type, floor)
            return ExecutionResult(
                outcome=outcome, bindings=[], evidence=[],
                coverage_floor=floor,
                notes=f"anchor {anchor_ref.label!r} not found in graph",
            )

        bindings: list[dict[str, GraphRef]] = []
        evidence: list[Edge] = []
        seen_binding_uids: set[str] = set()

        for anchor in anchor_refs:
            if role == "var_is_source":
                source_ref = None
                target_ref = anchor
            else:  # var_is_target
                source_ref = anchor
                target_ref = None
            matches = _edges_matching(
                graph,
                edge_type=constraint.type,
                source_ref=source_ref,
                target_ref=target_ref,
            )
            for edge in matches:
                bound_ref = edge.source if role == "var_is_source" else edge.target
                # The matched edge might be the inverse stored with swapped
                # endpoints — re-derive which endpoint corresponds to the
                # variable by excluding the anchor uid.
                if bound_ref == anchor:
                    bound_ref = edge.target if bound_ref == edge.source else edge.source
                if bound_ref.uid in seen_binding_uids:
                    continue
                seen_binding_uids.add(bound_ref.uid)
                bindings.append({var.name: bound_ref})
                evidence.append(edge)

        if not bindings:
            outcome = _empty_or_unknown(ctx, constraint.type, floor)
            return ExecutionResult(
                outcome=outcome, bindings=[], evidence=[],
                coverage_floor=floor,
                notes=f"no edges of type {constraint.type} touch {anchor_ref.label!r}",
            )

        return ExecutionResult(
            outcome="bindings", bindings=bindings, evidence=evidence,
            coverage_floor=floor,
            notes=f"matched {len(bindings)} binding(s)",
        )

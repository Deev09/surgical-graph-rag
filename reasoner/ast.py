"""QueryAST and primitives — phase0_design.md §5.5.

Pure-data AST. Compilers build ASTs; the executor reads them. No behavior
lives on AST nodes — the single executor is the source of truth for
evaluation.

The grammar here is the Phase 1 sketch: single-hop EdgeConstraint
aggregations. Phase 4 extends with full conjunctions, distance constraints,
negation, and richer aggregators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

from common.types import CameraPose
from graph.schema import EdgeType


@dataclass(frozen=True)
class Variable:
    name: str   # without the "?" prefix; convention is single letter


@dataclass(frozen=True)
class EntityRef:
    """A named entity in the query, resolved at execute time to a graph
    node by display_label or alias. May resolve to multiple uids; the
    executor handles ambiguity per the ambiguity_policy on the question."""
    label: str
    color: str | None = None
    zone: str | None = None  # legacy; bathroom-fixture-only per §7.3


Operand = Union[Variable, EntityRef]


@dataclass(frozen=True)
class EdgeConstraint:
    source: Operand
    type: EdgeType
    target: Operand


@dataclass(frozen=True)
class NotConstraint:
    inner: EdgeConstraint


@dataclass(frozen=True)
class DistanceConstraint:
    """For FAR / NEAR thresholds at query time."""
    a: Operand
    b: Operand
    op: Literal["lt", "le", "gt", "ge"]
    threshold_meters: float


Constraint = Union[EdgeConstraint, NotConstraint, DistanceConstraint]


@dataclass(frozen=True)
class Aggregation:
    """Root of a QueryAST. Binds a variable over a where-clause of
    constraints and applies an aggregation operator."""
    op: Literal["ANY", "ALL", "COUNT", "EXISTS", "ENUMERATE"]
    bind: Variable
    where: list[Constraint] = field(default_factory=list)
    frame_hint: Literal["world", "viewpoint"] | None = None
    camera_anchor: CameraPose | None = None


QueryAST = Aggregation

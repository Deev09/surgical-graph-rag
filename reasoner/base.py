"""Reasoner stage Protocols and result types — phase0_design.md §5.5.

Routing (Phase 4-final, Phase 1 only ships rules + executor + verbalizer):

    QueryCompiler(rules) →
        compiled →           ASTExecutor → (bindings|empty|unknown) → Verbalizer
        parser_failure →     QueryCompiler(llm) →
                                 compiled →       ASTExecutor → Verbalizer
                                 parser_failure → Verbalizer abstains
                                 out_of_schema →  Verbalizer abstains
        out_of_schema →      Verbalizer abstains

`empty` and `unknown` are both valid executor outcomes; neither triggers
escalation. The LLM compiles NL to AST; it never sees the graph as JSON.

`empty` vs `unknown` is decided in the executor by reading the
CompletenessProfile in ExecutionContext, NOT by reading extractor
diagnostics. Stages do not score themselves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from graph.schema import Edge, EdgeType, GraphRef, SceneGraphBundle
from reasoner.ast import QueryAST


@dataclass(frozen=True)
class CompileResult:
    ast: QueryAST | None
    outcome: Literal["compiled", "parser_failure", "out_of_schema"]
    compiler_name: str               # "rules_v1" | "llm_v1"
    notes: str = ""


@dataclass(frozen=True)
class CompletenessProfile:
    """Externally calibrated coverage priors for a (backend, extractor,
    scene) triple. Lives in eval/; attached by a calibration run. Never
    produced by an extractor or builder.

    source semantics:
      - "oracle"   → by construction recall = 1.0; executor short-circuits
                     no-bindings to `empty`.
      - "measured" → recall priors from a labeled calibration dataset;
                     executor applies the empty_recall_threshold rule.
      - "unknown"  → no calibration; executor short-circuits no-bindings
                     to `unknown`.
    """
    source: Literal["oracle", "measured", "unknown"]
    entity_recall_by_class: dict[str, float]
    edge_recall_by_type: dict[EdgeType, float]
    calibration_dataset: str | None = None


@dataclass(frozen=True)
class ExecutionContext:
    completeness: CompletenessProfile
    empty_recall_threshold: float = 0.95


@dataclass(frozen=True)
class ExecutionResult:
    outcome: Literal["bindings", "empty", "unknown", "abstain", "execution_error"]
    bindings: list[dict[str, GraphRef]]
    evidence: list[Edge]
    coverage_floor: float
    notes: str = ""


@dataclass(frozen=True)
class Answer:
    text: str
    answered_by: Literal["rules_compiler", "llm_compiler", "verbalizer_abstain"]
    outcome: Literal["bindings", "empty", "unknown", "abstain", "parser_failure"]
    cited_uids: list[str]
    cited_edges: list[str]


class QueryCompiler(Protocol):
    name: str
    version: str

    def compile(self, question: str, scene: SceneGraphBundle) -> CompileResult: ...


class ASTExecutor(Protocol):
    name: str
    version: str

    def execute(
        self,
        ast: QueryAST,
        graph: SceneGraphBundle,
        ctx: ExecutionContext,
    ) -> ExecutionResult: ...


class Verbalizer(Protocol):
    name: str
    version: str

    def verbalize(
        self,
        question: str,
        compile_result: CompileResult,
        exec_result: ExecutionResult | None,
        scene: SceneGraphBundle,
    ) -> Answer: ...

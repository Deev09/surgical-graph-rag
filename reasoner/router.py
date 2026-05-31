"""Router — phase0_design.md §5.5.

Orchestrates: compiler → executor → verbalizer.

Phase 1: only the rules compiler is wired. parser_failure routes to
verbalizer abstention. The LLM compiler is a Phase 4 deliverable; the
constructor accepts an `llm_compiler` parameter so the Phase 4 wiring
will be a single line. Until then it is ignored.

Per §5.5 routing policy:
  compiled       → ASTExecutor → Verbalizer
  parser_failure → (Phase 4: try llm_compiler; Phase 1: abstain)
  out_of_schema  → Verbalizer abstains

`empty` and `unknown` are returned as-is. Neither triggers escalation.
"""
from __future__ import annotations

from graph.schema import SceneGraphBundle
from reasoner.base import (
    ASTExecutor, Answer, CompileResult, ExecutionContext, QueryCompiler,
    Verbalizer,
)


class Router:
    name: str = "router_v1"
    version: str = "0.1"

    def __init__(
        self,
        *,
        compiler: QueryCompiler,
        executor: ASTExecutor,
        verbalizer: Verbalizer,
        llm_compiler: QueryCompiler | None = None,
    ):
        self.compiler = compiler
        self.executor = executor
        self.verbalizer = verbalizer
        self.llm_compiler = llm_compiler

    def answer(
        self,
        question: str,
        graph: SceneGraphBundle,
        ctx: ExecutionContext,
    ) -> Answer:
        cr = self.compiler.compile(question, graph)

        if cr.outcome == "compiled" and cr.ast is not None:
            er = self.executor.execute(cr.ast, graph, ctx)
            return self.verbalizer.verbalize(question, cr, er, graph)

        if cr.outcome == "parser_failure" and self.llm_compiler is not None:
            cr2 = self.llm_compiler.compile(question, graph)
            if cr2.outcome == "compiled" and cr2.ast is not None:
                er = self.executor.execute(cr2.ast, graph, ctx)
                return self.verbalizer.verbalize(question, cr2, er, graph)
            # LLM also failed → abstain via verbalizer using the LLM's CompileResult
            return self.verbalizer.verbalize(question, cr2, None, graph)

        # parser_failure (no LLM in Phase 1) or out_of_schema → abstain
        return self.verbalizer.verbalize(question, cr, None, graph)

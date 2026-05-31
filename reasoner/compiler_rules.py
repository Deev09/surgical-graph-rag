"""Rules-based QueryCompiler — phase0_design.md §5.5.

Phase 1: regex templates for single-hop directional and proximity
questions. Each match compiles to an Aggregation(ENUMERATE, ?x, [EdgeConstraint(?x, TYPE, EntityRef(anchor))]).

The compiler does NOT validate that the anchor exists in the graph (that
is the executor's job). It only handles NL → AST.

Patterns ported and tightened from tiny_graph_demo.py
SPATIAL_QUESTION_PATTERNS. The new compiler uses re.match (not search)
plus a trailing `\\??$` so accidental prefixes do not produce
false-positive parses.

The LLM compiler is stubbed (Phase 4). Parse-failure produces a
CompileResult with outcome='parser_failure' which the router maps to
verbalizer abstention in Phase 1.
"""
from __future__ import annotations

import re
from typing import Pattern

from graph.schema import EdgeType, SceneGraphBundle
from reasoner.ast import (
    Aggregation, EdgeConstraint, EntityRef, QueryAST, Variable,
)
from reasoner.base import CompileResult


# (regex, edge_type, notes)
_PATTERNS: list[tuple[Pattern[str], EdgeType, str]] = [
    (re.compile(r"what(?:'s| is) (?:to the )?left of (?:the )?(.+?)\??$"), "LEFT_OF", "left_of_pattern"),
    (re.compile(r"what(?:'s| is) (?:to the )?right of (?:the )?(.+?)\??$"), "RIGHT_OF", "right_of_pattern"),
    (re.compile(r"what(?:'s| is) (?:directly )?below (?:the )?(.+?)\??$"), "BELOW", "below_pattern"),
    (re.compile(r"what(?:'s| is) under (?:the )?(.+?)\??$"), "BELOW", "under_pattern"),
    (re.compile(r"what(?:'s| is) (?:directly )?above (?:the )?(.+?)\??$"), "ABOVE", "above_pattern"),
    (re.compile(r"what(?:'s| is) in front of (?:the )?(.+?)\??$"), "IN_FRONT_OF", "in_front_of_pattern"),
    (re.compile(r"what(?:'s| is) behind (?:the )?(.+?)\??$"), "BEHIND", "behind_pattern"),
    (re.compile(r"what(?:'s| is) near (?:the )?(.+?)\??$"), "NEAR", "near_pattern"),
    (re.compile(r"what(?:'s| is) (?:close to|next to) (?:the )?(.+?)\??$"), "NEAR", "near_synonym_pattern"),
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class RulesCompiler:
    """Implements the QueryCompiler Protocol."""
    name: str = "rules_v1"
    version: str = "0.1"

    def compile(self, question: str, scene: SceneGraphBundle) -> CompileResult:
        q = _normalize(question)
        for pattern, edge_type, note in _PATTERNS:
            m = pattern.match(q)
            if not m:
                continue
            anchor_label = m.group(1).strip()
            # Strip a trailing punctuation that the .+? captured.
            anchor_label = anchor_label.rstrip(".?!").strip()
            if not anchor_label:
                continue
            bind = Variable(name="x")
            ast: QueryAST = Aggregation(
                op="ENUMERATE",
                bind=bind,
                where=[
                    EdgeConstraint(
                        source=bind,
                        type=edge_type,
                        target=EntityRef(label=anchor_label),
                    ),
                ],
            )
            return CompileResult(
                ast=ast,
                outcome="compiled",
                compiler_name=self.name,
                notes=f"matched={note} edge_type={edge_type} anchor={anchor_label!r}",
            )
        return CompileResult(
            ast=None,
            outcome="parser_failure",
            compiler_name=self.name,
            notes="no rule pattern matched",
        )

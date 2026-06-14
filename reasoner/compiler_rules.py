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
    Aggregation, EdgeConstraint, EntityClassRef, EntityRef, QueryAST, SurfaceRef,
    Variable,
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


# P4.04 — "on the X" support queries (floor answerable via SUPPORTS; others
# deferred). P5.03 — "against the wall" -> CONTACTS_SURFACE, "near the wall"
# -> NEAR_SURFACE (both SurfaceRef-anchored stored relations), and "attached
# to the X" -> deferred (ATTACHED_TO not in P5; attachment never routes
# through contact). Deferred = out_of_schema with a "deferred:" note, NOT
# parser_failure and NOT empty.
_ON_PATTERN = re.compile(r"what(?:'s| is) on (?:the )?(.+?)\??$")
_AGAINST_PATTERN = re.compile(r"what(?:'s| is) against (?:the )?(.+?)\??$")
_NEAR_PATTERN = re.compile(r"what(?:'s| is) near (?:the )?(.+?)\??$")
_ATTACHED_PATTERN = re.compile(r"what(?:'s| is) attached to (?:the )?(.+?)\??$")

_ENTITY_SURFACE_ON_CLASSES = {
    "table": "table",
    "tabletop": "table",
    "desk": "desk",
    "chair": "chair",
    "seat": "chair",
    "stool": "stool",
    "bench": "bench",
    "shelf": "shelf",
    "sofa": "sofa",
    "plant stand": "plant-stand",
    "plant-stand": "plant-stand",
    "counter": "counter",
}
_DEFERRED_ON_GENERIC = (
    "deferred: this support class is not covered by EntitySurface in P6"
)
_DEFERRED_AGAINST = (
    "deferred: wall contact only supports 'against the wall' in P5; this "
    "surface is not a wall"
)
_DEFERRED_ATTACHED = (
    "deferred: ATTACHED_TO is not in P5 ('against' != 'attached'); attachment "
    "needs more than wall contact"
)


class RulesCompiler:
    """Implements the QueryCompiler Protocol."""
    name: str = "rules_v1"
    version: str = "0.1"

    def _compile_surface_query(self, q: str) -> CompileResult | None:
        """P4.04 support queries. Returns a CompileResult for "on the X" /
        "against the X", or None if neither pattern matches (fall through to
        the generic single-hop patterns)."""
        m = _ON_PATTERN.match(q)
        if m:
            noun = m.group(1).rstrip(".?!").strip()
            if noun == "floor":
                bind = Variable(name="x")
                ast: QueryAST = Aggregation(
                    op="ENUMERATE",
                    bind=bind,
                    where=[
                        EdgeConstraint(
                            source=SurfaceRef(surface_type="floor"),
                            type="SUPPORTS",
                            target=bind,
                        ),
                    ],
                )
                return CompileResult(
                    ast=ast,
                    outcome="compiled",
                    compiler_name=self.name,
                    notes="matched=on_floor_pattern edge_type=SUPPORTS surface=floor",
                )
            entity_class = _ENTITY_SURFACE_ON_CLASSES.get(noun)
            if entity_class is not None:
                bind = Variable(name="x")
                ast = Aggregation(
                    op="ENUMERATE",
                    bind=bind,
                    where=[
                        EdgeConstraint(
                            source=EntityClassRef(entity_class=entity_class),
                            type="SUPPORTS",
                            target=bind,
                        ),
                    ],
                )
                return CompileResult(
                    ast=ast,
                    outcome="compiled",
                    compiler_name=self.name,
                    notes=(
                        "matched=on_entity_surface_pattern edge_type=SUPPORTS "
                        f"entity_class={entity_class}"
                    ),
                )
            note = _DEFERRED_ON_GENERIC
            return CompileResult(
                ast=None, outcome="out_of_schema",
                compiler_name=self.name, notes=note,
            )
        m = _ATTACHED_PATTERN.match(q)
        if m:
            # "attached to the X" -> always deferred; never routes through
            # contact (D2: 'against' != 'attached').
            return CompileResult(
                ast=None, outcome="out_of_schema",
                compiler_name=self.name, notes=_DEFERRED_ATTACHED,
            )
        m = _AGAINST_PATTERN.match(q)
        if m:
            noun = m.group(1).rstrip(".?!").strip()
            if noun == "wall":
                return self._surface_relation_result(
                    "CONTACTS_SURFACE", "wall", "against_wall_pattern",
                )
            return CompileResult(
                ast=None, outcome="out_of_schema",
                compiler_name=self.name, notes=_DEFERRED_AGAINST,
            )
        m = _NEAR_PATTERN.match(q)
        if m:
            noun = m.group(1).rstrip(".?!").strip()
            if noun == "wall":
                return self._surface_relation_result(
                    "NEAR_SURFACE", "wall", "near_wall_pattern",
                )
            # "near the <entity>" falls through to the generic NEAR pattern.
            return None
        return None

    def _surface_relation_result(
        self, edge_type: EdgeType, surface_type: str, note_tag: str,
    ) -> CompileResult:
        """Compile RELATION(?x, SurfaceRef(surface_type)) for a stored
        entity->surface relation. The relation is chosen here (compiler
        pattern); the executor handles NEAR_SURFACE / CONTACTS_SURFACE through
        one parameterized branch."""
        bind = Variable(name="x")
        ast: QueryAST = Aggregation(
            op="ENUMERATE",
            bind=bind,
            where=[
                EdgeConstraint(
                    source=bind,
                    type=edge_type,
                    target=SurfaceRef(surface_type=surface_type),
                ),
            ],
        )
        return CompileResult(
            ast=ast, outcome="compiled", compiler_name=self.name,
            notes=f"matched={note_tag} edge_type={edge_type} surface={surface_type}",
        )

    def compile(self, question: str, scene: SceneGraphBundle) -> CompileResult:
        q = _normalize(question)
        surface_result = self._compile_surface_query(q)
        if surface_result is not None:
            return surface_result
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

"""Verbalizer — phase0_design.md §5.5.

Maps bindings (or empty / unknown / abstain outcomes) to a user-facing
Answer. Display labels come from Node.attributes['display_label'] when
present (preserves duplicate-suffix labels like 'chair_1'), falling back
to Node.label.

empty and unknown produce distinct NL strings per the design rule:
  empty   → "Nothing matches" (the graph confidently asserts none exists)
  unknown → "I don't have enough evidence" (absence of evidence)
"""
from __future__ import annotations

from graph.schema import GraphRef, Node, SceneGraphBundle
from reasoner.base import Answer, CompileResult, ExecutionResult


def _display_label(node: Node) -> str:
    return str(node.attributes.get("display_label") or node.label)


def _find_node(graph: SceneGraphBundle, uid: str) -> Node | None:
    for n in graph.nodes:
        if n.id == uid:
            return n
    return None


def _label_for_ref(graph: SceneGraphBundle, ref: GraphRef) -> str:
    if ref.kind == "surface":
        return f"surface {ref.uid}"
    node = _find_node(graph, ref.uid)
    if node is None:
        return f"unknown entity {ref.uid}"
    return _display_label(node).replace("_", " ")


def _join_human(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


class StandardVerbalizer:
    """Implements the Verbalizer Protocol."""
    name: str = "verbalizer_v1"
    version: str = "0.1"

    def verbalize(
        self,
        question: str,
        compile_result: CompileResult,
        exec_result: ExecutionResult | None,
        scene: SceneGraphBundle,
    ) -> Answer:
        if compile_result.outcome == "parser_failure":
            return Answer(
                text="I couldn't understand the question well enough to answer it.",
                answered_by="verbalizer_abstain",
                outcome="parser_failure",
                cited_uids=[], cited_edges=[],
            )
        if compile_result.outcome == "out_of_schema":
            note = compile_result.notes or ""
            if note.startswith("deferred:"):
                reason = note[len("deferred:"):].strip()
                # Explicit deferral — must NOT read as "nothing is there".
                return Answer(
                    text=(
                        f"I can't answer that yet — {reason}. "
                        "This isn't a claim that nothing is there."
                    ),
                    answered_by="verbalizer_abstain",
                    outcome="abstain",
                    cited_uids=[], cited_edges=[],
                )
            return Answer(
                text="That question isn't expressible in the current scene graph.",
                answered_by="verbalizer_abstain",
                outcome="abstain",
                cited_uids=[], cited_edges=[],
            )

        # compiled — exec_result must be present
        if exec_result is None:
            return Answer(
                text="No execution result was produced.",
                answered_by="verbalizer_abstain",
                outcome="abstain",
                cited_uids=[], cited_edges=[],
            )

        compiler_tag = (
            "llm_compiler" if compile_result.compiler_name.startswith("llm_")
            else "rules_compiler"
        )

        if exec_result.outcome == "execution_error":
            return Answer(
                text=f"The query couldn't be executed: {exec_result.notes}",
                answered_by="verbalizer_abstain",
                outcome="abstain",
                cited_uids=[], cited_edges=[],
            )
        if exec_result.outcome == "empty":
            return Answer(
                text="Nothing in the scene matches that.",
                answered_by=compiler_tag,
                outcome="empty",
                cited_uids=[], cited_edges=[],
            )
        if exec_result.outcome == "unknown":
            return Answer(
                text=(
                    "I don't have enough evidence to say. The underlying "
                    "extractor's recall is below the threshold needed to "
                    "treat a no-match result as a true negative."
                ),
                answered_by=compiler_tag,
                outcome="unknown",
                cited_uids=[], cited_edges=[],
            )
        if exec_result.outcome == "abstain":
            return Answer(
                text="I can't answer this question.",
                answered_by=compiler_tag,
                outcome="abstain",
                cited_uids=[], cited_edges=[],
            )

        # bindings — produce a natural-language listing
        labels: list[str] = []
        cited_uids: list[str] = []
        for binding in exec_result.bindings:
            for ref in binding.values():
                labels.append(_label_for_ref(scene, ref))
                cited_uids.append(ref.uid)
        cited_edges = [e.edge_id for e in exec_result.evidence]

        if len(labels) == 1:
            text = f"The {labels[0]}."
        else:
            text = f"The {_join_human(labels)}."

        return Answer(
            text=text,
            answered_by=compiler_tag,
            outcome="bindings",
            cited_uids=cited_uids,
            cited_edges=cited_edges,
        )

"""Additive LLM query parser — v1 regex parser is untouched.

parse_query_llm(q, *, strict) returns an LLMParseResult. Success carries a
ParsedQuery with the exact contract emitted by tiny_graph_demo.parse_query.
Failure (invalid JSON / schema / vocab / cross-field) returns
parsed=None with an error reason.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

from tiny_graph_demo import ParsedQuery, _label_equiv

from parsers.vocab import (
    ALLOWED_COLORS,
    ALLOWED_INTENTS,
    ALLOWED_LABEL_PHRASES,
    ALLOWED_RELATIONS,
    ALLOWED_TARGET_TYPES,
    ALLOWED_ZONE_FAMILIES,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


REQUIRED_KEYS = frozenset({
    "intent",
    "spatial_relation",
    "relation_anchor",
    "zone_family",
    "target_label",
    "target_type",
    "color",
    "zone",
    "anchor_objects",
})


class LLMParseError(Exception):
    def __init__(self, reason: str, raw_content: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.raw_content = raw_content


@dataclass
class LLMParseResult:
    parsed: ParsedQuery | None
    error: str | None
    raw_content: str
    retry_count: int
    latency_ms: float


def _build_prompt(query: str) -> str:
    intents = sorted(ALLOWED_INTENTS)
    relations = sorted(ALLOWED_RELATIONS)
    zone_families = sorted(ALLOWED_ZONE_FAMILIES)
    labels = sorted(ALLOWED_LABEL_PHRASES)
    types = sorted(ALLOWED_TARGET_TYPES)
    colors = sorted(ALLOWED_COLORS)

    header = (
        "You are a query parser for a room-scene QA system. Convert the user's "
        "question into a strict JSON object. Output ONLY the JSON object — no "
        "prose, no code fences, no commentary.\n\n"
        "Required keys (all of them; use null when a field does not apply):\n"
        f"  intent             one of: {intents}\n"
        f"  spatial_relation   one of: {relations} or null\n"
        "  relation_anchor    string (an object label from the vocab) or null\n"
        f"  zone_family        one of: {zone_families} or null\n"
        "  target_label       string (an object label from the vocab) or null\n"
        f"  target_type        one of: {types} or null\n"
        f"  color              one of: {colors} or null\n"
        "  zone               always null\n"
        "  anchor_objects     list of object-label strings (may be empty)\n\n"
        f"Object label vocab:\n  {labels}\n\n"
        "Intent semantics:\n"
        "  describe_spatial       'what is left/right/above/below/in_front_of/behind <object>'\n"
        "                         set spatial_relation AND relation_anchor\n"
        "  describe_zone          'what is on the <wall>' — set zone_family\n"
        "  describe_near_zone     'what is near the <wall>' — set zone_family\n"
        "  describe_near_anchor   'what is near the <object>' — set anchor_objects=[<object>]\n"
        "  find_location          'where is <object>' — set target_label\n"
        "  find_object / describe fallback; most fields null\n\n"
        "Critical rules:\n"
        "  - NEAR is NOT a spatial_relation; use describe_near_zone or describe_near_anchor.\n"
        "  - zone_family must come from the allowed list; NEVER invent node-zone strings.\n"
        "  - If intent is not describe_spatial, spatial_relation and relation_anchor MUST both be null.\n"
        "  - zone must always be null.\n"
        "  - Output strict JSON. No trailing commas, no comments, no code fences.\n\n"
    )

    examples = (
        'Examples:\n'
        'Q: "What is left of the sink?"\n'
        '{"intent": "describe_spatial", "spatial_relation": "LEFT_OF", "relation_anchor": "sink", '
        '"zone_family": null, "target_label": null, "target_type": null, "color": null, '
        '"zone": null, "anchor_objects": []}\n\n'
        'Q: "What sits under the mirror?"\n'
        '{"intent": "describe_spatial", "spatial_relation": "BELOW", "relation_anchor": "mirror", '
        '"zone_family": null, "target_label": null, "target_type": null, "color": null, '
        '"zone": null, "anchor_objects": []}\n\n'
        'Q: "What is near the back wall?"\n'
        '{"intent": "describe_near_zone", "spatial_relation": null, "relation_anchor": null, '
        '"zone_family": "back_wall", "target_label": null, "target_type": null, "color": null, '
        '"zone": null, "anchor_objects": []}\n\n'
        'Q: "What\'s attached to the right wall?"\n'
        '{"intent": "describe_zone", "spatial_relation": null, "relation_anchor": null, '
        '"zone_family": "right_wall", "target_label": null, "target_type": null, "color": null, '
        '"zone": null, "anchor_objects": []}\n\n'
        'Q: "Where is the vending machine?"\n'
        '{"intent": "find_location", "spatial_relation": null, "relation_anchor": null, '
        '"zone_family": null, "target_label": "vending machine", "target_type": null, "color": null, '
        '"zone": null, "anchor_objects": []}\n\n'
        'Q: "What is near the black door?"\n'
        '{"intent": "describe_near_anchor", "spatial_relation": null, "relation_anchor": null, '
        '"zone_family": null, "target_label": null, "target_type": null, "color": null, '
        '"zone": null, "anchor_objects": ["black door"]}\n\n'
    )

    footer = f'Now parse this query (output JSON only):\nQ: "{query}"\n'
    return header + examples + footer


def _extract_json(content: str) -> dict:
    s = content.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise LLMParseError(f"invalid JSON: {e}", raw_content=content)
    if not isinstance(obj, dict):
        raise LLMParseError("JSON root not an object", raw_content=content)
    return obj


def _resolve_label(value: str, *, strict: bool) -> bool:
    if value in ALLOWED_LABEL_PHRASES:
        return True
    if strict:
        return False
    return any(_label_equiv(value, p) for p in ALLOWED_LABEL_PHRASES)


def _validate(obj: dict, *, strict: bool) -> ParsedQuery:
    missing = REQUIRED_KEYS - obj.keys()
    if missing:
        raise LLMParseError(f"missing keys: {sorted(missing)}")
    unknown = obj.keys() - REQUIRED_KEYS
    if unknown:
        raise LLMParseError(f"unknown keys: {sorted(unknown)}")

    intent = obj["intent"]
    if intent not in ALLOWED_INTENTS:
        raise LLMParseError(f"invalid intent: {intent!r}")

    sr = obj["spatial_relation"]
    if sr is not None and sr not in ALLOWED_RELATIONS:
        raise LLMParseError(f"invalid spatial_relation: {sr!r}")

    ra = obj["relation_anchor"]
    if ra is not None:
        if not isinstance(ra, str):
            raise LLMParseError("relation_anchor must be string or null")
        if not _resolve_label(ra, strict=strict):
            raise LLMParseError(
                f"relation_anchor not in vocab ({'strict' if strict else 'fuzzy'}): {ra!r}"
            )

    zf = obj["zone_family"]
    if zf is not None and zf not in ALLOWED_ZONE_FAMILIES:
        raise LLMParseError(f"invalid zone_family: {zf!r}")

    tl = obj["target_label"]
    if tl is not None:
        if not isinstance(tl, str):
            raise LLMParseError("target_label must be string or null")
        if not _resolve_label(tl, strict=strict):
            raise LLMParseError(
                f"target_label not in vocab ({'strict' if strict else 'fuzzy'}): {tl!r}"
            )

    tt = obj["target_type"]
    if tt is not None and tt not in ALLOWED_TARGET_TYPES:
        raise LLMParseError(f"invalid target_type: {tt!r}")

    color = obj["color"]
    if color is not None and color not in ALLOWED_COLORS:
        raise LLMParseError(f"invalid color: {color!r}")

    zone = obj["zone"]
    if zone is not None:
        raise LLMParseError(f"zone must be null (v1 parity), got {zone!r}")

    ao = obj["anchor_objects"]
    if not isinstance(ao, list):
        raise LLMParseError("anchor_objects must be a list")
    for i, a in enumerate(ao):
        if not isinstance(a, str):
            raise LLMParseError(f"anchor_objects[{i}] must be string")
        if not _resolve_label(a, strict=strict):
            raise LLMParseError(
                f"anchor_objects[{i}] not in vocab "
                f"({'strict' if strict else 'fuzzy'}): {a!r}"
            )

    if intent == "describe_spatial":
        if sr is None or ra is None:
            raise LLMParseError(
                "describe_spatial requires spatial_relation and relation_anchor"
            )
    else:
        if sr is not None or ra is not None:
            raise LLMParseError(
                f"spatial_relation/relation_anchor must be null for intent={intent}"
            )

    if intent in ("describe_zone", "describe_near_zone") and zf is None:
        raise LLMParseError(f"{intent} requires zone_family")

    if intent == "describe_near_anchor" and not ao:
        raise LLMParseError("describe_near_anchor requires non-empty anchor_objects")

    return ParsedQuery(
        raw_query="",
        intent=intent,
        target_label=tl,
        target_type=tt,
        color=color,
        zone=None,
        anchor_objects=list(ao),
        spatial_relation=sr,
        relation_anchor=ra,
        zone_family=zf,
    )


def _call_model(prompt: str) -> str:
    if OpenAI is None:
        raise RuntimeError("openai not installed; run: pip install -r requirements.txt")
    base_url = os.environ.get(
        "CONTEXT_SURGEON_OPENAI_BASE_URL",
        "http://localhost:12434/engines/v1",
    )
    model = os.environ.get("CONTEXT_SURGEON_MODEL", "ai/llama3.1")
    client = OpenAI(base_url=base_url, api_key="not-needed")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    content = resp.choices[0].message.content or ""
    return content.strip()


def parse_query_llm(query: str, *, strict: bool = False) -> LLMParseResult:
    prompt = _build_prompt(query)
    t0 = time.perf_counter()
    retry_count = 0
    combined_raw = ""

    try:
        raw = _call_model(prompt)
        combined_raw = raw
        obj = _extract_json(raw)
        parsed = _validate(obj, strict=strict)
    except LLMParseError as first_err:
        retry_count = 1
        nudge = (
            "\n\nYour previous response was invalid: "
            f"{first_err.reason}\nReturn ONLY the JSON object with the exact schema above."
        )
        try:
            raw2 = _call_model(prompt + nudge)
            combined_raw = combined_raw + "\n---RETRY---\n" + raw2
            obj = _extract_json(raw2)
            parsed = _validate(obj, strict=strict)
        except LLMParseError as second_err:
            return LLMParseResult(
                parsed=None,
                error=second_err.reason,
                raw_content=combined_raw,
                retry_count=retry_count,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
    except Exception as e:
        return LLMParseResult(
            parsed=None,
            error=f"transport/runtime: {e!r}",
            raw_content=combined_raw,
            retry_count=retry_count,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    parsed.raw_query = query
    return LLMParseResult(
        parsed=parsed,
        error=None,
        raw_content=combined_raw,
        retry_count=retry_count,
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )

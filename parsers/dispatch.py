"""Parser mode dispatcher.

Routes to the v1 regex parser or the LLM parser based on mode. v1 behavior
is preserved byte-for-byte in regex mode because it calls parse_query
directly with no wrapping transform.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from tiny_graph_demo import ParsedQuery, parse_query

from parsers.llm_parser import parse_query_llm


MODES: tuple[str, ...] = ("regex", "llm", "llm_first", "llm_strict")


@dataclass
class DispatchResult:
    parsed: ParsedQuery | None
    source: str  # "regex" | "llm" | "llm_fallback" | "unparseable"
    error: str | None
    retry_count: int
    latency_ms: float
    raw_content: str | None


def parse(query: str, mode: str) -> DispatchResult:
    if mode not in MODES:
        raise ValueError(f"unknown parser mode: {mode!r}")

    if mode == "regex":
        t0 = time.perf_counter()
        p = parse_query(query)
        return DispatchResult(
            parsed=p,
            source="regex",
            error=None,
            retry_count=0,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            raw_content=None,
        )

    strict = (mode == "llm_strict")
    r = parse_query_llm(query, strict=strict)

    if mode == "llm_first":
        if r.parsed is not None:
            return DispatchResult(
                parsed=r.parsed,
                source="llm",
                error=None,
                retry_count=r.retry_count,
                latency_ms=r.latency_ms,
                raw_content=r.raw_content,
            )
        fallback = parse_query(query)
        return DispatchResult(
            parsed=fallback,
            source="llm_fallback",
            error=r.error,
            retry_count=r.retry_count,
            latency_ms=r.latency_ms,
            raw_content=r.raw_content,
        )

    # llm or llm_strict — no fallback, explicit unparseable on failure.
    if r.parsed is None:
        return DispatchResult(
            parsed=None,
            source="unparseable",
            error=r.error,
            retry_count=r.retry_count,
            latency_ms=r.latency_ms,
            raw_content=r.raw_content,
        )
    return DispatchResult(
        parsed=r.parsed,
        source="llm",
        error=None,
        retry_count=r.retry_count,
        latency_ms=r.latency_ms,
        raw_content=r.raw_content,
    )

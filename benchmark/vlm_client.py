"""Anthropic Claude vision client for spatial-QA evaluation.

Sends a question + scene images + scene canonical-id list to Claude and forces
structured output via the tool-use API. Returns a RunnerOutput-shaped dict:

    {
      "answer_entity_ids": [str, ...],
      "abstained": bool,
      "answer_text": str,
      "evidence": {
        "entity_ids": [str, ...],
        "source_frame_idx": int | null,
        "crop_bbox": [x, y, w, h] | null,
        "confidence": float in [0, 1]
      },
      "raw_response": str       # debug aid; not part of scoring
    }

Bumps PROMPT_TEMPLATE_VERSION when the schema or system prompt changes —
DiskCache uses it as part of the key, so old entries are invalidated.

MockVLMClient returns deterministic stub responses without calling the API.
Use it (eval_vlm.py --mock) to validate the runner pipeline end-to-end without
spending money or needing an API key.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROMPT_TEMPLATE_VERSION = "v0.1"
DEFAULT_MODEL_ID = "claude-haiku-4-5-20251001"


SYSTEM_PROMPT = """You are a careful visual-spatial QA system. You answer questions about a physical room shown in the provided photos.

Rules:
- Answer using only the canonical entity IDs from the list the user provides. If the answer is not one of those IDs, abstain.
- It is OK to abstain (set "abstained": true and return an empty answer_entity_ids list) when the photos do not let you decide.
- For multi-instance questions, return all matching IDs.
- For "where is X?" questions, return the entity IDs corresponding to X.
- Cite the photo whose 0-based index most clearly shows your answer in evidence.source_frame_idx.
- Always submit your answer via the submit_answer tool. Do not respond in plain text.
"""


def _entity_block(scene_objects: list[dict[str, Any]]) -> str:
    lines = []
    for o in scene_objects:
        cid = str(o["label"])
        m = re.match(r"^(.*)_\d+$", cid)
        display = m.group(1).replace("_", " ") if m else cid.replace("_", " ")
        lines.append(f"  - {cid}  (a {display})")
    return "\n".join(lines)


USER_PROMPT_TEMPLATE = """Question: {question}

Available canonical entity IDs in this scene (you must use ONLY these in your answer):
{entities_block}

Expected answer_type: {answer_type}
Question category: {category}

The photos below show the same room from different viewpoints. They are 0-indexed in the order shown.

Submit your answer via the submit_answer tool."""


ANSWER_TOOL_SCHEMA: dict[str, Any] = {
    "name": "submit_answer",
    "description": "Submit the final spatial-QA answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer_entity_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered list of canonical entity IDs that answer the question. Empty list if abstaining.",
            },
            "abstained": {
                "type": "boolean",
                "description": "True if the question cannot be answered from the photos.",
            },
            "answer_text": {
                "type": "string",
                "description": "Short natural-language explanation of the answer.",
            },
            "evidence": {
                "type": "object",
                "properties": {
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Canonical IDs cited as evidence (typically same as answer_entity_ids).",
                    },
                    "source_frame_idx": {
                        "type": ["integer", "null"],
                        "description": "0-based index of the photo most clearly showing the answer, or null.",
                    },
                    "crop_bbox": {
                        "type": ["array", "null"],
                        "items": {"type": "number"},
                        "description": "[x, y, w, h] normalized to [0,1] for the relevant region in source frame, or null.",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence in the answer.",
                    },
                },
                "required": ["entity_ids", "confidence"],
            },
        },
        "required": ["answer_entity_ids", "abstained", "answer_text", "evidence"],
    },
}


@dataclass
class VLMResponse:
    output: dict[str, Any]            # parsed structured answer
    raw_response: str                 # debug; the assistant's full message text or block dump
    latency_ms: float
    tokens: dict[str, int] | None = None
    error: str | None = None


class BaseVLMClient(ABC):
    model_id: str
    prompt_template_version: str = PROMPT_TEMPLATE_VERSION

    @abstractmethod
    def call(
        self,
        *,
        question_text: str,
        category: str,
        answer_type: str,
        scene_objects: list[dict[str, Any]],
        image_paths: list[Path],
    ) -> VLMResponse: ...

    def build_prompt_payload(
        self,
        *,
        question_text: str,
        category: str,
        answer_type: str,
        scene_objects: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "system": SYSTEM_PROMPT,
            "user": USER_PROMPT_TEMPLATE.format(
                question=question_text,
                entities_block=_entity_block(scene_objects),
                answer_type=answer_type,
                category=category,
            ),
            "tool_schema_name": ANSWER_TOOL_SCHEMA["name"],
            "tool_schema_version": PROMPT_TEMPLATE_VERSION,
        }


class AnthropicVLMClient(BaseVLMClient):
    """Anthropic Claude with vision + tool-use forced output.

    Requires:
      - `pip install anthropic`
      - ANTHROPIC_API_KEY env var
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, max_tokens: int = 1024):
        self.model_id = model_id
        self.max_tokens = max_tokens
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK not installed. Run: pip install anthropic"
            ) from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it before running eval_vlm.py."
            )
        from anthropic import Anthropic
        self._client = Anthropic()

    def _image_block(self, path: Path) -> dict[str, Any]:
        suffix = path.suffix.lower().lstrip(".")
        media_type = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "webp": "image/webp",
        }.get(suffix, "image/png")
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        }

    def call(
        self,
        *,
        question_text: str,
        category: str,
        answer_type: str,
        scene_objects: list[dict[str, Any]],
        image_paths: list[Path],
    ) -> VLMResponse:
        payload = self.build_prompt_payload(
            question_text=question_text,
            category=category,
            answer_type=answer_type,
            scene_objects=scene_objects,
        )
        content_blocks: list[dict[str, Any]] = [
            self._image_block(p) for p in image_paths
        ]
        content_blocks.append({"type": "text", "text": payload["user"]})

        t0 = time.perf_counter()
        try:
            msg = self._client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                system=payload["system"],
                tools=[ANSWER_TOOL_SCHEMA],
                tool_choice={"type": "tool", "name": "submit_answer"},
                messages=[{"role": "user", "content": content_blocks}],
            )
        except Exception as e:
            return VLMResponse(
                output={},
                raw_response="",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error=f"vlm_format: {type(e).__name__}: {e}",
            )
        latency = (time.perf_counter() - t0) * 1000.0

        tool_use = None
        text_blocks: list[str] = []
        for block in msg.content:
            btype = getattr(block, "type", None)
            if btype == "tool_use" and getattr(block, "name", "") == "submit_answer":
                tool_use = getattr(block, "input", {})
            elif btype == "text":
                text_blocks.append(getattr(block, "text", ""))

        if tool_use is None:
            return VLMResponse(
                output={},
                raw_response="\n".join(text_blocks),
                latency_ms=latency,
                error="vlm_format: no submit_answer tool_use block",
            )

        usage = getattr(msg, "usage", None)
        tokens = None
        if usage is not None:
            tokens = {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
            }
        return VLMResponse(
            output=dict(tool_use),
            raw_response=json.dumps(tool_use),
            latency_ms=latency,
            tokens=tokens,
        )


class MockVLMClient(BaseVLMClient):
    """Deterministic stub. Always abstains. Used by eval_vlm.py --mock to
    validate the runner pipeline end-to-end without API spend or photos."""

    def __init__(self, model_id: str = "mock-vlm-v0"):
        self.model_id = model_id

    def call(
        self,
        *,
        question_text: str,
        category: str,
        answer_type: str,
        scene_objects: list[dict[str, Any]],
        image_paths: list[Path],
    ) -> VLMResponse:
        out = {
            "answer_entity_ids": [],
            "abstained": True,
            "answer_text": "(mock) abstained",
            "evidence": {
                "entity_ids": [],
                "source_frame_idx": None,
                "crop_bbox": None,
                "confidence": 0.0,
            },
        }
        return VLMResponse(
            output=out,
            raw_response=json.dumps(out),
            latency_ms=0.1,
            tokens={"input_tokens": 0, "output_tokens": 0},
        )

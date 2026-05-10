"""Disk cache for VLM responses.

Cache key: sha256 over (model_id, prompt_template_version, question_id,
image_sha256_list, prompt_payload_json). Cache value: raw upstream response,
parsed RunnerOutput-shaped dict, latency, timestamp.

JSON-on-disk under runs/<runner>/cache/. Each entry is one file named after the
hash. Sequential eval runs over ~10–100 questions per scene don't hit the kind
of contention that justifies SQLite.

There is no TTL. If the upstream model behavior changes, bump
PROMPT_TEMPLATE_VERSION in the calling code (vlm_client.py) — that field is part
of the key, so old entries are no longer hit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class CacheKey:
    question_id: str
    model_id: str
    prompt_template_version: str
    image_sha256s: list[str]
    prompt_payload: dict[str, Any] = field(default_factory=dict)

    def hash(self) -> str:
        h = hashlib.sha256()
        for part in (self.model_id, self.prompt_template_version, self.question_id):
            h.update(part.encode())
            h.update(b"\x00")
        for s in self.image_sha256s:
            h.update(s.encode())
            h.update(b"\x00")
        h.update(json.dumps(self.prompt_payload, sort_keys=True).encode())
        return h.hexdigest()


class DiskCache:
    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: CacheKey) -> dict[str, Any] | None:
        path = self.cache_dir / f"{key.hash()}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def put(self, key: CacheKey, value: dict[str, Any]) -> None:
        path = self.cache_dir / f"{key.hash()}.json"
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

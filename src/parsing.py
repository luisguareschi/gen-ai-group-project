"""Robust JSON extraction helpers for LLM output fallback paths."""
from __future__ import annotations

import json
import re
from typing import Any

_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def extract_json(text: str) -> dict[str, Any]:
    """Best-effort parse of the first top-level JSON object in ``text``.

    Handles common LLM quirks: fenced code blocks, leading prose, trailing text.
    Raises ``ValueError`` if no JSON object can be parsed.
    """
    if not text:
        raise ValueError("empty text")

    stripped = text.strip()
    for candidate in (stripped, _strip_fences(stripped)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    match = _JSON_BLOCK.search(stripped)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse JSON object: {exc}") from exc
    raise ValueError("No JSON object found in text.")


def _strip_fences(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text.strip()

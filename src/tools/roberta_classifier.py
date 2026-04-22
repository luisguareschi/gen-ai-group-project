"""HuggingFace RoBERTa fake-news classifier (pre-computed signal for the Judge)."""
from __future__ import annotations

import httpx

HF_MODEL = "hamzab/roberta-fake-news-classification"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# Conservative char budget: ~1800 chars ≈ 450 tokens, leaving room for title + special tokens
_MAX_BODY_CHARS = 1800


def classify_with_roberta(title: str, body: str, api_key: str) -> dict:
    """Call the HuggingFace Inference API and return the classifier signal.

    Returns:
        {"label": "FAKE"|"REAL", "score": float}   on success
        {"label": None, "error": str}               on failure
    """
    body_truncated = (body or "")[:_MAX_BODY_CHARS]
    text = f"<title> {title or ''} <content> {body_truncated} <end>"

    try:
        response = httpx.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"inputs": text},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        # API returns [[{"label": "FAKE", "score": 0.95}, ...]]
        scores = data[0] if isinstance(data[0], list) else data
        top = max(scores, key=lambda x: x["score"])
        return {"label": top["label"].upper(), "score": round(top["score"], 3)}
    except Exception as exc:
        return {"label": None, "error": str(exc)}

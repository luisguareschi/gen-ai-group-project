"""HuggingFace RoBERTa fake-news classifier (pre-computed signal for the Judge)."""
from __future__ import annotations

import time

import httpx

HF_MODEL = "hamzab/roberta-fake-news-classification"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

_MAX_BODY_CHARS = 2200
_MAX_RETRIES = 5
_RETRY_WAIT = 20


def classify_with_roberta(title: str, body: str, api_key: str) -> dict:
    """Call the HuggingFace Inference API and return the classifier signal.

    Returns:
        {"label": "FAKE"|"REAL", "score": float}   on success
        {"label": None, "error": str}               on failure
    """
    text = f"{title or ''} {(body or '')}"
    print("[RoBERTa] Calling HuggingFace Inference API...")
    t0 = time.time()

    for attempt in range(1, _MAX_RETRIES + 1):
        current_text = text[:_MAX_BODY_CHARS - (250 * (attempt - 1))]
        try:
            response = httpx.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"inputs": current_text},
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                scores = data[0] if isinstance(data[0], list) else data
                top = max(scores, key=lambda x: x["score"])
                # API returns "TRUE" not "REAL"
                label = "REAL" if top["label"].upper() == "TRUE" else top["label"].upper()
                result = {"label": label, "score": round(top["score"], 3)}
                print(f"[RoBERTa] {result['label']} ({result['score']:.0%}) in {time.time() - t0:.1f}s")
                return result
            elif response.status_code == 503:
                print(f"[RoBERTa] Model loading, waiting {_RETRY_WAIT}s... (attempt {attempt}/{_MAX_RETRIES})")
                time.sleep(_RETRY_WAIT)
            elif response.status_code == 400:
                print(f"[RoBERTa] HTTP 400 — truncating further and retrying... (attempt {attempt}/{_MAX_RETRIES})")
                time.sleep(2)
            else:
                print(f"[RoBERTa] HTTP {response.status_code} in {time.time() - t0:.1f}s — {response.text[:120]}")
                return {"label": None, "error": f"HTTP {response.status_code}"}
        except Exception as exc:
            print(f"[RoBERTa] Exception on attempt {attempt}/{_MAX_RETRIES}: {exc}")
            time.sleep(_RETRY_WAIT)

    print(f"[RoBERTa] Failed after {_MAX_RETRIES} attempts in {time.time() - t0:.1f}s")
    return {"label": None, "error": f"Failed after {_MAX_RETRIES} attempts"}

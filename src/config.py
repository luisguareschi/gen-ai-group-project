"""Central configuration loaded from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass
class Settings:
    backend: str
    ollama_model: str
    ollama_base_url: str
    openai_api_key: str | None
    openai_model: str
    temperature: float
    max_claims: int
    max_article_chars: int


def get_settings() -> Settings:
    return Settings(
        backend=os.getenv("LLM_BACKEND", "ollama").lower(),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        max_claims=int(os.getenv("MAX_CLAIMS", "3")),
        max_article_chars=int(os.getenv("MAX_ARTICLE_CHARS", "2000")),
    )


settings = get_settings()

"""Factory for the CrewAI LLM based on configuration."""
from __future__ import annotations

from crewai import LLM

from .config import Settings, settings as default_settings


def build_llm(settings: Settings | None = None) -> LLM:
    s = settings or default_settings
    if s.backend == "ollama":
        return LLM(
            model=f"ollama/{s.ollama_model}",
            base_url=s.ollama_base_url,
            temperature=s.temperature,
        )
    if s.backend == "openai":
        if not s.openai_api_key:
            raise RuntimeError(
                "LLM_BACKEND=openai but OPENAI_API_KEY is not set in the environment."
            )
        return LLM(model=s.openai_model, temperature=s.temperature)
    raise ValueError(f"Unknown LLM backend: {s.backend!r}")

"""CrewAI-compatible DuckDuckGo search tool for the Fact Checker agent."""
from __future__ import annotations

from typing import Type

from crewai.tools import BaseTool
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field


class DuckDuckGoSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="A search query to verify a factual claim against recent web sources.",
    )
    max_results: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum number of web results to return (1-5).",
    )


class DuckDuckGoSearchTool(BaseTool):
    name: str = "duckduckgo_search"
    description: str = (
        "Search the web via DuckDuckGo and return snippets from the top results. "
        "Use this when Wikipedia does not return enough information to verify a claim, "
        "or when the claim is about a recent event. "
        "Input: a JSON object with 'query' (string) and optional 'max_results' (int, 1-5)."
    )
    args_schema: Type[BaseModel] = DuckDuckGoSearchInput

    def _run(self, query: str, max_results: int = 2) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results, timeout=5))
        except Exception as exc:
            return f"DuckDuckGo search failed: {exc}"

        if not results:
            return f"No DuckDuckGo results for query: {query!r}"

        blocks = [
            f"[{r['title']}]\n{r['body']}"
            for r in results
            if r.get("title") and r.get("body")
        ]
        return "\n\n".join(blocks) if blocks else f"No readable results for query: {query!r}"

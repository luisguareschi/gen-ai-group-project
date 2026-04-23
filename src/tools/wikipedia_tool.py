"""CrewAI-compatible Wikipedia search tool for the Fact Checker agent."""
from __future__ import annotations

import sys
import time
from typing import Type

import wikipedia
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class WikipediaSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="A short search query describing the factual claim to check (e.g. entity, event, date).",
    )
    max_results: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Maximum number of Wikipedia articles to summarize (1-5).",
    )


class WikipediaSearchTool(BaseTool):
    name: str = "wikipedia_search"
    description: str = (
        "Search Wikipedia and return short summaries of the top matching articles. "
        "Use this to verify factual claims about people, places, events, or statistics. "
        "Input: a JSON object with 'query' (string) and optional 'max_results' (int, 1-5)."
    )
    args_schema: Type[BaseModel] = WikipediaSearchInput

    def _run(self, query: str, max_results: int = 2) -> str:
        print(f"[Wikipedia] Searching: {query!r}", file=sys.stderr)
        t0 = time.time()
        try:
            titles = wikipedia.search(query, results=max_results)
        except Exception as exc:  # pragma: no cover - network failure path
            print(f"[Wikipedia] Failed in {time.time() - t0:.1f}s — {exc}", file=sys.stderr)
            return f"Wikipedia search failed: {exc}"

        if not titles:
            print(f"[Wikipedia] No results in {time.time() - t0:.1f}s", file=sys.stderr)
            return f"No Wikipedia results for query: {query!r}"

        blocks: list[str] = []
        for title in titles[:max_results]:
            try:
                summary = wikipedia.summary(title, sentences=2, auto_suggest=False, redirect=True)
            except wikipedia.DisambiguationError as exc:
                options = ", ".join(exc.options[:5])
                blocks.append(f"[{title}] Disambiguation. Options: {options}")
                continue
            except wikipedia.PageError:
                continue
            except Exception as exc:  # pragma: no cover
                blocks.append(f"[{title}] Error: {exc}")
                continue
            blocks.append(f"[{title}]\n{summary}")

        if not blocks:
            print(f"[Wikipedia] No readable pages in {time.time() - t0:.1f}s", file=sys.stderr)
            return f"No readable Wikipedia pages for query: {query!r}"
        print(f"[Wikipedia] {len(blocks)} result(s) in {time.time() - t0:.1f}s", file=sys.stderr)
        return "\n\n".join(blocks)

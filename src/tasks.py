"""Task builders for the sequential pipeline."""
from __future__ import annotations

from crewai import Agent, Task

from .config import settings
from .schemas import BiasOutput, ClaimsOutput, FactCheckOutput, JudgeOutput


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + " [...]"


def build_tasks(agents: dict[str, Agent], title: str, body: str) -> list[Task]:
    """Build the 4-task sequential pipeline for a single article."""
    title = (title or "").strip()
    body = _truncate(body, settings.max_article_chars)
    max_claims = settings.max_claims

    article_block = f"TITLE: {title}\n\nBODY:\n{body}"

    t1 = Task(
        description=(
            f"Read the news article below and extract up to {max_claims} concise, "
            "VERIFIABLE factual claims. Focus on statistics, named entities, "
            "concrete events, and dates. Ignore opinions and rhetorical language.\n\n"
            f"{article_block}\n\n"
            f"Return a JSON object matching the ClaimsOutput schema with exactly "
            f"{max_claims} items in 'claims' (or fewer if the article has fewer verifiable claims)."
        ),
        agent=agents["claim_extractor"],
        expected_output=(
            'A JSON object like {"claims": ["claim 1", "claim 2", "claim 3"]} '
            "containing short, verifiable factual claims."
        ),
        output_pydantic=ClaimsOutput,
    )

    t2 = Task(
        description=(
            "You will receive a JSON list of factual claims from the previous step. "
            "For EACH claim, call the wikipedia_search tool with a focused query, "
            "read the returned summaries, and decide a verdict:\n"
            "  - SUPPORTED: Wikipedia clearly confirms the claim.\n"
            "  - CONTRADICTED: Wikipedia clearly disproves the claim.\n"
            "  - UNVERIFIABLE: Wikipedia does not contain enough information.\n\n"
            "For each claim produce an object with: claim, verdict, confidence (0-1), "
            "evidence (a 1-2 sentence justification citing the Wikipedia page title).\n\n"
            "Do NOT fabricate evidence. If in doubt, use UNVERIFIABLE."
        ),
        agent=agents["fact_checker"],
        expected_output=(
            'A JSON object like {"results": [{"claim": "...", "verdict": "SUPPORTED", '
            '"confidence": 0.8, "evidence": "..."}]} with one entry per claim.'
        ),
        output_pydantic=FactCheckOutput,
    )

    t3 = Task(
        description=(
            "Analyze the BIAS, TONE, and FRAMING of the following news article. "
            "Do NOT evaluate the factual correctness - only the language. "
            "Look for emotionally loaded words, sensationalist headlines, "
            "one-sided framing, unattributed claims, missing counter-arguments, "
            "and ad-hominem attacks.\n\n"
            f"{article_block}\n\n"
            "Return a JSON object with: tone (single word), bias_score (0=neutral, "
            "1=highly biased), flags (list of short strings describing specific issues found)."
        ),
        agent=agents["bias_detector"],
        expected_output=(
            'A JSON object like {"tone": "sensationalist", "bias_score": 0.7, '
            '"flags": ["Emotionally loaded headline", "No counter-arguments"]}.'
        ),
        output_pydantic=BiasOutput,
    )

    t4 = Task(
        description=(
            "You are the Judge. Using the outputs of the previous three agents "
            "(claim extraction, fact-checking verdicts, and bias analysis), "
            "produce a FINAL classification of the article as REAL or FAKE.\n\n"
            "Decision heuristic:\n"
            "  - If most claims are CONTRADICTED -> likely FAKE.\n"
            "  - If most claims are SUPPORTED and bias_score is low -> likely REAL.\n"
            "  - High bias_score alone is a weak signal (REAL articles can be biased).\n"
            "  - UNVERIFIABLE claims are neutral; weigh them less.\n\n"
            "Return a JSON object with: label ('REAL' or 'FAKE'), confidence (0-1), "
            "summary (2-3 sentence human-readable explanation), and "
            "agent_inputs_summary {claim_verdicts: [...], bias_score: float, tone: str}."
        ),
        agent=agents["judge"],
        expected_output=(
            'A JSON object like {"label": "FAKE", "confidence": 0.82, '
            '"summary": "...", "agent_inputs_summary": {"claim_verdicts": '
            '["CONTRADICTED", "UNVERIFIABLE", "CONTRADICTED"], "bias_score": 0.7, '
            '"tone": "sensationalist"}}.'
        ),
        output_pydantic=JudgeOutput,
        context=[t1, t2, t3],
    )

    return [t1, t2, t3, t4]

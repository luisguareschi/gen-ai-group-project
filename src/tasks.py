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
            "Do NOT evaluate factual correctness — only the language and presentation.\n\n"
            "Score the article on this scale:\n"
            "  0.0–0.2  Neutral: dry, factual, wire-service style (Reuters/AP)\n"
            "  0.3–0.5  Mild bias: editorial framing, some loaded language\n"
            "  0.6–0.7  Moderate bias: clear agenda, emotional appeals\n"
            "  0.8–1.0  High bias: tabloid/propaganda style, inflammatory\n\n"
            "Tone must be one of: neutral, balanced, skeptical, editorial, "
            "sensationalist, inflammatory, alarmist\n\n"
            "Check for each of the following and list only the ones actually present:\n"
            "  - ALL CAPS or hyperbolic words in headline (SHOCKING, BREAKING, EXPOSED)\n"
            "  - Emotionally loaded adjectives (disastrous, radical, corrupt)\n"
            "  - Unattributed claims ('sources say', 'reportedly', 'many believe')\n"
            "  - One-sided framing with no counter-argument or expert dissent\n"
            "  - Ad-hominem attacks on individuals rather than evidence-based criticism\n"
            "  - Calls to action or outrage ('you should be angry', 'share this now')\n\n"
            "IMPORTANT: the bias_score MUST reflect what is actually in the article. "
            "A neutral wire-service article should score 0.1–0.2. "
            "Do not default to 0.7 — only score that high if multiple flags are present.\n\n"
            f"{article_block}\n\n"
            "Return a JSON object with: tone (one word from the list above), "
            "bias_score (0.0–1.0), flags (list of specific issues found, empty list if none)."
        ),
        agent=agents["bias_detector"],
        expected_output=(
            'A JSON object like {"tone": "neutral", "bias_score": 0.15, "flags": []} '
            'for a wire-service article, or {"tone": "sensationalist", "bias_score": 0.85, '
            '"flags": ["ALL CAPS headline", "Unattributed claims", "No counter-arguments"]} '
            "for a biased one."
        ),
        output_pydantic=BiasOutput,
    )

    t4 = Task(
        description=(
            "You are the Judge. Using the outputs of the previous three agents "
            "(claim extraction, fact-checking verdicts, and bias analysis), "
            "produce a FINAL classification of the article as REAL or FAKE.\n\n"
            "Step 1 — Count the fact-check verdicts:\n"
            "  supported    = number of SUPPORTED claims\n"
            "  contradicted = number of CONTRADICTED claims\n"
            "  unverifiable = number of UNVERIFIABLE claims\n\n"
            "Step 2 — Compute a base confidence using this formula:\n"
            "  total_verifiable = supported + contradicted\n"
            "  If total_verifiable > 0:\n"
            "    base = (supported - contradicted) / total_verifiable\n"
            "    confidence = 0.5 + abs(base) * 0.5   (maps -1..1 to 0.5..1.0)\n"
            "    label = REAL if base >= 0 else FAKE\n"
            "  If total_verifiable == 0 (all UNVERIFIABLE):\n"
            "    Use bias_score as fallback:\n"
            "      bias_score >= 0.6 -> label=FAKE, confidence = 0.5 + (bias_score - 0.6) * 0.5\n"
            "      bias_score <  0.6 -> label=REAL, confidence = 0.5 + (0.6 - bias_score) * 0.5\n\n"
            "Step 3 — Apply adjustments:\n"
            "  - If label=REAL and bias_score >= 0.7: reduce confidence by 0.05 "
            "(bias is a weak signal but adds uncertainty)\n"
            "  - confidence must always be in range 0.5–1.0\n\n"
            "Step 4 — Write a 2–3 sentence summary explaining the verdict. "
            "Reference specific verdicts and the bias score. "
            "Do not use generic filler like 'the article may be fake'.\n\n"
            "Return a JSON object with: label ('REAL' or 'FAKE'), "
            "confidence (0.5–1.0, calculated above), "
            "summary (2–3 sentences), and "
            "agent_inputs_summary {claim_verdicts: [...], bias_score: float, tone: str}."
        ),
        agent=agents["judge"],
        expected_output=(
            'A JSON object like {"label": "FAKE", "confidence": 0.95, '
            '"summary": "Two of three claims were directly contradicted by Wikipedia. '
            "The third was unverifiable. Combined with a high bias score of 0.8, "
            'this article is almost certainly fabricated.", '
            '"agent_inputs_summary": {"claim_verdicts": '
            '["CONTRADICTED", "UNVERIFIABLE", "CONTRADICTED"], "bias_score": 0.8, '
            '"tone": "sensationalist"}}.'
        ),
        output_pydantic=JudgeOutput,
        context=[t1, t2, t3],
    )

    return [t1, t2, t3, t4]

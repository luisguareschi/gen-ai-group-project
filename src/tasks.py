"""Task builders for the sequential pipeline."""
from __future__ import annotations

from typing import Literal

from crewai import Agent, Task

from .config import settings
from .schemas import BiasOutput, ClaimsOutput, FactCheckOutput, JudgeOutput


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + " [...]"


def build_search_tasks(agents: dict[str, Agent], title: str, body: str) -> list[Task]:
    """Build t1–t3: claim extraction, fact-checking (search + verdict), and bias detection."""
    title = (title or "").strip()
    body = _truncate(body, settings.max_article_chars)
    max_claims = settings.max_claims

    article_block = f"TITLE: {title}\n\nBODY:\n{body}"

    t1 = Task(
        description=(
            "Below is a news article. Read it carefully.\n\n"
            f"{article_block}\n\n"
            "--- END OF ARTICLE ---\n\n"
            f"Your task: extract up to {max_claims} factual claims FROM THE ARTICLE ABOVE. "
            "Do not use any knowledge from outside the article. "
            "If a claim is not explicitly stated in the text above, do not include it.\n\n"
            "Each claim MUST meet all of these criteria:\n"
            "  1. Directly stated in the article text — not inferred, implied, or "
            "drawn from general knowledge.\n"
            "  2. A complete sentence using the article's own language where possible, "
            "without adding or substituting information not in the original text.\n"
            "  3. Contains at least one of: a named entity (person, organisation, place, "
            "publication, law), a date or time period, a number or statistic, OR a "
            "specific factual assertion about a known concept, scientific finding, "
            "legal standard, or established process.\n"
            "  4. Makes a falsifiable assertion — something that can be confirmed or "
            "contradicted by looking it up.\n"
            "  5. Could stand alone as a fact-check query — a reader with no context "
            "could look it up and find a clear answer.\n\n"
            "Do NOT include:\n"
            "  - Anything not present in the article text\n"
            "  - Opinions or value judgements unless directly attributed to a named "
            "person or institution ('the policy was a disaster' → exclude; "
            "'Senator Smith called the policy disastrous in a Senate hearing' → include)\n"
            "  - Commentary phrases that are not falsifiable assertions "
            "('So much for...', 'It's like...', 'Check out...', 'Can you believe...') — "
            "even if they contain a named entity, these are editorial, not factual\n"
            "  - Vague generalisations ('many people believe...', 'experts say...')\n"
            "  - Restatements of the headline\n"
            "  - Predictions or hypotheticals\n"
            "  - Direct quotes presented as standalone claims — extract the underlying "
            "factual assertion instead\n"
            "  - Peripheral background facts that merely identify a person's profession, "
            "age, or affiliation, or confirm a well-known historical detail "
            "(e.g. 'Smith is a lawyer', 'The organisation was founded in 1954', "
            "'Jones had previously worked at Company X') when these are cited only as "
            "context and are not the article's central claim — only include these if no "
            "stronger central claims exist\n\n"
            f"When you have more qualifying claims than the {max_claims}-claim limit, "
            "PRIORITISE claims that are central to the article's main argument — claims "
            "whose truth or falsity directly supports or undermines what the article is "
            "asserting. DE-PRIORITISE peripheral background facts that merely identify "
            "who a person is or confirm well-known historical details cited only as "
            "context.\n\n"
            f"If the article contains fewer than {max_claims} claims that meet these "
            "criteria, return only the valid ones — do not pad with weak claims.\n\n"
            f"Return a JSON object with a 'claims' list containing 1–{max_claims} strings."
        ),
        agent=agents["claim_extractor"],
        expected_output=(
            'A JSON object like {"claims": ["Harvard University scientists published a '
            'study in the Journal of Dairy Astronomy claiming the Moon is made of cheddar cheese.", '
            '"Apollo 11 astronauts returned lunar samples in July 1969 that have been '
            'publicly documented by NASA."]} '
            "where each claim is drawn directly from the article text."
        ),
        output_pydantic=ClaimsOutput,
    )

    t2a = Task(
        description=(
            "You will receive a list of factual claims from the previous step. "
            "Your ONLY job right now is to search for evidence — do NOT produce verdicts yet.\n\n"
            "For EACH claim: call all available tools for every claim. Do not skip any tool.\n"
            "Do not write any verdicts or JSON. Just return the raw search results "
            "as plain text, clearly labelled by claim number."
        ),
        agent=agents["fact_checker"],
        expected_output=(
            "Plain text search results for each claim, e.g.:\n"
            "CLAIM 1: <claim text>\n"
            "Wikipedia: <summary>\n"
            "DuckDuckGo: <snippets>\n\n"
            "CLAIM 2: ..."
        ),
    )

    t2b = Task(
        description=(
            "You will receive raw search results for a list of factual claims. "
            "Do NOT call any tools. Your only job is to read the search results "
            "already provided and assign a verdict to each claim.\n\n"
            "For each claim assign one of:\n"
            "  - SUPPORTED: at least one source clearly confirms the claim.\n"
            "  - CONTRADICTED: at least one source clearly disproves the claim.\n"
            "  - UNVERIFIABLE: the search results do not contain enough information.\n\n"
            "For each claim produce: claim, verdict, confidence (0-1), "
            "evidence (1-2 sentences citing the source).\n\n"
            "Do NOT fabricate evidence. If in doubt, use UNVERIFIABLE."
        ),
        agent=agents["fact_checker"],
        expected_output=(
            'A JSON object like {"results": [{"claim": "...", "verdict": "SUPPORTED", '
            '"confidence": 0.8, "evidence": "..."}]} with one entry per claim.'
        ),
        output_pydantic=FactCheckOutput,
        context=[t2a],
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

    return [t1, t2a, t2b, t3]


def build_judge_task(
    agents: dict[str, Agent],
    label: Literal["REAL", "FAKE"],
    confidence: float,
    claim_verdicts: list[str],
    bias_score: float,
    tone: str,
    roberta_result: dict | None = None,
) -> Task:
    """Build the Judge task given pre-computed verdict values.

    The Judge's only job is to write a human-readable summary — all arithmetic
    has already been done in Python by _compute_verdict().
    """
    verdicts_str = ", ".join(claim_verdicts) if claim_verdicts else "none"

    if roberta_result and roberta_result.get("label"):
        roberta_line = (
            f"RoBERTa signal : {roberta_result['label']} "
            f"(confidence: {roberta_result['score']:.0%})"
        )
    else:
        roberta_line = "RoBERTa signal : unavailable"

    return Task(
        description=(
            "The pipeline has already computed the final verdict deterministically.\n\n"
            "Pre-computed values (do NOT change these):\n"
            f"  label      = {label}\n"
            f"  confidence = {confidence:.3f}\n\n"
            "Signals used to reach this verdict:\n"
            f"  Claim verdicts : {verdicts_str}\n"
            f"  Bias score     : {bias_score:.2f} (tone: {tone})\n"
            f"  {roberta_line}\n\n"
            "Your ONLY task: write a 2–3 sentence summary explaining this verdict to a reader. "
            "Reference the specific claim verdicts, the bias score, and the RoBERTa signal if available. "
            "Do not recalculate anything. Do not use generic filler like 'the article may be fake'.\n\n"
            "Return a JSON object with:\n"
            f"  label              = '{label}' (exact, do not change)\n"
            f"  confidence         = {confidence:.3f} (exact, do not change)\n"
            "  summary            = your 2–3 sentence explanation\n"
            f"  agent_inputs_summary = {{claim_verdicts: {claim_verdicts!r}, "
            f"bias_score: {bias_score}, tone: '{tone}'}}"
        ),
        agent=agents["judge"],
        expected_output=(
            f'A JSON object like {{"label": "{label}", "confidence": {confidence:.3f}, '
            '"summary": "Two claims were UNVERIFIABLE and one CONTRADICTED by Wikipedia. '
            f'The bias score of {bias_score:.2f} indicates sensationalist framing. '
            'Combined, these signals strongly indicate fabricated content.", '
            f'"agent_inputs_summary": {{"claim_verdicts": {claim_verdicts!r}, '
            f'"bias_score": {bias_score}, "tone": "{tone}"}}}}'
        ),
        output_pydantic=JudgeOutput,
    )

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
            "Step 2 — Compute the fact-check signal:\n"
            "  score  = (supported * 1.0) + (contradicted * -1.0)\n"
            "  weight = (supported * 1.0) + (unverifiable * 0.5) + (contradicted * 1.0)\n"
            "  fact_signal = score / weight if weight > 0 else 0.0\n\n"
            "Step 3 — Compute the bias signal:\n"
            "  bias_signal = -(bias_score - 0.45)\n"
            "  (neutral article bias=0.45 → 0.0 effect,\n"
            "   biased article bias=1.0 → -0.55 FAKE push,\n"
            "   neutral article bias=0.0 → +0.45 REAL push)\n\n"
            "Step 4 — Combine both signals (fact-checking 60%, bias 40%):\n"
            "  combined = (fact_signal * 0.6) + (bias_signal * 0.4)\n"
            "  confidence = 0.5 + abs(combined) * 0.5\n"
            "  label = FAKE if combined < 0 else REAL\n"
            "  (When combined = 0.0 exactly, default to REAL)\n"
            "  confidence must always be in range 0.5–1.0\n\n"
            "  Worked examples:\n"
            "    4 SUP, 1 UNVER, bias=0.2 -> fact=+0.89, bias=+0.25, combined=+0.63 -> REAL, confidence=0.82\n"
            "    2 CONT, bias=0.85        -> fact=-1.0,  bias=-0.40, combined=-0.76 -> FAKE, confidence=0.88\n"
            "    2 SUP, 2 UNVER, bias=0.65-> fact=+0.67, bias=-0.20, combined=+0.32 -> REAL, confidence=0.66\n"
            "    0 SUP, 4 UNVER, bias=0.6 -> fact=0.0,   bias=-0.15, combined=-0.06 -> FAKE, confidence=0.53\n"
            "    0 SUP, 4 UNVER, bias=0.2 -> fact=0.0,   bias=+0.25, combined=+0.10 -> REAL, confidence=0.55\n\n"
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

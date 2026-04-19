"""CrewAI agent definitions for the fake-news detection pipeline."""
from __future__ import annotations

from crewai import Agent, LLM

from .tools import WikipediaSearchTool


def build_agents(llm: LLM) -> dict[str, Agent]:
    """Create the four agents that make up the sequential pipeline."""
    claim_extractor = Agent(
        role="Claim Extractor",
        goal="Extract a short list of verifiable factual claims from a news article.",
        backstory=(
            "You are a veteran investigative journalist trained at a fact-checking "
            "organization. You are precise, skeptical, and only interested in "
            "statements that can be verified against external sources: statistics, "
            "named entities, events, dates, and concrete actions by public figures."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    fact_checker = Agent(
        role="Fact Checker",
        goal=(
            "For each extracted claim, search Wikipedia and decide whether the claim "
            "is SUPPORTED, CONTRADICTED, or UNVERIFIABLE given the evidence."
        ),
        backstory=(
            "You are a senior researcher at a fact-checking organization. "
            "You never fabricate evidence. If Wikipedia does not contain enough "
            "information, you mark a claim as UNVERIFIABLE instead of guessing."
        ),
        tools=[WikipediaSearchTool()],
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    bias_detector = Agent(
        role="Bias Detector",
        goal=(
            "Analyze the language, tone, and framing of a news article and produce "
            "a structured bias assessment."
        ),
        backstory=(
            "You are a linguistics professor specializing in media studies and "
            "computational rhetoric. You detect emotional manipulation, loaded "
            "language, sensationalist framing, and missing counter-arguments."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    judge = Agent(
        role="Judge",
        goal=(
            "Aggregate the claim verdicts and bias analysis into a final REAL/FAKE "
            "classification with a confidence score and a clear explanation."
        ),
        backstory=(
            "You are the chief editor of a fact-checking platform. You synthesize "
            "evidence from your team and deliver clear, justified verdicts that a "
            "reader can understand."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    return {
        "claim_extractor": claim_extractor,
        "fact_checker": fact_checker,
        "bias_detector": bias_detector,
        "judge": judge,
    }

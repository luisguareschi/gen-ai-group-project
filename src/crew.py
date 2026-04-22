"""High-level entry point: build and run the CrewAI pipeline for one article."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from crewai import Crew, Process

from .agents import build_agents
from .config import settings
from .llm import build_llm
from .parsing import extract_json
from .schemas import BiasOutput, ClaimsOutput, FactCheckOutput, JudgeOutput
from .tasks import build_tasks
from .tools.roberta_classifier import classify_with_roberta


@dataclass
class PipelineResult:
    claims: ClaimsOutput | None
    fact_check: FactCheckOutput | None
    bias: BiasOutput | None
    judge: JudgeOutput | None
    raw: dict[str, Any]

    def to_row(self, article_id: str | int | None = None) -> dict[str, Any]:
        """Flatten into a CSV-ready row."""
        row: dict[str, Any] = {"article_id": article_id}
        row["claims"] = self.claims.model_dump_json() if self.claims else ""
        row["fact_check"] = self.fact_check.model_dump_json() if self.fact_check else ""
        row["bias"] = self.bias.model_dump_json() if self.bias else ""
        if self.judge is not None:
            row["label"] = self.judge.label
            row["confidence"] = self.judge.confidence
            row["summary"] = self.judge.summary
            row["bias_score"] = self.judge.agent_inputs_summary.bias_score
            row["tone"] = self.judge.agent_inputs_summary.tone
        else:
            row["label"] = ""
            row["confidence"] = ""
            row["summary"] = ""
            row["bias_score"] = ""
            row["tone"] = ""
        return row


def _coerce(task_output: Any, model):
    """Extract a Pydantic model from a CrewAI TaskOutput."""
    if task_output is None:
        return None
    pyd = getattr(task_output, "pydantic", None)
    if isinstance(pyd, model):
        return pyd
    raw = getattr(task_output, "raw", None) or str(task_output)
    try:
        data = extract_json(raw)
        return model.model_validate(data)
    except Exception:
        return None


def run_pipeline(title: str, body: str) -> PipelineResult:
    """Run the full 4-agent sequential pipeline on a single article."""
    llm = build_llm()
    agents = build_agents(llm)

    # Run RoBERTa classifier before the crew — result is injected into the Judge's context
    roberta_result = None
    if settings.huggingface_api_key:
        roberta_result = classify_with_roberta(title, body, settings.huggingface_api_key)

    tasks = build_tasks(agents, title, body, roberta_result=roberta_result)

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    crew_output = crew.kickoff()

    t1_out, t2_out, t3_out, t4_out = tasks[0].output, tasks[1].output, tasks[2].output, tasks[3].output

    return PipelineResult(
        claims=_coerce(t1_out, ClaimsOutput),
        fact_check=_coerce(t2_out, FactCheckOutput),
        bias=_coerce(t3_out, BiasOutput),
        judge=_coerce(t4_out, JudgeOutput),
        raw={"crew_output": str(crew_output)},
    )

"""High-level entry point: build and run the CrewAI pipeline for one article."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from crewai import Crew, Process

from .agents import build_agents
from .config import settings
from .llm import build_llm
from .parsing import extract_json
from .schemas import BiasOutput, ClaimsOutput, FactCheckOutput, JudgeOutput
from .tasks import build_judge_task, build_search_tasks
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


def _compute_verdict(
    fact_check: FactCheckOutput | None,
    bias: BiasOutput | None,
    roberta_result: dict | None,
) -> tuple[Literal["REAL", "FAKE"], float, list[str]]:
    """Deterministically compute label, confidence, and verdict list from agent outputs."""
    verdicts = [r.verdict for r in (fact_check.results if fact_check else [])]
    supported = verdicts.count("SUPPORTED")
    contradicted = verdicts.count("CONTRADICTED")
    unverifiable = verdicts.count("UNVERIFIABLE")

    score = supported * 1.0 + contradicted * -1.0
    weight = supported * 1.0 + unverifiable * 0.5 + contradicted * 1.0
    fact_signal = score / weight if weight > 0 else 0.0

    bias_score = bias.bias_score if bias else 0.45
    bias_signal = -(bias_score - 0.45)

    if roberta_result and roberta_result.get("label"):
        roberta_signal = 1.0 if roberta_result["label"] == "REAL" else -1.0
        roberta_weight = roberta_result["score"]
    else:
        roberta_signal = 0.0
        roberta_weight = 0.0

    combined = fact_signal * 0.60 + bias_signal * 0.35 + roberta_signal * roberta_weight * 0.05
    confidence = round(min(1.0, max(0.5, 0.5 + abs(combined) * 0.5)), 3)
    label: Literal["REAL", "FAKE"] = "FAKE" if combined < 0 else "REAL"

    print(
        f"[Scorer] fact_signal={fact_signal:.3f}  bias_signal={bias_signal:.3f}  "
        f"roberta={roberta_signal * roberta_weight:.3f}  combined={combined:.3f}  "
        f"→ {label} ({confidence:.1%})"
    )
    return label, confidence, verdicts


def run_pipeline(title: str, body: str) -> PipelineResult:
    """Run the full pipeline: search+analysis crew → Python scoring → Judge summary."""
    llm = build_llm()
    agents = build_agents(llm)

    roberta_result = None
    if settings.huggingface_api_key:
        roberta_result = classify_with_roberta(title, body, settings.huggingface_api_key)
    else:
        print("[RoBERTa] Skipped — HUGGINGFACE_API_KEY not set in .env")

    # Phase 1: claim extraction, fact-checking (search + verdict), bias detection
    search_tasks = build_search_tasks(agents, title, body)
    crew1 = Crew(
        agents=list(agents.values()),
        tasks=search_tasks,
        process=Process.sequential,
        verbose=True,
    )
    crew1.kickoff()

    t1_out = _coerce(search_tasks[0].output, ClaimsOutput)
    t2_out = _coerce(search_tasks[2].output, FactCheckOutput)  # index 2 = t2b (verdicts)
    t3_out = _coerce(search_tasks[3].output, BiasOutput)

    # Phase 2: deterministic scoring — no LLM involved
    label, confidence, claim_verdicts = _compute_verdict(t2_out, t3_out, roberta_result)

    # Phase 3: Judge writes a human-readable summary only
    bias_score = t3_out.bias_score if t3_out else 0.45
    tone = t3_out.tone if t3_out else "unknown"
    judge_task = build_judge_task(
        agents, label, confidence, claim_verdicts, bias_score, tone, roberta_result
    )
    crew2 = Crew(
        agents=[agents["judge"]],
        tasks=[judge_task],
        process=Process.sequential,
        verbose=True,
    )
    crew2.kickoff()

    t4_out = _coerce(judge_task.output, JudgeOutput)

    return PipelineResult(
        claims=t1_out,
        fact_check=t2_out,
        bias=t3_out,
        judge=t4_out,
        raw={"label": label, "confidence": confidence},
    )

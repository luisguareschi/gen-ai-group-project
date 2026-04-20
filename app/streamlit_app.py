"""Streamlit UI for the Multi-Agent Fake News Detection System."""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import settings as default_settings  # noqa: E402
from src.crew import run_pipeline  # noqa: E402


st.set_page_config(page_title="Fake News Detector", page_icon=":mag:", layout="wide")


def sidebar_controls() -> None:
    st.sidebar.title("Settings")
    backend = st.sidebar.selectbox(
        "LLM backend",
        ["ollama", "openai"],
        index=0 if default_settings.backend == "ollama" else 1,
    )
    os.environ["LLM_BACKEND"] = backend

    if backend == "ollama":
        ollama_models = [
            "qwen3:8b",
            "qwen2.5:7b",
            "qwen2.5:3b",
            "gemma3:4b",
        ]
        default_model = default_settings.ollama_model
        options = ollama_models if default_model in ollama_models else [default_model] + ollama_models
        model = st.sidebar.selectbox("Ollama model", options, index=options.index(default_model))
        st.sidebar.caption(
            "Larger models (7b, 8b) produce more reliable JSON and better reasoning "
            "but are slower. Smaller models (3b, 4b) are faster but may produce "
            "inconsistent outputs or miss nuanced claims."
        )
        base_url = st.sidebar.text_input("Ollama base URL", value=default_settings.ollama_base_url)
        os.environ["OLLAMA_MODEL"] = model
        os.environ["OLLAMA_BASE_URL"] = base_url
    else:
        model = st.sidebar.text_input("OpenAI model", value=default_settings.openai_model)
        api_key = st.sidebar.text_input("OpenAI API key", type="password")
        os.environ["OPENAI_MODEL"] = model
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    temperature = st.sidebar.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=default_settings.temperature, step=0.05
    )
    st.sidebar.caption(
        "Lower values (0.1–0.3) make outputs more deterministic and consistent — "
        "recommended for fact-checking. Higher values increase creativity but reduce "
        "reliability of structured JSON outputs."
    )
    os.environ["LLM_TEMPERATURE"] = str(temperature)

    max_claims = st.sidebar.slider("Max claims", 1, 5, default_settings.max_claims)
    st.sidebar.caption(
        "More claims give the Judge stronger signal but increase runtime roughly "
        "linearly — each claim triggers a Wikipedia and potentially a web search. "
        "3–4 is the recommended range for speed vs. accuracy."
    )
    os.environ["MAX_CLAIMS"] = str(max_claims)

    st.sidebar.caption("Make sure `ollama serve` is running if using Ollama.")


def verdict_card(label: str, confidence: float, summary: str) -> None:
    color = "#16a34a" if label == "REAL" else "#dc2626"
    st.markdown(
        f"""
<div style='border:2px solid {color};border-radius:12px;padding:16px;background:{color}15'>
  <div style='font-size:14px;color:{color};font-weight:600;letter-spacing:0.1em'>FINAL VERDICT</div>
  <div style='font-size:36px;font-weight:800;color:{color}'>{label}</div>
  <div style='font-size:14px;color:#555;margin-top:8px'>Confidence: <b>{confidence:.0%}</b></div>
  <div style='margin-top:12px;font-size:15px'>{summary}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_result(result) -> None:
    judge = result.judge
    if judge is not None:
        verdict_card(judge.label, float(judge.confidence), judge.summary)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Bias score", f"{judge.agent_inputs_summary.bias_score:.2f}")
        with col2:
            st.metric("Tone", judge.agent_inputs_summary.tone)
    else:
        st.error("Judge did not return a parseable verdict. See raw output below.")

    with st.expander("Agent 1 - Claim Extractor"):
        if result.claims is not None:
            for i, c in enumerate(result.claims.claims, 1):
                st.markdown(f"**{i}.** {c}")
        else:
            st.write("(no output)")

    with st.expander("Agent 2 - Fact Checker"):
        if result.fact_check is not None:
            for r in result.fact_check.results:
                icon = {"SUPPORTED": ":white_check_mark:", "CONTRADICTED": ":x:", "UNVERIFIABLE": ":grey_question:"}.get(r.verdict, "")
                st.markdown(f"{icon} **{r.verdict}** ({r.confidence:.0%}) - {r.claim}")
                st.caption(r.evidence)
        else:
            st.write("(no output)")

    with st.expander("Agent 3 - Bias Detector"):
        if result.bias is not None:
            st.write(f"**Tone:** {result.bias.tone}")
            st.write(f"**Bias score:** {result.bias.bias_score:.2f}")
            if result.bias.flags:
                st.write("**Flags:**")
                for f in result.bias.flags:
                    st.markdown(f"- {f}")
        else:
            st.write("(no output)")

    with st.expander("Raw crew output"):
        st.code(result.raw.get("crew_output", ""), language="text")


def single_article_tab() -> None:
    st.subheader("Analyze a single article")
    uploaded = st.file_uploader("Upload a .txt file (optional)", type=["txt"], key="single_file")
    default_body = ""
    if uploaded is not None:
        default_body = uploaded.read().decode("utf-8", errors="ignore")
    title = st.text_input("Title", value="")
    body = st.text_area("Body", value=default_body, height=300, placeholder="Paste the article body here...")

    if st.button("Analyze", type="primary", disabled=not (title or body)):
        with st.spinner("Running agents..."):
            start = time.time()
            result = run_pipeline(title, body)
            st.caption(f"Completed in {time.time() - start:.1f}s")
        render_result(result)


def batch_tab() -> None:
    st.subheader("Batch analysis")
    st.caption("Upload a CSV with `title` and `text` columns. Each row will be processed sequentially.")
    up = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")
    if up is None:
        return

    df = pd.read_csv(up)
    missing = {"title", "text"} - set(df.columns)
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        return
    st.write(f"Loaded {len(df)} rows.")
    st.dataframe(df.head())

    if st.button("Run batch", type="primary"):
        progress = st.progress(0.0)
        status = st.empty()
        rows: list[dict] = []
        for i, rec in enumerate(df.itertuples(index=False), 1):
            status.write(f"Processing {i}/{len(df)}: {str(getattr(rec, 'title', ''))[:80]}")
            try:
                result = run_pipeline(str(getattr(rec, "title", "")), str(getattr(rec, "text", "")))
                flat = result.to_row(article_id=i)
                flat["title"] = str(getattr(rec, "title", ""))[:200]
                rows.append(flat)
            except Exception as exc:
                rows.append({"article_id": i, "error": str(exc)})
            progress.progress(i / len(df))
        status.write("Done.")
        out_df = pd.DataFrame(rows)
        st.dataframe(out_df)
        buf = io.StringIO()
        out_df.to_csv(buf, index=False)
        st.download_button(
            "Download results.csv",
            data=buf.getvalue(),
            file_name="results.csv",
            mime="text/csv",
        )


def main() -> None:
    st.title("Multi-Agent Fake News Detector")
    st.caption("Claim Extractor -> Fact Checker (Wikipedia) -> Bias Detector -> Judge")
    sidebar_controls()
    tab1, tab2 = st.tabs(["Single article", "Batch"])
    with tab1:
        single_article_tab()
    with tab2:
        batch_tab()


if __name__ == "__main__":
    main()

# Multi-Agent Fake News Detection System

A CrewAI-powered multi-agent pipeline that classifies news articles as **REAL** or **FAKE** using a sequential chain of four specialized LLM agents, backed by a local Ollama model (default `qwen2.5:3b`).

## Pipeline

```
Article -> Claim Extractor -> Fact Checker (Wikipedia) -> Bias Detector -> Judge -> Verdict
```

1. **Claim Extractor** - pulls 3 verifiable factual claims out of the article.
2. **Fact Checker** - queries Wikipedia for each claim and assigns `SUPPORTED` / `CONTRADICTED` / `UNVERIFIABLE`.
3. **Bias Detector** - analyzes tone, loaded language and framing.
4. **Judge** - aggregates everything into a final `REAL` / `FAKE` verdict with a confidence score.

## Setup

### 1. Python environment

Requires **Python 3.10–3.13**. Python 3.14+ is not supported by `crewai`.

```bash
python3.13 -m venv .venv      # or python3.12
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ollama

Make sure Ollama is running and the model is pulled:

```bash
ollama pull qwen2.5:3b
ollama serve     # leave running in a separate terminal
```

### 3. Environment

```bash
cp .env.example .env
# edit values if needed
```

## Usage

### Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

- **Single Article** tab: paste a title + body, get a verdict with each agent's full output.
- **Batch** tab: upload a CSV (`title,text` columns) and download annotated results.

### Notebooks

- `notebooks/01_pipeline.ipynb` - end-to-end demo of the CrewAI pipeline.
- `notebooks/02_evaluation.ipynb` - metrics, confusion matrix, and error analysis on the Kaggle sample.

### Batch evaluation

```bash
python scripts/download_dataset.py          # fetches Kaggle Fake/Real news
python scripts/run_batch.py --n 100 --out results/results.csv
```

The batch runner is resumable: rerun the command and it will skip article IDs that already appear in the output CSV.

## Backend switching

Edit `.env`:

```
LLM_BACKEND=ollama        # or "openai"
OLLAMA_MODEL=qwen2.5:3b
```

For OpenAI, set `LLM_BACKEND=openai` and `OPENAI_API_KEY=sk-...`.

## Project structure

```
src/        pipeline code (agents, tasks, tools, schemas, crew)
app/        Streamlit UI
notebooks/  implementation + evaluation notebooks
scripts/    dataset download + batch runner
data/       dataset files (gitignored)
results/    CSV outputs (gitignored)
```

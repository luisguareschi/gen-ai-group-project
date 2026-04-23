# Multi-Agent Fake News Detection System

A CrewAI-powered multi-agent pipeline that classifies news articles as **REAL** or **FAKE** using a sequential chain of four specialized LLM agents, backed by a local Ollama model or OpenAI.

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

Check which models you already have, then pull one if needed:

```bash
ollama ls                  # list locally available models
ollama pull qwen3:8b       # pull a model if not already present
ollama serve               # leave running in a separate terminal
```

Supported models: `qwen2.5:3b`, `qwen2.5:7b`, `qwen2.5:14b`, `qwen3:8b`, `gemma3:4b`, `gemma4:e2b`.

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
- **Sidebar**: switch backend (Ollama/OpenAI), select model, and adjust temperature at runtime.

### Notebooks

- `notebooks/01_pipeline.ipynb` - end-to-end demo of the CrewAI pipeline.
- `notebooks/02_evaluation.ipynb` - metrics, confusion matrix, and error analysis on the Kaggle sample.

### Batch evaluation

```bash
python scripts/download_dataset.py          # fetches Kaggle Fake/Real news
python scripts/run_batch.py --n 100 --out results/results.csv
```

The batch runner is resumable: rerun the command and it will skip article IDs that already appear in the output CSV.

## Changing models

Where you make the change depends on how you're running the pipeline:

| Context | Where to change |
|---------|----------------|
| Streamlit UI | Sidebar → **Ollama model** dropdown (takes effect immediately) |
| Scripts / notebooks | Edit `OLLAMA_MODEL` in `.env` |
| Default for new installs | Edit `OLLAMA_MODEL` in `.env.example` |

First check what you have locally, then pull anything new:

```bash
ollama ls
ollama pull qwen3:8b
```

Available Ollama models: `qwen2.5:3b`, `qwen2.5:7b`, `qwen2.5:14b`, `qwen3:8b`, `gemma3:4b`, `gemma4:e2b`.

### Switching to OpenAI

Set the following in `.env` (or via the Streamlit sidebar):

```
LLM_BACKEND=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini   # or gpt-4o, etc.
```

## Project structure

```
src/        pipeline code (agents, tasks, tools, schemas, crew)
app/        Streamlit UI
notebooks/  implementation + evaluation notebooks
scripts/    dataset download + batch runner
data/       dataset files (gitignored)
results/    CSV outputs (gitignored)
```

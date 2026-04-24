# Key Insights from Building the Fake News Detection Pipeline

## 1. WELFake Dataset Leakage

The WELFake dataset covers US political news from **2015–2017**. This period is well within the training data of every major LLM, including Qwen, GPT-4, and others. When the model is asked to fact-check claims from these articles, it is not actually retrieving and reasoning about new information — it is pattern-matching against what it already memorised during pre-training.

This has two consequences:
- **Inflated accuracy**: The model appears highly accurate (e.g. 88% with Qwen2.5:3b) not because the pipeline is working well, but because the model already "knows" which articles are fake from training exposure.
- **Wikipedia/DuckDuckGo adds noise**: When the model already has the answer baked in, search results from Wikipedia or DDG introduce conflicting or redundant context that confuses it. The model starts trying to reconcile what it knows with what it retrieved, often producing worse verdicts than if it had used its internal knowledge alone.

**Takeaway**: High accuracy on WELFake is not a reliable signal of pipeline quality. The dataset is effectively contaminated for any LLM released after ~2020.

---

## 2. No Good Post-2024 Datasets

There is no widely available, clean, labelled fake news dataset covering events from mid-2024 onwards that is equivalent to WELFake in size and quality. The algozee/fake-news Kaggle dataset, for example, turned out to contain synthetic template-generated articles — not real news — making it useless for evaluation.

This matters because a truly robust evaluation would require articles about events the model has never seen. Without such a dataset, it is very hard to distinguish genuine pipeline performance from training data memorisation.

**Takeaway**: Evaluating LLM-based fact-checking on pre-cutoff data is fundamentally flawed. A credible evaluation needs a held-out post-cutoff test set, which is currently very difficult to construct at scale.

---

## 3. Smaller Models Do Less Overthinking

Counterintuitively, Qwen2.5:3b often outperformed Qwen2.5:7b on WELFake accuracy. The likely explanation is not that 3b is "smarter" — it is that 3b reasons less. It pattern-matches quickly to the most salient features of the article (tone, named entities, headline style) and outputs a verdict. The 7b model thinks more, considers counterarguments, hedges more, and sometimes talks itself out of the correct answer.

On a dataset where the model has already seen the data during training, shallow pattern-matching wins. On genuinely unseen data, this advantage would disappear.

**Takeaway**: Benchmark results on contaminated datasets favour fast, shallow models. This should not be taken as evidence that smaller models are better at fact-checking in general.

---

## 4. Local Models Are Unreliable at Tool Calling

Qwen2.5:3b essentially never triggers external tools (Wikipedia, DuckDuckGo) when assigned as a CrewAI agent. Instead of generating a structured function call, it outputs the tool parameters as plain text — e.g. `Wikipedia: {"query": "Putin", "max_results": null}` — which the framework cannot parse and execute.

Qwen2.5:14b behaves slightly better (the output format changes with the correct API endpoint), but tool execution is still unreliable. This is a fundamental limitation of local models: tool/function calling is a learned behaviour that requires substantial model capacity and specific fine-tuning. OpenAI's models were purpose-built for it; Ollama local models treat it as an afterthought.

There are two tool-calling mechanisms in LLM frameworks:
- **ReAct (text-based)**: The model outputs `Action: tool_name` / `Action Input: {...}` as text; the framework parses it and calls the Python function. No special model training needed, but format compliance is fragile.
- **Native function calling (API-level)**: Tool schemas are passed as structured JSON; the model returns a structured function call object that the API intercepts and executes. Reliable with OpenAI, poorly supported in Ollama.

**Takeaway**: For a pipeline that depends on real-time web searches, use OpenAI or call the tools directly in Python rather than relying on local model tool invocation.

---

## 5. Deterministic Scoring Extracted to Python

Early versions of the pipeline let the Judge LLM agent compute the final label and confidence score. This was problematic: the model would sometimes change the formula, hallucinate a different confidence, or produce different results on identical inputs.

The solution was to extract all arithmetic into a deterministic Python function (`_compute_verdict`) that runs between the two crew phases:

```
combined = fact_signal * 0.60 + bias_signal * 0.35 + roberta_signal * roberta_weight * 0.05
```

The Judge agent then receives the pre-computed label and confidence as fixed values and is only asked to write a human-readable summary explaining them. This guarantees reproducibility and prevents the model from second-guessing the verdict.

**Takeaway**: Use LLMs for what they are good at (reading, reasoning, writing). Use Python for arithmetic that must be deterministic and reproducible. Never let an LLM "decide" a numerical score if you can compute it programmatically.

---

## 6. `output_pydantic` Can Cause Agents to Skip Tools

CrewAI's `output_pydantic` parameter on a Task tells the framework to validate and coerce the agent's output into a Pydantic model. The unintended side effect is that it makes the agent eager to produce a structured JSON response immediately — it sees the schema it needs to satisfy and jumps straight to outputting it, skipping any tool calls it was supposed to make first.

This was the root cause of the fact-checker not searching Wikipedia or DuckDuckGo: the task had `output_pydantic=FactCheckOutput`, so the agent produced a JSON verdict immediately without searching anything.

The fix was to split the fact-checking into two tasks:
- **t2a**: Search only — no `output_pydantic`, plain text output. Forces the agent to call tools.
- **t2b**: Verdict only — has `output_pydantic=FactCheckOutput`, uses `context=[t2a]`. Reads t2a's results and assigns verdicts without calling any tools.

**Takeaway**: If an agent is supposed to use tools before producing structured output, do not put `output_pydantic` on that task. Split search and synthesis into separate tasks.

---

## 7. Bias Signal Is the Most Reliable Feature for Local Models

Because tool calling is unreliable with local models, the fact-checker often produces no useful signal (all verdicts are UNVERIFIABLE). In practice, the bias detector ends up carrying most of the weight in the final score. The bias task does not require any tools — it is pure reading comprehension and linguistic analysis, which local models handle well.

This is a reasonable proxy: genuine fake news articles tend to use more inflammatory language, unattributed claims, and one-sided framing than wire-service reporting. But it is not a substitute for real fact-checking, and it will fail on well-written disinformation that mimics neutral style.

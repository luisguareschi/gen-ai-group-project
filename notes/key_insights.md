# Key Insights from Building the Fake News Detection Pipeline

## 1. WELFake Dataset Leakage

The WELFake dataset covers US political news from **2015–2017**. This period is well within the training data of every major LLM, including Qwen, GPT-4, and others. When the model is asked to fact-check claims from these articles, it is not actually retrieving and reasoning about new information. It is pattern-matching against what it already memorised during pre-training.

This has two consequences:
- **Inflated accuracy**: The model appears highly accurate (e.g. 88% with Qwen2.5:3b) not because the pipeline is working well, but because the model already "knows" which articles are fake from training exposure.
- **Wikipedia/DuckDuckGo adds noise**: When the model already has the answer baked in, search results from Wikipedia or DDG introduce conflicting or redundant context that confuses it. The model starts trying to reconcile what it knows with what it retrieved, often producing worse verdicts than if it had used its internal knowledge alone.

**Takeaway**: High accuracy on WELFake is not a reliable signal of pipeline quality. The dataset is effectively contaminated for any LLM released after ~2020.

---

## 2. No Good Post-2024 Datasets

There is no widely available, clean, labelled fake news dataset covering events from mid-2024 onwards that is equivalent to WELFake in size and quality. The algozee/fake-news Kaggle dataset, for example, turned out to contain synthetic template-generated articles rather than real news, making it useless for evaluation.

This matters because a truly robust evaluation would require articles about events the model has never seen. Without such a dataset, it is very hard to distinguish genuine pipeline performance from training data memorisation.

**Takeaway**: Evaluating LLM-based fact-checking on pre-cutoff data is fundamentally flawed. A credible evaluation needs a held-out post-cutoff test set, which is currently very difficult to construct at scale.

---

## 3. Smaller Models Do Less Overthinking

Counterintuitively, Qwen2.5:3b often outperformed Qwen2.5:7b on WELFake accuracy. The likely explanation is not that 3b is "smarter". It is that 3b reasons less. It pattern-matches quickly to the most salient features of the article (tone, named entities, headline style) and outputs a verdict. The 7b model thinks more, considers counterarguments, hedges more, and sometimes talks itself out of the correct answer.

On a dataset where the model has already seen the data during training, shallow pattern-matching wins. On genuinely unseen data, this advantage would disappear.

**Takeaway**: Benchmark results on contaminated datasets favour fast, shallow models. This should not be taken as evidence that smaller models are better at fact-checking in general.

---

## 4. Local Models Are Unreliable at Tool Calling

Qwen2.5:3b essentially never triggers external tools (Wikipedia, DuckDuckGo) when assigned as a CrewAI agent. Instead of generating a structured function call, it outputs the tool parameters as plain text, for example `Wikipedia: {"query": "Putin", "max_results": null}`, which the framework cannot parse and execute.

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

## 6. `output_pydantic` Can Cause Agents to Skip Tools, and Splitting Doesn't Fully Cure Hallucinated Evidence

CrewAI's `output_pydantic` parameter on a Task tells the framework to validate and coerce the agent's output into a Pydantic model. The unintended side effect is that it makes the agent eager to produce a structured JSON response immediately. It sees the schema it needs to satisfy and jumps straight to outputting it, skipping any tool calls it was supposed to make first.

This was the root cause of the fact-checker not searching Wikipedia or DuckDuckGo: the task had `output_pydantic=FactCheckOutput`, so the agent produced a JSON verdict immediately without searching anything.

The fix was to split the fact-checking into two tasks:
- **t2a**: Search only, no `output_pydantic`, plain text output. Forces the agent to call tools.
- **t2b**: Verdict only, has `output_pydantic=FactCheckOutput`, uses `context=[t2a]`. Reads t2a's results and assigns verdicts without calling any tools.

However, the split fixed the structural problem without solving the deeper epistemic one. Even with tools nominally invoked, local models still fabricate plausible-looking sources when the tools return little. In the 100-article evaluation, article 43702 ("Al Qaeda warns Myanmar") cites `https://example.com/al_qaeda_support_rohingya_muslims_fleeing_to_bangladesh` as supporting evidence, a URL the model invented. Article 6815 cites a "DuckDuckGo snippet" that simply quotes the article back to itself almost verbatim.

**Takeaway**: If an agent is supposed to use tools before producing structured output, do not put `output_pydantic` on that task; split search and synthesis into separate tasks. But also recognise that structural fixes (task splits, schemas) cannot prevent the model from inventing things when it has no real evidence. Real mitigation requires provenance checks that compare cited evidence against actual tool returns, or a model with stronger tool-use fidelity.

---

## 7. CrewAI Is a Good Abstraction, Until You Need to Do Anything in Python

CrewAI's value proposition is declarative: you describe agents (role, goal, backstory), describe tasks (description, expected output, optional schema), wire them into a crew, and let the framework handle orchestration, context passing, and tool dispatch. When everything stays inside that pattern, the code reads cleanly and the framework earns its keep.

The trouble starts the moment you need behaviour the abstraction doesn't naturally express. Two examples in this project, both already discussed above:

- **Deterministic scoring (#5)**: arithmetic that must be reproducible cannot live inside an LLM agent. We pulled it out into `_compute_verdict` and split the run into two crews ("phase 1" agents, then Python, then "phase 2" Judge). The pipeline is no longer a single declarative crew; it is a Python script that happens to call CrewAI twice.
- **Forcing tool use (#6)**: getting the fact-checker to actually search required splitting one task into two and manually wiring `context=[t2a]` between them. That works, but at that point you are doing manual dataflow plumbing that the framework was supposed to hide.

Each individual workaround is fine. The cumulative effect is that the code base ends up in an awkward middle state: some control flow is declarative CrewAI, some is imperative Python, and a reader has to hold both mental models at once to follow what happens. A pure Python implementation with direct LLM calls would have been more verbose but more uniform; a pure CrewAI implementation would have been cleaner but couldn't have hit the determinism and tool-use requirements. The hybrid is the worst of both for readability, even though it was the right pragmatic choice given the constraints.

### When to break out of the framework

CrewAI can technically force tool calls via `tool_choice` in LiteLLM, so running `_compute_verdict` as a Python tool exposed to an agent is possible. We chose not to for a simpler reason: if removing the LLM from a step wouldn't change the output, remove it. `_compute_verdict` is just arithmetic on structured data, and routing it through an agent adds failure points with zero benefit.

The rule we settled on: keep things in the framework when the model's judgement is the value. Extract to Python when it is just computation.

The two-crew split is itself a symptom of CrewAI not having a native concept of a deterministic step between tasks. Building our own minimal orchestration, a simple `while` loop that calls the LLM, dispatches tool calls, and runs Python in between, would handle this more naturally at the cost of losing CrewAI's boilerplate savings. Frameworks like LangGraph also solve this more cleanly, with explicit nodes for deterministic steps, but the tradeoff is added framework complexity to learn and reason about.

**Takeaway**: Choose CrewAI when the problem genuinely fits the declarative agent pattern, and accept that as soon as you start reaching outside it you pay a comprehension tax. For projects that need deterministic post-processing, manual orchestration between phases, or strong control over when tools fire, plain Python with an LLM SDK or a graph-based framework like LangGraph is often the better starting point. The brief's offer of "plain Python with direct API calls if you prefer more control" is not just a fallback; it is the right default for any pipeline whose hardest requirements lie outside agent coordination.

---

## 8. Bias Signal Carries Most of the Weight, But the Rubric Is Effectively Bimodal

Because tool calling is unreliable with local models, the fact-checker often produces no useful signal (all verdicts are UNVERIFIABLE). In practice, the bias detector ends up carrying a large share of the final score. The bias task does not require any tools. It is pure reading comprehension and linguistic analysis, which local models handle well. Quantitatively, the deterministic scorer in `_compute_verdict` weights the signals at 60% fact-check, 35% bias, and 5% RoBERTa, so when fact-check collapses to UNVERIFIABLE the bias term dominates the remaining variance.

There is a second issue with how the bias signal behaves. The bias prompt defines four bands (0.0–0.2, 0.3–0.5, 0.6–0.7, 0.8–1.0), but in the 100-article evaluation almost every score lands at either 0.15 or 0.85. The model treats it as a binary "neutral wire-style vs. blog-style" detector and picks the band centre. This works on Kaggle Fake/Real because Reuters articles look obviously different from blog-style fakes, but it would collapse on disinformation written in a neutral style.

**Takeaway**: The bias signal is doing useful work given how unreliable the fact-checker is, but it is exploiting a stylistic artifact of the dataset rather than measuring bias in a generalisable way. It is also not a substitute for real fact-checking, and it will fail on well-written disinformation that mimics neutral style.

---

## 9. RoBERTa Is the Only Signal Trained for This Task, Yet Weighted at 5%

The HuggingFace `hamzab/roberta-fake-news-classification` model is the only component in the pipeline that has actually been fine-tuned on a fake/real classification objective. It achieves roughly 90% accuracy on this dataset on its own. The LLM-derived fact and bias signals, by contrast, are noisier proxies built on general-purpose reasoning. Despite this, the deterministic scorer assigns RoBERTa only 5% weight versus 60% for fact-check and 35% for bias.

This was a deliberate choice to keep the pipeline genuinely "agentic" rather than letting one classifier dominate the verdict and reduce the other agents to ornamentation. It is worth flagging as a design tension rather than an obvious win.

**Takeaway**: In a real deployment you would either raise the RoBERTa weight substantially, or use it as a tie-breaker only when the agent signals disagree. For an academic project that is meant to demonstrate multi-agent coordination, keeping it as a minor signal is defensible, but it should be acknowledged explicitly rather than buried in the scoring formula.

---

## 10. The Pipeline Is Overconfident on Its Errors

The deterministic scorer maps any combined signal of magnitude ≥ 0.9 to a confidence above 0.9, regardless of whether the underlying evidence was hallucinated. The 100-article evaluation surfaces several high-confidence wrong predictions: article 43702 ("Al Qaeda warns Myanmar") was predicted FAKE with 0.95 confidence and was actually REAL; article 34392 (Pennsylvania budget) similarly predicted FAKE at 0.93. In both cases the bias detector scored the article as sensationalist on stylistic features alone, while the fact-checker invented "supporting" or "contradicting" evidence with high self-reported confidence.

The confidence number reflects how much the three signals agree with each other, not how often the system is right when it is that confident.

**Takeaway**: Treat the reported confidence as a measure of internal signal agreement, not as a calibrated probability of correctness. For any user-facing use, the confidence should be recalibrated against held-out accuracy (e.g. via isotonic regression) or simply replaced with a coarser bucket like "high / medium / low".

---

## 11. Sequential Architecture (Option A) Was the Right Call Here

The brief offered a choice between a sequential pipeline and a parallel one in which the fact-checker and bias detector run concurrently. We chose sequential. The reasoning, in hindsight, holds up:

- The bias detector is fast (no tools, pure text analysis), so parallelising it with the fact-checker yields only modest wall-clock savings.
- The slow component is the fact-checker's search loop, which is not parallelisable with itself.
- Most of the issues we surfaced (insight #6, hallucinated URLs in insight #6, output_pydantic interactions) were trace-driven and would have been substantially harder to debug in interleaved logs from concurrent agents.

**Takeaway**: Parallelism only pays when the parallel branches are individually slow and the debugging cost stays manageable. For this pipeline neither condition held, so the simpler architecture won on both speed-to-build and debuggability.

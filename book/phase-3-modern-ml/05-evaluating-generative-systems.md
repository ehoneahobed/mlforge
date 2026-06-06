# Module 3.5 — Evaluating Generative Systems

**Mode:** curated breadth
**Gate:** ship gate
**Est. effort:** 8–12 focused hours

> You can now build and adapt generative systems. This module confronts the question that
> quietly decides whether any of them are worth anything: how do you know it is good? For a
> classifier you had accuracy. For an open-ended generator, evaluation is genuinely hard, and
> getting it right is one of the most valuable and least taught skills in modern ML.

## Why this matters

Evaluation is the bottleneck of modern AI development. When a model can produce any text or any
image, there is no single number that captures "good." Teams routinely ship systems on the
strength of a few impressive demos, then discover in production that the system fails in ways no
one measured. The practitioners who are trusted with real systems are the ones who can build an
evaluation they, and a skeptical reviewer, can believe.

This module ties together the whole phase: the fine-tuned model from 3.2, the RAG system from
3.3, and generative output in general all need rigorous evaluation. It also extends the
evaluation discipline you built in Phase 1 (Module 1.6) into the much harder generative setting,
where the ground truth is often a matter of judgment rather than a label.

## What you will be able to do

By the end of this module you will be able to:

- Explain why evaluating generative systems is fundamentally harder than classification.
- Use and critique standard benchmarks and automatic metrics, knowing their failure modes.
- Apply LLM-as-a-judge evaluation correctly, including its biases.
- Design and build a task-specific evaluation suite for a system you built.

## Prerequisites

- [Module 1.6](../phase-1-classical-ml/06-evaluation-calibration-leakage.md): the evaluation
  mindset (trustworthy measurement, leakage), which this module extends.
- At least one generative system to evaluate, ideally your Module 3.2 fine-tune or Module 3.3
  RAG system.

## Curated path

1. **The foundational idea: "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
   (Zheng et al., 2023).** The paper that established using a strong LLM as an evaluator and
   crowd-sourced human preference (Chatbot Arena). Read it for both the method and its honest
   discussion of limitations.
   https://arxiv.org/abs/2306.05685

2. **A practical guide to LLM-as-a-judge.** Read a current, hands-on guide to building judge
   prompts, choosing pairwise versus pointwise scoring, and avoiding the known biases. Evidently
   AI's guide is thorough and current.
   https://www.evidentlyai.com/llm-guide/llm-as-a-judge

3. **Standardized benchmarking: the EleutherAI LM Evaluation Harness.** The de facto open tool
   for running LLMs against standard benchmarks. Read its overview to understand what these
   benchmarks measure and how to run them.
   https://github.com/EleutherAI/lm-evaluation-harness

4. **RAG and task-specific evaluation: RAGAS** (revisited from Module 3.3) for the retrieval
   case, plus the Hugging Face `evaluate` library for standard metrics.
   https://docs.ragas.io/ and https://huggingface.co/docs/evaluate/

**Deliberately skip for now:** building your own crowd-sourced arena, and deep statistical
treatment of inter-rater agreement. Focus on a credible task-specific suite that combines
automatic metrics, an LLM judge, and a small human-checked set.

## Background: the ideas in words

**Why it is hard.** For a classifier, a prediction is right or wrong. For a generator, there are
many acceptable outputs and many subtly bad ones, and "quality" depends on the task (factual
accuracy, helpfulness, style, safety). A single metric cannot capture this, so evaluation
becomes a designed system rather than a number.

**Automatic metrics and benchmarks.** Metrics like perplexity (how well a model predicts text)
and overlap scores (BLEU, ROUGE) are cheap but weak proxies for quality; they miss meaning.
Standardized **benchmarks** (run with tools like the LM Evaluation Harness) let you compare
models on fixed tasks, but they have a serious flaw: **contamination**, where benchmark data has
leaked into training, inflating scores. Treat leaderboard numbers with the same suspicion you
learned in Phase 1.

**LLM-as-a-judge.** A strong LLM can score or compare outputs, agreeing with human judgment
surprisingly often, and it scales far better than human raters. But it has biases you must
control: **position bias** (favoring the first option), **verbosity bias** (favoring longer
answers), and **self-preference** (favoring its own style). Mitigations include randomizing
order, using pairwise comparisons, and calibrating against a human-labeled sample.

**Human evaluation and arenas.** Ultimately, human preference is the gold standard.
**Chatbot Arena** popularized large-scale pairwise human voting to rank models. For your own
systems, a small, carefully labeled human set is invaluable for checking that your automatic and
LLM-judge metrics actually track what you care about.

**The synthesis.** A credible evaluation combines layers: cheap automatic metrics for fast
iteration, an LLM judge for scale, and a small human-checked set to validate the whole stack.
And, as always, you guard against leakage and contamination.

## Knowledge check

1. Why can you not evaluate a text generator the way you evaluate a classifier?
2. What do perplexity and overlap metrics (BLEU/ROUGE) measure, and why are they weak proxies
   for quality?
3. What is benchmark contamination, and why does it make leaderboard numbers untrustworthy?
4. Name three biases of LLM-as-a-judge evaluation and a mitigation for each.
5. Why keep a small human-labeled set even when using automatic and LLM-judge metrics?
6. Sketch a layered evaluation for a system you built in this phase.

## Project (ship gate)

**Design and build an evaluation suite for a generative system you built.**

- Take your Module 3.2 fine-tune or Module 3.3 RAG system (or another generative system) and
  build a real evaluation for it.
- Include at least three layers: an **automatic metric** appropriate to the task, an
  **LLM-as-a-judge** component (with bias mitigations applied), and a **small human-labeled
  set** you score yourself, used to check whether the automatic and judge metrics agree with
  your judgment.
- Address **contamination and limits** explicitly: argue why your test data was not seen in
  training/tuning, and document what your evaluation does and does not capture.
- Produce a verdict: based on your evaluation, is the system good enough, and for what?

**Definition of done:** a runnable evaluation suite, results across all layers, a check of
agreement between automatic/judge metrics and your human labels, and a written, defensible
verdict with its limitations.

## The workshop: ship it

Build this in its own repository, `mlforge-llm-eval`, using the project habits from Module 0.2.

1. Set up the project:

```bash
mkdir mlforge-llm-eval && cd mlforge-llm-eval
uv init && uv add evaluate ragas datasets pandas matplotlib && mkdir src data
```

2. Build the evaluation suite in `src/` (automatic metrics, the LLM-judge with randomized order,
   and the loader for your human-labeled set in `data/`), targeting the system you built earlier.
3. Commit at checkpoints: "Automatic metrics", then "LLM-as-judge with bias controls", then
   "Human-set agreement check and verdict".
4. Add a README with your results, the agreement analysis, and your defensible verdict.
5. Ship it:

```bash
gh repo create mlforge-llm-eval --public --source=. --push
```

   (No `gh`? Create an empty public repo, then `git remote add origin <url>` and
   `git push -u origin main`.)

**Done when:** `mlforge-llm-eval` is on GitHub, evaluates a real system across multiple layers,
checks its metrics against human judgment, and states an honest verdict with limitations.

## Going deeper (optional)

- Read about position and length biases in LLM judges and the "panel of judges" approach.
- Explore HELM (Holistic Evaluation of Language Models) for a broad, multi-metric framework.
- Build a tiny "arena" that asks a friend to vote on pairs of outputs and compare to your LLM
  judge.

## Canonical references

- Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*.
- EleutherAI, *LM Evaluation Harness*; Hugging Face, *Evaluate*; RAGAS documentation.

# Module 3.1 — Inside Large Language Models

**Mode:** hybrid
**Gate:** concept check
**Est. effort:** 8–12 focused hours

> You built a small GPT from scratch in Module 2.8. A frontier language model is the same
> architecture, scaled by many orders of magnitude and trained on much of the internet. This
> module closes the gap between the two: what changes when you scale, how these models are
> actually trained, and why scale produces capabilities that small models simply do not have.

## Why this matters

Large language models are the defining technology of the current moment, and you are unusually
well prepared to understand them, because you have implemented their core mechanism yourself.
What you have not yet seen is the rest of the recipe: how raw text becomes tokens, what the
training objective actually is at scale, why bigger models trained on more data predictably get
better (scaling laws), and how a model that only predicts the next token ends up able to write
code, answer questions, and follow instructions.

Understanding this is what separates someone who uses an LLM API from someone who can reason
about what these models can and cannot do, why they hallucinate, why prompting works, and where
the field is heading. The rest of Phase 3 (fine-tuning, retrieval, evaluation) assumes this
mental model.

## What you will be able to do

By the end of this module you will be able to:

- Explain tokenization and why models operate on tokens rather than characters or words.
- Describe the pretraining objective and how the GPT you built scales to a frontier model.
- Explain scaling laws and what "emergence" means, with appropriate skepticism.
- Distinguish pretraining, fine-tuning, and prompting, and say when each is the right tool.

## Prerequisites

- [Module 2.8](../phase-2-deep-learning/08-transformer-from-scratch.md): you have built a
  transformer and a small GPT. This module is largely "your nano-GPT, at scale."

## Curated path

1. **Karpathy, "Intro to Large Language Models" (the one-hour talk).** A clear, high-level map
   of what an LLM is, how it is trained (pretraining then fine-tuning), and how to think about
   its capabilities and limits. The best single orientation.
   https://www.youtube.com/watch?v=zjkBMFhNj_g

2. **Karpathy, "Let's build the GPT Tokenizer."** Tokenization is the one major piece of the
   LLM pipeline you have not built. This video constructs a byte-pair-encoding tokenizer from
   scratch and explains why so many model quirks trace back to tokenization. Part of
   *Neural Networks: Zero to Hero*.
   https://github.com/karpathy/nn-zero-to-hero

3. **3Blue1Brown, the Transformers/LLM chapters.** "But what is a GPT?" and the attention
   chapters give the visual, intuitive picture of how these models process text. Rewatch now
   that you have built one.
   https://www.3blue1brown.com/topics/neural-networks

4. **The Hugging Face LLM Course, the opening chapters.** How transformer models are used in
   practice, the model/tokenizer/pipeline abstractions, and the landscape of model types. This
   is also your on-ramp to the tools you will use in Module 3.2.
   https://huggingface.co/learn/llm-course/

**Deliberately skip for now:** the training-infrastructure details (distributed training,
data pipelines at scale); those are Phase 4. Here, understand the concepts and the recipe.

## Background: the ideas in words

**Tokenization.** Models do not read characters or whole words; they read **tokens**, chunks
of text (often sub-words) produced by an algorithm like byte-pair encoding. The model has a
fixed vocabulary of tokens and an embedding for each. Many surprising model behaviors (trouble
spelling, odd arithmetic, sensitivity to spacing) come from tokenization, which is why it is
worth understanding directly.

**Pretraining.** The core training objective is the same one your nano-GPT used:
**next-token prediction**. Show the model enormous amounts of text and have it predict the next
token, over and over. That is it. The astonishing result is that to predict the next token well
across the whole internet, the model is forced to learn grammar, facts, reasoning patterns, and
more. This unsupervised (technically self-supervised) objective is what makes internet-scale
training possible: no human labeling required.

**Scaling laws.** Empirically, model performance improves smoothly and predictably as you
increase three things together: model size (parameters), data, and compute. These **scaling
laws** are why the field raced to build larger models; you can forecast the gain from scaling
before you spend the money. The Chinchilla result refined this, showing many early models were
too large for their training data.

**Emergence, carefully.** Some capabilities appear fairly suddenly as models scale past a
threshold, which people call **emergence**. Treat this with care: part of it is real, and part
is an artifact of how capabilities are measured. The honest summary is that bigger models can
do qualitatively new things, but the "sudden" framing is sometimes overstated.

**Pretraining vs fine-tuning vs prompting.** A pretrained "base" model predicts text. To make
it helpful (an assistant), it is **fine-tuned** on instruction-following and aligned to human
preferences (Module 3.2). At use time, **prompting** (including few-shot examples in the
context) steers the model without changing its weights, exploiting its in-context learning
ability. Knowing which lever to pull, prompt, retrieve, or fine-tune, is a core practical
skill that the next modules build.

## Knowledge check

1. What is a token, and why do models operate on tokens rather than characters or words?
2. State the pretraining objective. Why does next-token prediction force a model to learn so
   much?
3. What do scaling laws describe, and why were they so consequential for the field?
4. What does "emergence" mean in LLMs, and why should you be somewhat skeptical of the term?
5. Distinguish pretraining, fine-tuning, and prompting. Give a situation where each is the
   right choice.
6. Name two model behaviors that trace back to tokenization.

## Project

A hands-on exploration to make the concepts concrete (this is understanding-focused, so it
goes to your learning log):

- **Probe a tokenizer.** Using the Hugging Face `transformers` tokenizer for a small open
  model, tokenize a variety of text (normal sentences, numbers, code, rare words, other
  languages) and document surprising splits. Connect what you see to the tokenization
  discussion.
- **Probe in-context learning.** With a small open model (or an API), show the same task with
  zero, one, and a few examples in the prompt, and document how few-shot prompting changes the
  output.
- **Write a one-page explainer** in your own words: how a frontier LLM is built, from
  tokenization through pretraining to an instruction-following assistant, and why scale matters.

**Definition of done:** the tokenizer and in-context-learning explorations with your notes, and
the one-page explainer.

## The workshop: log it

This module builds understanding, so it goes in your `modelwright-learning-log` repository (from
Module 0.1).

1. Add `phase-3/01-llm-internals/` with a notebook (`probes.ipynb`) for the tokenizer and
   few-shot explorations, and your one-page explainer (`explainer.md`).
2. Commit and push:

```bash
git add -A && git commit -m "Module 3.1: inside large language models" && git push
```

**Done when:** your probes and explainer are in `modelwright-learning-log`.

## Going deeper (optional)

- Read the scaling-laws papers: Kaplan et al. (2020) and the Chinchilla paper (Hoffmann et al.,
  2022).
- Karpathy's longer "Deep Dive into LLMs like ChatGPT" covers the full modern pipeline,
  including the post-training stages you will study in Module 3.2.
- Jay Alammar, "The Illustrated GPT-2," for a visual tour of a real decoder-only model.
  https://jalammar.github.io/illustrated-gpt2/

## Canonical references

- Karpathy, A. *Intro to Large Language Models* and *Let's build the GPT Tokenizer*.
- Hugging Face, *LLM Course*.
- Kaplan et al. (2020), *Scaling Laws for Neural Language Models*; Hoffmann et al. (2022),
  *Training Compute-Optimal Large Language Models* (Chinchilla).

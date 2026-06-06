# Module 3.2 — Fine-tuning & Adaptation

**Mode:** hybrid
**Gate:** build gate
**Est. effort:** 12–16 focused hours

> A pretrained model knows a great deal in general but nothing about your specific task. This
> module is about closing that gap: taking an open model and adapting it, efficiently enough to
> do on a single free GPU, so it does what you need. You will fine-tune a real model and measure
> exactly how much better it gets.

## Why this matters

Fine-tuning is how general models become useful for specific jobs, and it is one of the most
in-demand practical skills in the field. The naive approach, retraining all of a model's
billions of parameters, is out of reach without a cluster. The breakthrough that democratized
fine-tuning is **parameter-efficient fine-tuning (PEFT)**, especially **LoRA**, which trains a
tiny number of extra parameters while freezing the original model. With LoRA you can adapt a
capable open model on a single consumer or free-tier GPU.

This module also demystifies how raw pretrained models become helpful assistants:
**instruction tuning** and **alignment** (RLHF and its simpler successor DPO). You do not need
to run full alignment yourself, but understanding it explains why modern chat models behave as
they do, and it is core literacy for anyone working with LLMs.

## What you will be able to do

By the end of this module you will be able to:

- Explain the difference between full fine-tuning and parameter-efficient methods, and why LoRA
  works.
- Fine-tune an open model on a task using Hugging Face `transformers`, `peft`, and `trl`.
- Measure the improvement over the base model honestly.
- Explain instruction tuning and the alignment family (RLHF, DPO) at a conceptual level.

## Prerequisites

- [Module 3.1](./01-inside-large-language-models.md): you understand pretraining and the
  pretraining-then-fine-tuning recipe.
- [Module 2.5](../phase-2-deep-learning/05-making-training-work.md): the training craft, which
  applies directly.
- Access to a GPU. A free Google Colab or Kaggle notebook is enough for the small models used
  here.

## The tools: what you will use and how they fit together

Three Hugging Face libraries work together for fine-tuning, and it helps to see the map before
the details:

- **`transformers`** gives you the model and its tokenizer: `AutoModelForCausalLM` and
  `AutoTokenizer` load a pretrained open model by name.
- **`peft`** adds parameter-efficient fine-tuning. You wrap the model in a `LoraConfig` so only
  small low-rank adapter weights are trained; the base model stays frozen.
- **`trl`** provides high-level trainers for the post-training stages: `SFTTrainer` for
  supervised (instruction) fine-tuning, and `DPOTrainer` for preference alignment. It plugs
  directly into `peft`, so you train LoRA adapters with a few lines of config.

The workflow: load a model and tokenizer with `transformers`, define a `LoraConfig` with
`peft`, hand both to `trl`'s `SFTTrainer` along with your dataset, and train. The result is a
small adapter you can save, share, and load on top of the base model.

## Curated path

1. **The Hugging Face LLM Course, the fine-tuning chapters, including LoRA.** The practical
   spine: how to fine-tune with `transformers` and `trl`, and the LoRA chapter specifically.
   https://huggingface.co/learn/llm-course/ (LoRA: https://huggingface.co/learn/llm-course/chapter11/4)

2. **The `peft` documentation, the LoRA conceptual guide and quickstart.** What LoRA does and
   how to configure it (`r`, `alpha`, target modules, dropout).
   https://huggingface.co/docs/peft and the LoRA guide:
   https://huggingface.co/docs/peft/main/en/conceptual_guides/lora

3. **The `trl` documentation, `SFTTrainer`.** The high-level path to supervised fine-tuning,
   with its native PEFT integration.
   https://huggingface.co/docs/trl

4. **For alignment, conceptually.** Read the Hugging Face material on RLHF and DPO to understand
   how preference alignment works, even though you will not run a full pipeline here. The
   `DPOTrainer` docs (in `trl`) show the practical, simpler-than-RLHF approach.

**Deliberately skip for now:** full RLHF with a trained reward model and PPO (heavy and
specialized), and quantization details beyond using QLoRA if you need it for memory. Focus on a
clean LoRA SFT run you can measure.

## Background: the ideas in words

**Full fine-tuning vs PEFT.** Full fine-tuning updates every weight; for a large model that
needs enormous memory (you must store gradients and optimizer state for billions of
parameters). **LoRA** instead freezes the original weights and inserts small trainable
low-rank matrices into the linear layers. The intuition: the *change* a task requires is
low-rank (much simpler than the full weight matrix), so a small adapter captures it. You train
a tiny fraction of the parameters, store a tiny adapter, and can swap adapters per task.
**QLoRA** goes further by quantizing the frozen base model to 4-bit, shrinking memory enough to
fine-tune sizable models on one GPU.

**Instruction tuning (SFT).** A base model just continues text. **Supervised fine-tuning** on
examples of instructions paired with good responses teaches it to follow instructions and
behave like an assistant. This is the first post-training stage.

**Alignment (RLHF and DPO).** To make a model helpful and harmless beyond what SFT gives,
it is tuned on human **preferences**: given two responses, which did people prefer?
**RLHF** trains a reward model on these preferences, then optimizes the model against it with
reinforcement learning, which is powerful but fiddly. **DPO (Direct Preference Optimization)**
achieves a similar effect by optimizing directly on the preference pairs with a simple loss,
no separate reward model or RL loop, which is why it has become a popular, accessible default.

## Knowledge check

1. Why is full fine-tuning of a large model impractical on a single GPU, and what specifically
   consumes the memory?
2. Explain LoRA. Why is it reasonable to assume the task-specific update is low-rank?
3. What does QLoRA add to LoRA, and what problem does it solve?
4. What is supervised (instruction) fine-tuning, and what does it change about a base model's
   behavior?
5. Contrast RLHF and DPO. Why has DPO become a popular alternative?
6. You have a small, task-specific dataset and one free GPU. Outline the fine-tuning approach
   you would use and why.

## Build gate

Fine-tune an open model with LoRA and measure the lift.

**Specification:**

- Choose a small open model (something that fits a free GPU) and a task with a dataset (a
  classification or instruction dataset, or a domain style you want to adapt to).
- Establish a **baseline**: evaluate the base model on a held-out set before any fine-tuning.
- Fine-tune with **LoRA** via `peft` and `trl`'s `SFTTrainer`. Track the run.
- Evaluate the fine-tuned model on the same held-out set and **measure the improvement**.

**What it must show:**

- A clear before-and-after comparison on a metric appropriate to the task, with the base model
  as the honest baseline.
- A working, saved LoRA adapter and the code to load and use it.
- A short analysis: did it help, by how much, and where does the fine-tuned model still fail?

## Project

Package it and write it up (reuse your template, plus experiment tracking from Module 0.2):

- The fine-tuning code, the saved adapter, and the evaluation.
- A report: the task and data, your LoRA configuration and why, the before-and-after numbers,
  and an honest discussion of remaining failures.

**Definition of done:** a measured, reproducible LoRA fine-tune that beats the base model on a
defined metric, with the write-up.

## The workshop: ship it

Build this in its own repository, `modelwright-llm-finetune`, using the project habits from
Module 0.2.

1. Set up the project:

```bash
mkdir modelwright-llm-finetune && cd modelwright-llm-finetune
uv init && uv add transformers peft trl datasets accelerate wandb && mkdir src
```

2. Write the baseline evaluation, the LoRA fine-tuning script, and the post-tuning evaluation
   in `src/`. Run the training on a free GPU notebook (Colab/Kaggle) with the code committed
   here; commit the saved adapter (it is small) or document where to get it.
3. Commit at checkpoints: "Baseline evaluation of base model", then "LoRA fine-tune", then
   "Post-tuning evaluation and comparison".
4. Add a README with your before-and-after results and configuration.
5. Ship it:

```bash
gh repo create modelwright-llm-finetune --public --source=. --push
```

   (No `gh`? Create an empty public repo, then `git remote add origin <url>` and
   `git push -u origin main`.)

**Done when:** `modelwright-llm-finetune` is on GitHub, the fine-tuned model measurably beats the
baseline, and the README documents the lift and the configuration.

## Going deeper (optional)

- Read the LoRA paper (Hu et al., 2021) and the QLoRA paper (Dettmers et al., 2023).
- Read the DPO paper (Rafailov et al., 2023) and try a small DPO run with `trl`.
- Explore quantization (`bitsandbytes`) and how it interacts with LoRA.

## Canonical references

- Hugging Face, *LLM Course* (fine-tuning and LoRA), *PEFT* and *TRL* documentation.
- Hu et al. (2021), *LoRA*; Dettmers et al. (2023), *QLoRA*; Rafailov et al. (2023), *DPO*.

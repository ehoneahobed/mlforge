# Module 4.2 — Training at Scale

**Mode:** hybrid
**Gate:** concept check
**Est. effort:** 8–12 focused hours

> You have trained models on a single GPU. Frontier models train on thousands of GPUs for weeks,
> and even routine industry training spans several. This module is about what changes when one
> GPU is not enough: how training is spread across hardware, the tricks that make it fit and run
> fast, and the economics that decide what is worth doing. You will practice the single-GPU
> techniques directly and understand the multi-GPU ones clearly.

## Why this matters

Scale is a defining feature of modern ML, and the techniques that make large training possible
are now standard knowledge even for engineers who rarely touch a cluster. Understanding them
tells you why training runs cost what they cost, why some models are feasible and others are
not, and how to make your own training faster and cheaper. Two engineers can run the same job
and one finishes in a third of the time and memory because they used mixed precision, gradient
accumulation, and the right parallelism.

This is a concept-check module with hands-on parts you can actually run. The single-GPU
efficiency techniques (mixed precision, gradient accumulation, gradient checkpointing) you will
practice; the multi-GPU parallelism strategies you will understand conceptually and, if you have
access to more than one GPU, try. The judgment is what transfers.

## What you will be able to do

By the end of this module you will be able to:

- Use mixed precision, gradient accumulation, and gradient checkpointing to train larger models
  faster on one GPU.
- Explain data, tensor, and pipeline parallelism and when each is used.
- Explain what makes large training runs expensive and how the techniques reduce cost.
- Use Hugging Face `accelerate` to make a training loop scale across devices.

## Prerequisites

- [Module 2.5](../phase-2-deep-learning/05-making-training-work.md): you can train networks well.
- Access to at least one GPU (a free Colab or Kaggle notebook). Multi-GPU access is helpful for
  the optional parts but not required.

## Curated path

1. **The single-GPU efficiency techniques: Hugging Face's "Methods and tools for efficient
   training on a single GPU."** Mixed precision, gradient accumulation, gradient checkpointing,
   and optimizer choices, all of which you can run yourself.
   https://huggingface.co/docs/transformers/perf_train_gpu_one

2. **Distributed training, conceptually: the Hugging Face Ultra-Scale Playbook.** The current,
   authoritative explanation of how LLMs are trained across GPU clusters, built from thousands of
   real experiments. Read it for the concepts (the parallelism strategies and memory tricks),
   not to reproduce it.
   https://huggingface.co/spaces/nanotron/ultrascale-playbook

3. **Hugging Face `accelerate`.** The practical tool that takes a normal PyTorch training loop and
   makes it run on one GPU, many GPUs, or multiple machines with minimal changes. Read the
   quickstart and adapt one of your earlier training loops.
   https://huggingface.co/docs/accelerate

4. **Chip Huyen, *Designing Machine Learning Systems*, the model-development and training
   sections,** for the systems and cost framing of training at scale.
   https://github.com/chiphuyen/dmls-book

**Deliberately skip for now:** writing custom CUDA kernels and the deepest internals of specific
parallelism libraries (Megatron, DeepSpeed) unless you specialize. Understand the strategies and
practice the single-GPU efficiency tools.

## Background: the ideas in words

**Why scale is hard.** A model and its training state (the weights, their gradients, and the
optimizer's per-parameter state) must fit in GPU memory, and the computation must finish in
reasonable time. Large models break both limits, so you reduce memory per GPU and spread work
across GPUs.

**Single-GPU efficiency.** **Mixed precision** stores and computes in 16-bit where safe, roughly
halving memory and speeding up math, while keeping a 32-bit master copy for stability.
**Gradient accumulation** simulates a large batch by summing gradients over several small
batches before updating, so you get large-batch behavior without large-batch memory.
**Gradient checkpointing** trades compute for memory by not storing all intermediate activations,
recomputing them during the backward pass instead. These three let you train substantially
larger models on the same hardware.

**Parallelism across GPUs.** **Data parallelism** puts a full copy of the model on each GPU, feeds
each a different slice of the batch, and averages the gradients; it is the most common and
scales throughput. When the model itself is too big for one GPU, **tensor parallelism** splits
individual layers (their weight matrices) across GPUs, and **pipeline parallelism** puts
different layers on different GPUs and streams data through like an assembly line. Real
large-scale training combines all three.

**The economics.** Training cost is roughly GPUs times time times price, and the scaling laws
from Module 3.1 let you forecast the benefit. This is why efficiency matters so much: a technique
that cuts memory by 40% can be the difference between a run that fits on affordable hardware and
one that does not, and a speedup is a direct cost saving on a run that may cost real money.

## Knowledge check

1. What three things must fit in GPU memory during training, and which often dominates?
2. Explain mixed precision and why it both saves memory and speeds up training.
3. What does gradient accumulation simulate, and why is it useful on a small GPU?
4. Contrast data, tensor, and pipeline parallelism, and say when each is needed.
5. What does gradient checkpointing trade for what?
6. Why does a 40% memory reduction matter beyond convenience?

## Project

A hands-on efficiency study plus a written scaling analysis:

- Take a training job from an earlier module (your Module 2.6 CNN or a small fine-tune). Measure
  its baseline memory use and time per epoch.
- Apply the single-GPU efficiency techniques one at a time, mixed precision, gradient
  accumulation (to reach a larger effective batch), and gradient checkpointing, and measure the
  effect of each on memory and speed.
- Refactor the loop to use `accelerate` so it would scale to multiple GPUs unchanged.
- Write a short scaling analysis: which technique bought what, and a clear explanation (in your
  own words) of how you would distribute this training across many GPUs and why.

**Definition of done:** the before-and-after efficiency measurements, an `accelerate`-ready
training loop, and the written scaling analysis.

## The workshop: ship it

Build this in its own repository, `modelwright-scaling-study`, using the project habits from
Module 0.2.

1. Set up the project:

```bash
mkdir modelwright-scaling-study && cd modelwright-scaling-study
uv init && uv add torch torchvision accelerate matplotlib && mkdir src
```

2. Write a configurable training script in `src/` where each efficiency technique is a flag, so
   the comparison is clean. Run it on a free GPU notebook with the code committed here.
3. Commit at checkpoints: "Baseline measurements", then "Efficiency techniques compared", then
   "accelerate-ready loop".
4. Add a README with your memory/speed table and the scaling analysis.
5. Ship it:

```bash
gh repo create modelwright-scaling-study --public --source=. --push
```

   (No `gh`? Create an empty public repo, then `git remote add origin <url>` and
   `git push -u origin main`.)

**Done when:** `modelwright-scaling-study` is on GitHub with measured efficiency gains and a clear
written analysis of how the job would scale across GPUs.

## Going deeper (optional)

- Read about DeepSpeed ZeRO and Fully Sharded Data Parallel (FSDP), which shard optimizer state
  and parameters across GPUs.
- If you can access multiple GPUs (some cloud free tiers, or Kaggle's dual-GPU), run a real data
  parallel job with `accelerate`.
- Read about flash attention and why memory-efficient attention matters at scale.

## Canonical references

- Hugging Face, *Efficient training on a single GPU*, *Ultra-Scale Playbook*, and *Accelerate*.
- Huyen, C. *Designing Machine Learning Systems*, model-development chapters.

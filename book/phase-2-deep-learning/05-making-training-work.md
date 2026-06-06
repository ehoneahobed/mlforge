# Module 2.5 — Making Training Work

**Mode:** hybrid
**Gate:** ship gate
**Est. effort:** 10–14 focused hours

> You can now build and train networks. This module is about the difference between a network
> that technically trains and one that trains *well*. It is the craft layer, the accumulated
> tricks that turn a stubborn, diverging model into one that learns fast and generalizes. This
> is some of the most practically valuable material in the whole curriculum.

## Why this matters

Most of the day-to-day skill of a deep learning practitioner lives here. Two people with the
same architecture and data can get wildly different results, and the difference is almost
always in the training details: how the weights were initialized, whether the inputs and
activations were normalized, which optimizer and learning rate were used, how overfitting was
controlled, and how carefully the person debugged when the loss misbehaved.

None of this is glamorous and little of it is in the headline papers, but it is what
separates "I trained a model" from "I trained a model that works." It is also intensely
practical: every choice here has a visible effect on a loss curve you can watch.

## What you will be able to do

By the end of this module you will be able to:

- Choose sensible weight initialization and explain why it matters.
- Apply normalization (batch or layer) and say what problem it solves.
- Choose among optimizers (SGD, momentum, Adam, AdamW) and set a learning rate and schedule.
- Control overfitting with weight decay, dropout, and early stopping.
- Diagnose a misbehaving training run methodically instead of by guessing.

## Prerequisites

- [Module 2.4](./04-pytorch-in-depth.md): you can build and train models in PyTorch.

## Curated path

1. **Karpathy, "A Recipe for Training Neural Networks."** The single best practical guide to
   getting training right and to debugging it when it goes wrong. Read it slowly; it is dense
   with hard-won advice.
   https://karpathy.github.io/2019/04/25/recipe/

2. **Karpathy `makemore`, part 3: "activations & gradients, batchnorm."** A careful, visual
   walk through why initialization and normalization matter, watching activations and
   gradients as they flow. This makes the abstract concrete.
   https://github.com/karpathy/nn-zero-to-hero

3. **CS231n notes: "Neural Networks 2 and 3."** Initialization, batch normalization,
   regularization (L2, dropout), and the practical side of optimization (momentum, Adam,
   learning-rate schedules, babysitting the training process).
   https://cs231n.github.io/neural-networks-2/ and https://cs231n.github.io/neural-networks-3/

4. **Dive into Deep Learning (d2l.ai), the optimization and numerical-stability chapters,** for
   a careful treatment of why initialization and normalization keep gradients healthy, and how
   the optimizers differ.
   https://d2l.ai/

**Deliberately skip for now:** exotic optimizers and the latest normalization variants. Master
the standard toolkit (He init, batch/layer norm, AdamW, cosine or step schedules, weight decay,
dropout, early stopping); it covers the vast majority of real cases.

## Background: the craft, in words

**Initialization.** If weights start too large, activations explode; too small, they vanish,
and gradients with them. Good initialization (He for ReLU, Xavier/Glorot for tanh) keeps the
scale of activations roughly constant across layers so gradients flow. Karpathy's part 3 shows
this happening live.

**Normalization.** Even with good init, the distribution of a layer's inputs shifts during
training, slowing it down. **Batch normalization** renormalizes activations within each
mini-batch; **layer normalization** does it per example (and is what transformers use). Both
let you train deeper networks faster and more stably.

**Optimizers and the learning rate.** Plain SGD works but is slow and sensitive. **Momentum**
accelerates it by accumulating a velocity. **Adam/AdamW** adapt the step size per parameter and
are the common default. The **learning rate** is the single most important hyperparameter: too
high diverges, too low crawls. **Schedules** (warmup then decay, or cosine) help a lot. The
recipe: find the largest learning rate that does not diverge, then decay it.

**Regularization.** To fight overfitting: **weight decay** (penalize large weights, the L2 idea
from Phase 1), **dropout** (randomly zero activations during training so the network cannot
rely on any one unit), and **early stopping** (stop when validation loss stops improving). Use
the validation curve to tell you when each is needed.

**Debugging.** When training misbehaves, do not guess. Karpathy's recipe is the method:
overfit a single batch first (if you cannot, something is broken), check the input data with
your own eyes, verify the loss at initialization is what theory predicts, add complexity one
piece at a time, and watch the curves. Most "the model won't learn" problems are a data bug, a
shape bug, or a learning rate that is wildly off.

## Knowledge check

1. What goes wrong if weights are initialized too large or too small, and what does good
   initialization preserve?
2. What problem does normalization solve, and how do batch norm and layer norm differ?
3. Contrast SGD, momentum, and Adam in one sentence each. What is the practical default?
4. Why is the learning rate the most important hyperparameter, and what does a learning-rate
   schedule do?
5. Name three regularization techniques and what each does.
6. Your training loss will not go down at all. List, in order, the first four things you would
   check.

## Project (ship gate)

**Run a training-improvement study, and document what each change buys you.**

- Start from a deliberately mediocre training setup (a network that trains but underperforms:
  poor init, plain SGD, no normalization, no regularization, a bad learning rate) on a real
  dataset.
- Improve it methodically, one change at a time: fix initialization, add normalization, switch
  to AdamW, tune the learning rate and add a schedule, then add regularization as needed. After
  each change, record the effect on the validation curve.
- Produce an **ablation table or set of curves** showing the contribution of each change, and
  write up which mattered most and why, plus one bug you diagnosed using the recipe's method.

**Definition of done:** the before-and-after results, the per-change ablation, and a written
analysis. This study is itself a strong portfolio piece, because it demonstrates judgment, not
just coding.

## The workshop: ship it

Build this in its own repository, `mlforge-training-recipe`, using the project habits from
Module 0.2.

1. Set up the project:

```bash
mkdir mlforge-training-recipe && cd mlforge-training-recipe
uv init && uv add torch torchvision matplotlib && mkdir src
```

2. Write a configurable training script in `src/` so each improvement is a flag or config
   change, making the ablation clean to run and reproduce.
3. Commit at checkpoints: "Baseline (mediocre) run", then one commit per improvement, then
   "Ablation summary".
4. Add a README with your ablation table/curves and your written analysis.
5. Ship it:

```bash
gh repo create mlforge-training-recipe --public --source=. --push
```

   (No `gh`? Create an empty public repo, then `git remote add origin <url>` and
   `git push -u origin main`.)

**Done when:** `mlforge-training-recipe` is on GitHub with a reproducible ablation showing how
each technique improved training, and an analysis of which mattered most.

## Going deeper (optional)

- Implement batch normalization from scratch (forward and backward) to truly understand it;
  Karpathy's part 3 shows the way.
- Read the original Batch Normalization and Adam papers.
- Explore learning-rate finders and one-cycle schedules (fast.ai popularized these).

## Canonical references

- Karpathy, A. *A Recipe for Training Neural Networks*, and *Neural Networks: Zero to Hero*.
- Stanford CS231n notes, *Neural Networks 2 and 3*.
- Dive into Deep Learning (d2l.ai), optimization and numerical-stability chapters.

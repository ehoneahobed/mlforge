# Module 2.4 — PyTorch in Depth

**Mode:** curated breadth
**Gate:** build gate
**Est. effort:** 8–12 focused hours

> You used PyTorch in Module 2.1 to get a network running, and you built the engine underneath
> it in Modules 2.2 and 2.3. Now you connect the two: you will understand exactly what PyTorch
> is doing, because you have done it by hand, and you will learn to wield it well. This is the
> framework you will use for the rest of the curriculum.

## Why this matters

You now know what an autograd engine and a training loop really are, because you built them.
That puts you in the best possible position to learn PyTorch properly: not as magic, but as a
fast, well-engineered version of the machinery you already understand. The goal of this module
is fluency. You should be able to reach for the right PyTorch construct without friction, read
other people's PyTorch, and know which part of the framework maps to which part of your
from-scratch code.

The payoff is leverage. Every later module (CNNs, transformers, fine-tuning) is written in
PyTorch. Time spent now becoming genuinely comfortable with tensors, `autograd`, `nn.Module`,
and the data pipeline pays back many times over.

## What you will be able to do

By the end of this module you will be able to:

- Manipulate tensors fluently and understand devices (CPU/GPU), dtypes, and shapes.
- Explain how PyTorch's `autograd` corresponds to the engine you built, including `grad_fn`,
  `.backward()`, and `zero_grad()`.
- Build models with `nn.Module`, and feed them data with `Dataset` and `DataLoader`.
- Write a clean, correct training loop with a real optimizer and learning rate.
- Reimplement your from-scratch MLP in idiomatic PyTorch and confirm it matches.

## Prerequisites

- [Module 2.3](./03-neural-nets-backprop-from-scratch.md) (your from-scratch MLP), which this
  module re-expresses in PyTorch.

## PyTorch: the essentials

Here is the mental map, so the tutorials below have somewhere to land. PyTorch is your
from-scratch engine, scaled and hardened:

- **Tensors** are NumPy arrays that also remember how they were computed (for autograd) and
  can live on a GPU. `x = torch.tensor(...)`, `x.shape`, `x.to("cuda")`.
- **Autograd** is exactly your `backward()`. Set `requires_grad=True`, run a forward pass, call
  `loss.backward()`, and every tensor's `.grad` fills in. PyTorch builds the same computation
  graph you built, just automatically and efficiently. Remember to `optimizer.zero_grad()`
  each step, for the same accumulation reason you met in Module 2.2.
- **`nn.Module`** is how you package a model: subclass it, define layers in `__init__`, and the
  forward pass in `forward()`. `nn.Linear`, `nn.ReLU`, and friends are the layers you built.
- **`Dataset` and `DataLoader`** handle batching, shuffling, and loading data efficiently, so
  you do not hand-roll the mini-batch loop.
- **Optimizers** (`torch.optim.SGD`, `torch.optim.Adam`) update the weights; you pass them the
  model's parameters and call `optimizer.step()`.

The canonical training loop, which you will write a hundred times:

```python
for epoch in range(epochs):
    for xb, yb in train_loader:
        optimizer.zero_grad()          # clear old gradients
        preds = model(xb)              # forward
        loss = loss_fn(preds, yb)      # compute loss
        loss.backward()                # backward (autograd)
        optimizer.step()               # update weights
```

That loop is the entire heart of deep learning in PyTorch. Everything else is variations on it.

## Curated path

1. **PyTorch, "Learn the Basics," in full.** You skimmed this in Module 2.1; now work it
   thoroughly, especially the tensors, autograd, `build_model`, and optimization sections.
   https://pytorch.org/tutorials/beginner/basics/intro.html

2. **The autograd mechanics explainer.** Read the official "A Gentle Introduction to
   `torch.autograd`" and notice how `grad_fn` is exactly the `_backward` closures from your
   own engine.
   https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

3. **Karpathy `makemore`, the PyTorch-ification parts.** Watching Karpathy move the same model
   from raw tensors into clean PyTorch modules reinforces the mapping between your code and the
   framework.
   https://github.com/karpathy/nn-zero-to-hero

**Deliberately skip for now:** distributed training, mixed precision, and custom CUDA. Those
are Phase 4. Here, master single-device PyTorch.

## Knowledge check

1. What two things does a PyTorch tensor track that a NumPy array does not?
2. Walk through what `loss.backward()` does, in terms of the engine you built in Module 2.2.
3. Why must you call `optimizer.zero_grad()` each iteration? What bug appears if you forget?
4. What is the role of `nn.Module`, and what goes in `__init__` versus `forward`?
5. What do `Dataset` and `DataLoader` each do for you?
6. In the standard training loop, what is the precise order of the four key calls, and why
   does the order matter?

## Build gate

Reimplement your Module 2.3 MLP in idiomatic PyTorch and confirm parity.

**Specification:**

- The same MLP architecture as Module 2.3, built with `nn.Module` and `nn.Linear`.
- A `Dataset`/`DataLoader` pipeline for MNIST.
- A clean training loop using `torch.optim` and a sensible learning rate, tracking train and
  validation metrics.

**Tests it must pass:**

- **Parity with your from-scratch version.** Trained on the same data with the same
  architecture and learning rate, your PyTorch MLP should reach comparable accuracy to your
  NumPy one. Understand and explain any meaningful gap.
- **Clean structure.** The model is an `nn.Module`; data flows through a `DataLoader`; the loop
  uses an optimizer. No manual gradient code.

## Project

Package it and write a short comparison (reuse your template):

- The PyTorch MLP, trained, with curves.
- A short note mapping each piece of your from-scratch code to its PyTorch equivalent
  (your `Linear.backward` to autograd, your manual loop to the optimizer, and so on), and a
  sentence on what PyTorch gave you that your version did not (speed, GPU, less code).

**Definition of done:** the PyTorch MLP matching your from-scratch accuracy, plus the mapping
write-up.

## The workshop: ship it

Build this in its own repository, `mlforge-pytorch-mlp`, using the project habits from
Module 0.2.

1. Set up the project:

```bash
mkdir mlforge-pytorch-mlp && cd mlforge-pytorch-mlp
uv init && uv add torch torchvision matplotlib && mkdir src
```

2. Write the `nn.Module` model, the `DataLoader` pipeline, and the training loop in `src/`.
3. Commit at checkpoints: "Model and data pipeline", then "Training loop and curves", then
   "Parity with from-scratch MLP".
4. Add a README with your curves and the from-scratch-to-PyTorch mapping note.
5. Ship it:

```bash
gh repo create mlforge-pytorch-mlp --public --source=. --push
```

   (No `gh`? Create an empty public repo, then `git remote add origin <url>` and
   `git push -u origin main`.)

**Done when:** `mlforge-pytorch-mlp` is on GitHub, matches your from-scratch accuracy, and the
README explains how the framework maps to what you built by hand.

## Going deeper (optional)

- Move training to a GPU (a free Google Colab or Kaggle notebook works) and measure the
  speedup.
- Read the d2l.ai chapters that use PyTorch idioms for a second style.
- Explore `torchvision` datasets and transforms, which you will use in Module 2.6.

## Canonical references

- PyTorch documentation, *Learn the Basics* and *torch.autograd*.
- Karpathy, A. *Neural Networks: Zero to Hero*.
